"""
Dynamically generate MCP tools from TDC prompt templates.
"""

import logging
import re

from typing import Any, Optional

from mcp.types import Tool

from txgemma.prompts import PromptTemplate, get_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Placeholder Metadata
# -------------------------

def get_placeholder_type(placeholder: str) -> str:
    """
    Infer JSON schema type for a placeholder.

    Args:
        placeholder: Placeholder name (e.g., 'Drug SMILES')

    Returns:
        JSON schema type ('string', 'number', 'integer', 'boolean')
    """
    placeholder_lower = placeholder.lower()

    # Numeric types
    if any(word in placeholder_lower for word in ["count", "number", "quantity", "index"]):
        return "integer"
    if any(word in placeholder_lower for word in ["dose", "concentration", "score", "value"]):
        return "number"

    # Boolean types
    if any(word in placeholder_lower for word in ["is", "has", "can", "should"]):
        return "boolean"

    # Default to string
    return "string"


def get_placeholder_description(placeholder: str, usage_count: int | None = None) -> str:
    """
    Generate human-readable description for a placeholder.

    Leverages usage statistics to enhance descriptions.

    Args:
        placeholder: Placeholder name
        usage_count: Number of templates using this placeholder (optional)

    Returns:
        Description string
    """
    # Known placeholder descriptions
    descriptions = {
        "Drug SMILES": "SMILES string representation of the drug molecule",
        "Product SMILES": "SMILES string of the product/target molecule",
        "Molecule SMILES": "SMILES string of the molecule",
        "Target sequence": "Amino acid sequence of the target protein",
        "Protein sequence": "Amino acid sequence of the protein",
        "Epitope amino acid sequence": "Amino acid sequence of the epitope region",
        "Indication": "Disease or medical condition being treated",
        "Disease": "Name of the disease or medical condition",
        "Trial phase": "Clinical trial phase (1, 2, or 3)",
        "Phase": "Clinical development phase",
        "Cell line": "Cell line identifier (e.g., HeLa, MCF-7, A549)",
        "Dosage": "Drug dosage amount and unit",
        "Dose": "Administered dose of the drug",
        "Property name": "Name of the molecular property to predict",
        "Target name": "Name or identifier of the biological target",
    }

    # Try exact match first
    if placeholder in descriptions:
        desc = descriptions[placeholder]
    else:
        # Fallback: generate from placeholder name
        desc = placeholder.replace("_", " ").replace("{", "").replace("}", "")
        desc = f"Input parameter: {desc}"

    # Optionally add usage info
    if usage_count and usage_count > 1:
        desc += f" (used in {usage_count} tools)"

    return desc


def get_placeholder_pattern(placeholder: str) -> str | None:
    """
    Get regex pattern for validating placeholder values.

    Args:
        placeholder: Placeholder name

    Returns:
        Regex pattern string, or None if no validation needed
    """
    placeholder_lower = placeholder.lower()

    # SMILES validation (basic)
    if "smiles" in placeholder_lower:
        # Very basic SMILES pattern - just check it has some chemical-like characters
        return r"^[A-Za-z0-9@+\-\[\]\(\)=#$:\.]+$"

    # Amino acid sequence (single-letter codes)
    if "sequence" in placeholder_lower or "epitope" in placeholder_lower:
        return r"^[ACDEFGHIKLMNPQRSTVWY]+$"

    # Trial/Phase numbers
    if "phase" in placeholder_lower:
        return r"^[1-3]$"

    return None

def normalize_parameter_name(placeholder: str) -> str:
    """
    Convert TDC placeholder to valid Python/Pydantic identifier.
    
    Args:
        placeholder: Original placeholder name (e.g., "Drug SMILES")
    
    Returns:
        Normalized name (e.g., "drug_smiles")
    
    Examples:
        >>> normalize_parameter_name("Drug SMILES")
        'drug_smiles'
        >>> normalize_parameter_name("Target sequence")
        'target_sequence'
        >>> normalize_parameter_name("Trial phase")
        'trial_phase'
        >>> normalize_parameter_name("Epitope amino acid sequence")
        'epitope_amino_acid_sequence'
    """
    # Convert to lowercase and replace spaces with underscores
    normalized = placeholder.lower().replace(" ", "_")
    
    # Replace other non-alphanumeric characters with underscores
    normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized)
    
    # Remove consecutive underscores
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    return normalized

def get_parameter_mapping(
    templates: dict[str, PromptTemplate] | None = None
) -> dict[str, str]:
    """
    Get mapping from normalized names back to original placeholder names.
    
    This is used when executing tools - we receive normalized parameter names
    from the MCP client, but need to substitute original names in the prompt.
    
    Args:
        templates: Optional dict of templates. If None, loads all templates.
    
    Returns:
        Dictionary mapping normalized_name -> original_placeholder_name
        
    Example:
        {
            'drug_smiles': 'Drug SMILES',
            'target_sequence': 'Target sequence',
            'trial_phase': 'Trial phase',
            ...
        }
    """
    if templates is None:
        from txgemma.prompts import get_loader
        loader = get_loader()
        templates = loader.all()
    
    mapping = {}
    
    # Collect all unique placeholders across all templates
    for template in templates.values():
        for placeholder in template.placeholders:
            normalized = normalize_parameter_name(placeholder)
            mapping[normalized] = placeholder
    
    return mapping


# Global mapping - computed once at module load
_PARAMETER_MAPPING = None

def get_cached_parameter_mapping() -> dict[str, str]:
    """Get cached parameter mapping (computed once)."""
    global _PARAMETER_MAPPING
    if _PARAMETER_MAPPING is None:
        _PARAMETER_MAPPING = get_parameter_mapping()
    return _PARAMETER_MAPPING

# -------------------------
# Tool Building
# -------------------------

def _build_description_from_prompt(tool_name: str, prompt_text: str) -> str:
    """
    Build tool description, ending after the Question line.
    
    The prompt structure is:
    - Instructions: ...
    - Context: ...
    - Question: ... (may be multi-line for classifications starting with "(A)")
    - [Parameter lines or Answer: - we exclude these]
    
    For classifications, the Question is multi-line:
      Question: Given a drug SMILES string, predict whether it
      (A) option 1 (B) option 2
    
    For numeric predictions, the Question is single-line:
      Question: Given a drug SMILES string, predict from 000 to 1000...
    
    We include the "(A)..." line if present, otherwise stop after Question line.
    
    Args:
        tool_name: MCP tool name
        prompt_text: Complete prompt template text
        
    Returns:
        Formatted description ending with the Question (and classification options if present)
    """
    formatted_name = tool_name.replace('_', ' ')
    
    # Find the Question line
    question_match = re.search(r'(.*Question:.*?)(\n|$)', prompt_text, re.DOTALL)
    
    if not question_match:
        # No Question found - fallback to full prompt cleaned up
        cleaned_prompt = prompt_text
        cleaned_prompt = re.sub(r'\s*Answer:\s*$', '', cleaned_prompt, flags=re.IGNORECASE)
        return f"**{formatted_name}**\n\n{cleaned_prompt.strip()}"
    
    # Get everything up to and including the Question line
    prompt_until_question = question_match.group(1)
    
    # Get the text after the Question line
    remaining_text = prompt_text[question_match.end():]
    
    # Check if the next line starts with "(A)" - if so, it's classification options
    classification_match = re.match(r'^\s*(\(A\).*?)(\n|$)', remaining_text)
    
    if classification_match:
        # Include the classification line
        classification_line = classification_match.group(1)
        cleaned_prompt = f"{prompt_until_question}\n{classification_line}"
    else:
        # No classification - just use up to Question line
        cleaned_prompt = prompt_until_question
    
    return f"**{formatted_name}**\n\n{cleaned_prompt.strip()}"


# -------------------------
# Tool Building
# -------------------------

def build_tool_from_template(
    template: PromptTemplate,
    placeholder_stats: dict[str, int] | None = None,
) -> Tool:
    """
    Build an MCP Tool from a prompt template.
    
    Uses normalized parameter names (e.g., "drug_smiles") for the schema,
    but preserves original names (e.g., "Drug SMILES") in title for display.
    
    The tool description includes the FULL prompt text so users/agents can see:
    - Instructions
    - Context (what the metric/task means)
    - Question format (input/output specification)
    - Expected answer format (numeric ranges, classification options, etc.)
    """
    properties = {}
    required = []
    
    for placeholder in template.placeholders:
        # Normalize the parameter name for the schema
        param_name = normalize_parameter_name(placeholder)
        required.append(param_name)
        
        usage_count = placeholder_stats.get(placeholder) if placeholder_stats else None
        
        prop_schema = {
            "type": get_placeholder_type(placeholder),
            "description": get_placeholder_description(placeholder, usage_count),
            "title": placeholder,  # ← Original name for MCP Inspector display
        }
        
        # Add pattern validation if available
        pattern = get_placeholder_pattern(placeholder)
        if pattern:
            prop_schema["pattern"] = pattern
        
        properties[param_name] = prop_schema
    
    # Build description from full prompt text
    # This ensures users/agents see all context, output formats, etc.
    description = _build_description_from_prompt(template.name, template.template)
    
    # Create the tool with normalized parameter names
    tool = Tool(
        name=template.name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )
    
    return tool

def build_tools(
    *,
    filter_placeholder: str | None = None,
    filter_placeholders: list[str] | None = None,
    match_all: bool = True,
    exact_match: bool = True,
    exclude_complex: bool = False,
    max_placeholders: int | None = None,
    exclude_name_pattern: str | None = None,
) -> list[Tool]:
    """
    Build MCP tools from TDC prompt definitions with flexible filtering.

    Args:
        filter_placeholder: Only build tools using this placeholder (e.g., "Drug SMILES")
        filter_placeholders: Only build tools using these placeholders
        match_all: If True, tool must use ALL placeholders. If False, ANY.
        exact_match: If True, exact placeholder match. If False, fuzzy substring match.
        exclude_complex: If True, skip tools with many placeholders (>2)
        max_placeholders: Maximum number of placeholders per tool (None = no limit)
        exclude_name_pattern: Regex pattern to exclude tools by name
                             (e.g., "^ToxCast" to exclude all ToxCast tools)

    Returns:
        List of MCP Tool objects

    Examples:
        # All tools
        >>> build_tools()

        # Only Drug SMILES tools
        >>> build_tools(filter_placeholder="Drug SMILES")

        # Only drug-target interaction tools
        >>> build_tools(
        ...     filter_placeholders=["Drug SMILES", "Target sequence"],
        ...     match_all=True
        ... )

        # Simple tools only (≤2 placeholders)
        >>> build_tools(max_placeholders=2)

        # Any sequence-related tools (fuzzy match)
        >>> build_tools(filter_placeholder="sequence", exact_match=False)
        
        # Exclude all ToxCast tools
        >>> build_tools(exclude_name_pattern="^ToxCast")
        
        # Drug SMILES tools, excluding ToxCast, simple only
        >>> build_tools(
        ...     filter_placeholder="Drug SMILES",
        ...     max_placeholders=1,
        ...     exclude_name_pattern="^ToxCast"
        ... )
    """
    loader = get_loader()

    # Compile exclusion regex pattern if provided
    exclude_pattern = None
    if exclude_name_pattern:
        try:
            exclude_pattern = re.compile(exclude_name_pattern)
            logger.info(f"Excluding tools matching pattern: {exclude_name_pattern}")
        except re.error as e:
            logger.error(f"Invalid exclude pattern '{exclude_name_pattern}': {e}")
            raise ValueError(f"Invalid regex pattern: {exclude_name_pattern}") from e

    # Get placeholder statistics for better descriptions
    placeholder_stats = loader.placeholder_stats()

    # Apply placeholder filters to get template subset
    if filter_placeholder:
        templates = loader.filter_by_placeholder(filter_placeholder, exact=exact_match)
    elif filter_placeholders:
        templates = loader.filter_by_placeholders(filter_placeholders, match_all=match_all)
    else:
        templates = loader.all()

    # Apply name exclusion filter (before complexity filtering for efficiency)
    excluded_count = 0
    if exclude_pattern:
        filtered_templates = {}
        for name, template in templates.items():
            if exclude_pattern.search(name):
                logger.debug(f"Excluding tool '{name}' (matches exclude pattern)")
                excluded_count += 1
            else:
                filtered_templates[name] = template
        templates = filtered_templates

    # Apply complexity filters
    if max_placeholders is not None:
        templates = {
            name: tmpl
            for name, tmpl in templates.items()
            if tmpl.placeholder_count() <= max_placeholders
        }
    elif exclude_complex:
        # Default threshold for "complex" is >2 placeholders
        templates = {
            name: tmpl for name, tmpl in templates.items() if tmpl.placeholder_count() <= 2
        }

    # Build tools
    tools = []
    for name, template in templates.items():
        try:
            tool = build_tool_from_template(template, placeholder_stats)
            tools.append(tool)
            logger.debug(
                f"Built tool: {name} "
                f"({len(template.placeholders)} parameter{'s' if len(template.placeholders) != 1 else ''})"
            )
        except Exception as e:
            logger.error(f"Failed to build tool '{name}': {e}")

    # Log summary
    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} tool(s) matching pattern '{exclude_name_pattern}'")

    logger.info(
        f"Built {len(tools)} tools "
        f"(filter_placeholder={filter_placeholder}, "
        f"filter_placeholders={filter_placeholders}, "
        f"max_placeholders={max_placeholders}, "
        f"exclude_pattern={exclude_name_pattern}) "
        f"from {len(loader.all())} total"
    )

    return tools


def get_tool_names(
    *,
    filter_placeholder: str | None = None,
    filter_placeholders: list[str] | None = None,
    match_all: bool = True,
) -> list[str]:
    """
    Get list of tool names with optional filtering.

    Lightweight alternative to build_tools() when you only need names.

    Args:
        filter_placeholder: Only include tools using this placeholder
        filter_placeholders: Only include tools using these placeholders
        match_all: If True, tool must use ALL placeholders. If False, ANY.

    Returns:
        List of tool names
    """
    loader = get_loader()

    if filter_placeholder:
        templates = loader.filter_by_placeholder(filter_placeholder)
    elif filter_placeholders:
        templates = loader.filter_by_placeholders(filter_placeholders, match_all=match_all)
    else:
        templates = loader.all()

    return list(templates.keys())


# -------------------------
# Tool Introspection
# -------------------------

def analyze_tools() -> dict[str, Any]:
    """
    Analyze all available tools and return statistics.

    Returns:
        Dictionary with tool analysis:
        - total_tools: Total number of tools
        - total_placeholders: Total unique placeholders
        - placeholder_usage: Dict of placeholder -> usage count
        - tools_by_complexity: Dict of placeholder_count -> tool_count
        - most_common_placeholders: Top 10 placeholders
    """
    loader = get_loader()

    all_templates = loader.all()
    placeholder_stats = loader.placeholder_stats()

    # Group by complexity
    tools_by_complexity = {}
    for template in all_templates.values():
        count = template.placeholder_count()
        tools_by_complexity[count] = tools_by_complexity.get(count, 0) + 1

    return {
        "total_tools": len(all_templates),
        "total_placeholders": len(placeholder_stats),
        "placeholder_usage": placeholder_stats,
        "tools_by_complexity": tools_by_complexity,
        "most_common_placeholders": loader.most_common_placeholders(10),
        "simple_tools": len([t for t in all_templates.values() if t.placeholder_count() <= 2]),
        "complex_tools": len([t for t in all_templates.values() if t.placeholder_count() > 2]),
    }


def suggest_tool_subsets() -> dict[str, list[str]]:
    """
    Suggest useful subsets of tools based on common use cases.

    Returns:
        Dictionary mapping use case -> list of tool names
    """

    return {
        "drug_discovery": get_tool_names(filter_placeholder="Drug SMILES"),
        "protein_analysis": get_tool_names(filter_placeholder="sequence"),
        "simple_predictions": get_tool_names(filter_placeholders=["Drug SMILES"], match_all=True),
        "drug_target_interaction": get_tool_names(
            filter_placeholders=["Drug SMILES", "Target sequence"], match_all=True
        ),
    }