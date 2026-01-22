"""
Dynamically generate MCP tools from TDC prompt templates.
"""

import logging
from typing import Any

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


# -------------------------
# Tool Building
# -------------------------


def build_tool_from_template(
    template: PromptTemplate,
    placeholder_stats: dict[str, int] | None = None,
) -> Tool:
    """
    Build an MCP Tool from a prompt template.

    Args:
        template: PromptTemplate instance
        placeholder_stats: Optional placeholder usage statistics for better descriptions

    Returns:
        MCP Tool object with full schema
    """
    # Build input schema from placeholders
    properties = {}

    for placeholder in template.placeholders:
        usage_count = placeholder_stats.get(placeholder) if placeholder_stats else None

        prop_schema = {
            "type": get_placeholder_type(placeholder),
            "description": get_placeholder_description(placeholder, usage_count),
        }

        # Add pattern validation if available
        pattern = get_placeholder_pattern(placeholder)
        if pattern:
            prop_schema["pattern"] = pattern

        properties[placeholder] = prop_schema

    # Create the tool with full schema
    tool = Tool(
        name=template.name,
        description=template.get_description(),
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": template.placeholders,
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
) -> list[Tool]:
    """
    Build MCP tools from TDC prompt definitions with flexible filtering.

    Args:
        filter_placeholder: Only build tools using this placeholder (e.g., "Drug SMILES")
        filter_placeholders: Only build tools using these placeholders
        match_all: If True, tool must use ALL placeholders. If False, ANY.
        exact_match: If True, exact placeholder match. If False, fuzzy substring match.
        exclude_complex: If True, skip tools with many placeholders
        max_placeholders: Maximum number of placeholders per tool (None = no limit)

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
    """
    loader = get_loader()

    # Get placeholder statistics for better descriptions
    placeholder_stats = loader.placeholder_stats()

    # Apply filters to get template subset
    if filter_placeholder:
        templates = loader.filter_by_placeholder(filter_placeholder, exact=exact_match)
    elif filter_placeholders:
        templates = loader.filter_by_placeholders(filter_placeholders, match_all=match_all)
    else:
        templates = loader.all()

    # Apply complexity filter
    if max_placeholders is not None:
        templates = {
            name: tmpl
            for name, tmpl in templates.items()
            if tmpl.placeholder_count() <= max_placeholders
        }
    elif exclude_complex:
        # Default threshold for "complex"
        templates = {
            name: tmpl for name, tmpl in templates.items() if tmpl.placeholder_count() <= 2
        }

    # Build tools
    tools = []
    for name, template in templates.items():
        try:
            tool = build_tool_from_template(template, placeholder_stats)
            tools.append(tool)
            logger.info(
                f"Built tool: {name} "
                f"({len(template.placeholders)} parameter{'s' if len(template.placeholders) != 1 else ''})"
            )
        except Exception as e:
            logger.error(f"Failed to build tool '{name}': {e}")

    logger.info(f"Successfully built {len(tools)} tools (filtered from {len(loader.all())} total)")
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
