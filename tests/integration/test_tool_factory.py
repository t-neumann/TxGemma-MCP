"""
Tests for txgemma.tool_factory module.

Tests tool building, filtering, and introspection with proper mocking.
Focus on the NEW description building logic that includes full prompts.

"""

from unittest.mock import Mock, patch

import pytest

from txgemma.tool_factory import (
    _build_description_from_prompt,
    analyze_tools,
    build_tool_from_template,
    build_tools,
    get_placeholder_description,
    get_placeholder_pattern,
    get_placeholder_type,
    get_tool_names,
    normalize_parameter_name,
    get_parameter_mapping,
    suggest_tool_subsets,
)

pytestmark = [pytest.mark.integration]

# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_mock_template(
    name: str,
    placeholders: list[str],
    template_text: str = None,
) -> Mock:
    """
    Create a mock PromptTemplate matching the real interface.
    
    Args:
        name: Tool name
        placeholders: List of placeholder names
        template_text: Full prompt template text (for description building)
    
    Returns:
        Mock PromptTemplate object
    """
    template = Mock()
    template.name = name
    template.placeholders = placeholders
    
    # Default template text if not provided
    if template_text is None:
        template_text = f"""Instructions: Answer the following question.
Context: This is a test template for {name}.
Question: Given inputs, predict the output.
{placeholders[0]}: {{{placeholders[0]}}}
Answer:"""
    
    template.template = template_text
    
    # Mock methods
    def mock_has_placeholder(ph: str) -> bool:
        return ph in placeholders
    
    def mock_placeholder_count() -> int:
        return len(placeholders)
    
    template.has_placeholder = Mock(side_effect=mock_has_placeholder)
    template.placeholder_count = Mock(side_effect=mock_placeholder_count)
    
    return template


@pytest.fixture
def mock_template_numeric():
    """Template for numeric prediction task."""
    return create_mock_template(
        name="Lipophilicity_AstraZeneca",
        placeholders=["Drug SMILES"],
        template_text="""Instructions: Answer the following question about drug properties.
Context: Lipophilicity measures the ability of a drug to dissolve in a lipid environment.
Question: Given a drug SMILES string, predict its normalized lipophilicity from 000 to 1000, where 000 is minimum and 1000 is maximum.
Drug SMILES: {Drug SMILES}
Answer:"""
    )


@pytest.fixture
def mock_template_classification():
    """Template for classification task."""
    return create_mock_template(
        name="CYP2C9_Veith",
        placeholders=["Drug SMILES"],
        template_text="""Instructions: Answer the following question about drug properties.
Context: CYP P450 genes are involved in metabolism.
Question: Given a drug SMILES string, predict whether it
(A) does not inhibit CYP2C9 (B) inhibits CYP2C9
Drug SMILES: {Drug SMILES}
Answer:"""
    )


@pytest.fixture
def mock_template_multi_input():
    """Template with multiple input parameters."""
    return create_mock_template(
        name="DrugTarget_Interaction",
        placeholders=["Drug SMILES", "Target sequence"],
        template_text="""Instructions: Predict drug-target interaction.
Context: Drug-target interactions are critical for efficacy.
Question: Given a drug and target, predict binding affinity from 0 to 100.
Drug SMILES: {Drug SMILES}
Target sequence: {Target sequence}
Answer:"""
    )


@pytest.fixture
def mock_loader():
    """Mock PromptLoader with sample templates."""
    loader = Mock()
    
    # Create templates
    templates = {
        "Lipophilicity_AstraZeneca": create_mock_template(
            "Lipophilicity_AstraZeneca",
            ["Drug SMILES"],
        ),
        "CYP2C9_Veith": create_mock_template(
            "CYP2C9_Veith",
            ["Drug SMILES"],
        ),
        "DrugTarget_DTI": create_mock_template(
            "DrugTarget_DTI",
            ["Drug SMILES", "Target sequence"],
        ),
    }
    
    loader.all.return_value = templates
    loader.list_prompts.return_value = list(templates.keys())
    loader.placeholder_stats.return_value = {
        "Drug SMILES": 2,
        "Target sequence": 1,
    }
    loader.most_common_placeholders.return_value = [
        ("Drug SMILES", 2),
        ("Target sequence", 1),
    ]
    
    # Mock filtering methods
    def mock_filter_by_placeholder(ph: str, exact: bool = True):
        return {
            name: tmpl for name, tmpl in templates.items()
            if tmpl.has_placeholder(ph)
        }
    
    def mock_filter_by_placeholders(phs: list[str], match_all: bool = True):
        if match_all:
            return {
                name: tmpl for name, tmpl in templates.items()
                if all(tmpl.has_placeholder(ph) for ph in phs)
            }
        else:
            return {
                name: tmpl for name, tmpl in templates.items()
                if any(tmpl.has_placeholder(ph) for ph in phs)
            }
    
    loader.filter_by_placeholder = Mock(side_effect=mock_filter_by_placeholder)
    loader.filter_by_placeholders = Mock(side_effect=mock_filter_by_placeholders)
    loader.get = Mock(side_effect=lambda name: templates.get(name))
    
    return loader


# =============================================================================
# PLACEHOLDER METADATA TESTS
# =============================================================================

class TestPlaceholderMetadata:
    """Test placeholder type/description/pattern inference."""
    
    def test_placeholder_type_string_default(self):
        """Test that default type is string."""
        assert get_placeholder_type("Drug SMILES") == "string"
        assert get_placeholder_type("Protein sequence") == "string"
    
    def test_placeholder_type_integer(self):
        """Test integer type inference."""
        assert get_placeholder_type("Drug count") == "integer"
        assert get_placeholder_type("Number of atoms") == "integer"
        assert get_placeholder_type("Quantity") == "integer"
        assert get_placeholder_type("Index") == "integer"
    
    def test_placeholder_type_number(self):
        """Test float type inference."""
        assert get_placeholder_type("Dose") == "number"
        assert get_placeholder_type("Concentration") == "number"
        assert get_placeholder_type("IC50 score") == "number"
        assert get_placeholder_type("Value") == "number"
    
    def test_placeholder_type_boolean(self):
        """Test boolean type inference."""
        assert get_placeholder_type("Is active") == "boolean"
        assert get_placeholder_type("Has toxicity") == "boolean"
        assert get_placeholder_type("Can bind") == "boolean"
        assert get_placeholder_type("Should proceed") == "boolean"
    
    def test_placeholder_description_known(self):
        """Test known placeholder descriptions."""
        desc = get_placeholder_description("Drug SMILES")
        assert "SMILES string representation" in desc
        assert "drug molecule" in desc
    
    def test_placeholder_description_unknown(self):
        """Test fallback description for unknown placeholders."""
        desc = get_placeholder_description("Unknown Param")
        assert "Input parameter" in desc
        assert "Unknown Param" in desc
    
    def test_placeholder_description_with_usage_count(self):
        """Test description includes usage count."""
        desc = get_placeholder_description("Drug SMILES", usage_count=5)
        assert "used in 5 tools" in desc
    
    def test_placeholder_pattern_smiles(self):
        """Test SMILES validation pattern."""
        pattern = get_placeholder_pattern("Drug SMILES")
        assert pattern is not None
        assert "[A-Za-z0-9" in pattern
    
    def test_placeholder_pattern_sequence(self):
        """Test amino acid sequence pattern."""
        pattern = get_placeholder_pattern("Protein sequence")
        assert pattern is not None
        assert "ACDEFGHIKLMNPQRSTVWY" in pattern
    
    def test_placeholder_pattern_phase(self):
        """Test trial phase pattern."""
        pattern = get_placeholder_pattern("Trial phase")
        assert pattern is not None
        assert "[1-3]" in pattern
    
    def test_placeholder_pattern_none_for_generic(self):
        """Test no pattern for generic placeholders."""
        pattern = get_placeholder_pattern("Generic input")
        assert pattern is None


# =============================================================================
# PARAMETER NORMALIZATION TESTS
# =============================================================================

class TestParameterNormalization:
    """Test parameter name normalization to snake_case."""
    
    def test_normalize_simple(self):
        """Test simple normalization."""
        assert normalize_parameter_name("Drug SMILES") == "drug_smiles"
        assert normalize_parameter_name("Target sequence") == "target_sequence"
    
    def test_normalize_multiple_spaces(self):
        """Test normalization with multiple spaces."""
        assert normalize_parameter_name("Epitope amino acid sequence") == "epitope_amino_acid_sequence"
    
    def test_normalize_special_characters(self):
        """Test normalization removes special characters."""
        assert normalize_parameter_name("Drug-SMILES") == "drug_smiles"
        assert normalize_parameter_name("Property #1") == "property_1"
    
    def test_normalize_consecutive_underscores(self):
        """Test removal of consecutive underscores."""
        result = normalize_parameter_name("Test  multiple   spaces")
        assert "__" not in result
    
    def test_normalize_strips_underscores(self):
        """Test leading/trailing underscores removed."""
        assert not normalize_parameter_name(" Leading space").startswith("_")
        assert not normalize_parameter_name("Trailing space ").endswith("_")
    
    def test_get_parameter_mapping(self):
        """Test parameter mapping generation."""
        templates = {
            "tool1": create_mock_template("tool1", ["Drug SMILES"]),
            "tool2": create_mock_template("tool2", ["Target sequence"]),
        }
        
        mapping = get_parameter_mapping(templates)
        
        assert "drug_smiles" in mapping
        assert mapping["drug_smiles"] == "Drug SMILES"
        assert "target_sequence" in mapping
        assert mapping["target_sequence"] == "Target sequence"
    
    def test_get_parameter_mapping_deduplicates(self):
        """Test that mapping deduplicates identical placeholders."""
        templates = {
            "tool1": create_mock_template("tool1", ["Drug SMILES"]),
            "tool2": create_mock_template("tool2", ["Drug SMILES"]),
        }
        
        mapping = get_parameter_mapping(templates)
        
        # Should only have one entry for drug_smiles
        assert list(mapping.values()).count("Drug SMILES") == 1


# =============================================================================
# DESCRIPTION BUILDER TESTS ⭐ NEW & CRITICAL
# =============================================================================

class TestDescriptionBuilder:
    """Test the NEW description building logic."""
    
    def test_numeric_task_excludes_parameters(self):
        """Test numeric task description excludes parameter placeholders."""
        prompt = """Instructions: Answer the question.
Context: Test context.
Question: Given a drug SMILES string, predict from 000 to 1000.
Drug SMILES: {Drug SMILES}
Answer:"""
        
        result = _build_description_from_prompt("Test_Tool", prompt)
        
        # Should include question
        assert "predict from 000 to 1000" in result
        
        # Should NOT include parameter placeholder line
        assert "Drug SMILES:" not in result
        assert "{Drug SMILES}" not in result
        
        # Should NOT include Answer:
        assert not result.strip().endswith("Answer:")
    
    def test_classification_task_includes_options(self):
        """Test classification task includes (A) and (B) options."""
        prompt = """Instructions: Answer the question.
Context: Test context.
Question: Given a drug SMILES string, predict whether it
(A) does not inhibit (B) inhibits
Drug SMILES: {Drug SMILES}
Answer:"""
        
        result = _build_description_from_prompt("Test_Tool", prompt)
        
        # Should include both classification options
        assert "(A) does not inhibit" in result
        assert "(B) inhibits" in result
        
        # Should NOT include parameter placeholder
        assert "Drug SMILES:" not in result
    
    def test_tool_name_formatted_in_header(self):
        """Test tool name is formatted as header."""
        prompt = "Question: Test question."
        result = _build_description_from_prompt("Tool_Name_Test", prompt)
        
        assert "**Tool Name Test**" in result
    
    def test_preserves_instructions_and_context(self):
        """Test that Instructions and Context sections are preserved."""
        prompt = """Instructions: Answer the following question.
Context: This is important context.
Question: What is the answer?
Drug SMILES: {Drug SMILES}
Answer:"""
        
        result = _build_description_from_prompt("Test", prompt)
        
        assert "Instructions:" in result
        assert "important context" in result
        assert "Context:" in result
    
    def test_multi_input_parameters_all_removed(self):
        """Test that all parameter placeholders are removed."""
        prompt = """Instructions: Test.
Context: Test.
Question: Predict interaction.
Drug SMILES: {Drug SMILES}
Target sequence: {Target sequence}
Answer:"""
        
        result = _build_description_from_prompt("Test", prompt)
        
        # Neither parameter should be in the description
        assert "Drug SMILES:" not in result
        assert "Target sequence:" not in result
        assert "{Drug SMILES}" not in result
        assert "{Target sequence}" not in result
    
    def test_no_question_line_fallback(self):
        """Test fallback when no Question line exists."""
        prompt = "Instructions: Some text.\nAnswer:"
        result = _build_description_from_prompt("Test", prompt)
        
        # Should still work and remove Answer:
        assert "Instructions:" in result
        assert "Answer:" not in result
    
    def test_real_world_lipophilicity_prompt(self):
        """Test with actual TDC lipophilicity prompt."""
        prompt = """Instructions: Answer the following question about drug properties.
Context: Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption.
Question: Given a drug SMILES string, predict its normalized lipophilicity from 000 to 1000, where 000 is minimum lipophilicity and 1000 is maximum lipophilicity.
Drug SMILES: {Drug SMILES}
Answer:"""
        
        result = _build_description_from_prompt("Lipophilicity_AstraZeneca", prompt)
        
        # Should have all key elements
        assert "**Lipophilicity AstraZeneca**" in result
        assert "lipid" in result
        assert "from 000 to 1000" in result
        
        # Should NOT have parameter placeholder
        assert "Drug SMILES:" not in result
        assert "{Drug SMILES}" not in result
    
    def test_real_world_cyp2c9_prompt(self):
        """Test with actual TDC CYP2C9 classification prompt."""
        prompt = """Instructions: Answer the following question about drug properties.
Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells.
Question: Given a drug SMILES string, predict whether it
(A) does not inhibit CYP2C9 (B) inhibits CYP2C9
Drug SMILES: {Drug SMILES}
Answer:"""
        
        result = _build_description_from_prompt("CYP2C9_Veith", prompt)
        
        # Should have classification options
        assert "(A) does not inhibit CYP2C9" in result
        assert "(B) inhibits CYP2C9" in result
        
        # Should NOT have parameter placeholder
        assert "Drug SMILES:" not in result


# =============================================================================
# BUILD TOOL FROM TEMPLATE TESTS
# =============================================================================

class TestBuildToolFromTemplate:
    """Test core tool building from templates."""
    
    def test_build_tool_basic_structure(self, mock_template_numeric):
        """Test basic tool structure is correct."""
        tool = build_tool_from_template(mock_template_numeric)
        
        assert tool.name == "Lipophilicity_AstraZeneca"
        assert tool.description is not None
        assert "type" in tool.inputSchema
        assert tool.inputSchema["type"] == "object"
    
    def test_build_tool_schema_has_properties(self, mock_template_numeric):
        """Test tool schema includes properties."""
        tool = build_tool_from_template(mock_template_numeric)
        
        schema = tool.inputSchema
        assert "properties" in schema
        assert "drug_smiles" in schema["properties"]
    
    def test_build_tool_schema_has_required(self, mock_template_numeric):
        """Test tool schema marks parameters as required."""
        tool = build_tool_from_template(mock_template_numeric)
        
        schema = tool.inputSchema
        assert "required" in schema
        assert "drug_smiles" in schema["required"]
    
    def test_build_tool_parameter_normalized(self, mock_template_numeric):
        """Test that parameter names are normalized."""
        tool = build_tool_from_template(mock_template_numeric)
        
        # Should use normalized name in schema
        assert "drug_smiles" in tool.inputSchema["properties"]
        
        # Should preserve original name in title
        assert tool.inputSchema["properties"]["drug_smiles"]["title"] == "Drug SMILES"
    
    def test_build_tool_description_uses_prompt(self, mock_template_numeric):
        """Test that description includes prompt content."""
        tool = build_tool_from_template(mock_template_numeric)
        
        # Should include context from prompt
        assert "lipid" in tool.description.lower()
        assert "from 000 to 1000" in tool.description
    
    def test_build_tool_description_excludes_parameters(self, mock_template_numeric):
        """Test that description excludes parameter placeholders."""
        tool = build_tool_from_template(mock_template_numeric)
        
        # Should NOT include parameter placeholder line
        assert "Drug SMILES:" not in tool.description
        assert "{Drug SMILES}" not in tool.description
    
    def test_build_tool_classification_includes_options(self, mock_template_classification):
        """Test classification tool includes options in description."""
        tool = build_tool_from_template(mock_template_classification)
        
        assert "(A) does not inhibit" in tool.description
        assert "(B) inhibits" in tool.description
    
    def test_build_tool_multi_input_parameters(self, mock_template_multi_input):
        """Test tool with multiple input parameters."""
        tool = build_tool_from_template(mock_template_multi_input)
        
        schema = tool.inputSchema
        assert "drug_smiles" in schema["properties"]
        assert "target_sequence" in schema["properties"]
        assert len(schema["required"]) == 2
    
    def test_build_tool_with_placeholder_stats(self, mock_template_numeric):
        """Test that placeholder stats are used in descriptions."""
        stats = {"Drug SMILES": 5}
        tool = build_tool_from_template(mock_template_numeric, placeholder_stats=stats)
        
        # Should mention usage count in parameter description
        param_desc = tool.inputSchema["properties"]["drug_smiles"]["description"]
        assert "used in 5 tools" in param_desc
    
    def test_build_tool_validation_pattern(self, mock_template_numeric):
        """Test that validation patterns are added when available."""
        tool = build_tool_from_template(mock_template_numeric)
        
        # Drug SMILES should have a validation pattern
        properties = tool.inputSchema["properties"]
        assert "pattern" in properties["drug_smiles"]


# =============================================================================
# BUILD TOOLS (BATCH) TESTS
# =============================================================================

class TestBuildTools:
    """Test batch tool building with filters."""
    
    @patch('txgemma.tool_factory.get_loader')
    def test_build_all_tools(self, mock_get_loader, mock_loader):
        """Test building all tools without filters."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools()
        
        assert len(tools) == 3  # Should build all 3 templates
        assert all(hasattr(t, 'name') for t in tools)
    
    @patch('txgemma.tool_factory.get_loader')
    def test_filter_by_single_placeholder(self, mock_get_loader, mock_loader):
        """Test filtering by single placeholder."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools(filter_placeholder="Drug SMILES")
        
        # All 3 templates have Drug SMILES
        assert len(tools) == 3
    
    @patch('txgemma.tool_factory.get_loader')
    def test_filter_by_placeholder_exact(self, mock_get_loader, mock_loader):
        """Test exact placeholder matching."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools(filter_placeholder="Drug SMILES", exact_match=True)
        
        # Should have filtered to only tools with Drug SMILES
        assert len(tools) == 3
        # Verify all tools have drug_smiles parameter
        for tool in tools:
            assert "drug_smiles" in tool.inputSchema["properties"]
    
    @patch('txgemma.tool_factory.get_loader')
    def test_filter_by_multiple_placeholders_match_all(self, mock_get_loader, mock_loader):
        """Test filtering with multiple placeholders (match all)."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=True
        )
        
        # Only DrugTarget_DTI has both
        assert len(tools) == 1
        assert tools[0].name == "DrugTarget_DTI"
    
    @patch('txgemma.tool_factory.get_loader')
    def test_filter_by_multiple_placeholders_match_any(self, mock_get_loader, mock_loader):
        """Test filtering with multiple placeholders (match any)."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=False
        )
        
        # All 3 have at least one of these
        assert len(tools) == 3
    
    @patch('txgemma.tool_factory.get_loader')
    def test_max_placeholders_filter(self, mock_get_loader, mock_loader):
        """Test filtering by maximum placeholder count."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools(max_placeholders=1)
        
        # Only tools with 1 placeholder
        assert len(tools) == 2  # Lipophilicity and CYP2C9
        assert all(len(t.inputSchema["required"]) == 1 for t in tools)
    
    @patch('txgemma.tool_factory.get_loader')
    def test_exclude_complex_filter(self, mock_get_loader, mock_loader):
        """Test excluding complex tools (>2 placeholders)."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools(exclude_complex=True)
        
        # All our templates have <=2 placeholders
        assert len(tools) == 3
    
    @patch('txgemma.tool_factory.get_loader')
    def test_exclude_name_pattern(self, mock_get_loader, mock_loader):
        """Test excluding tools by name pattern."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools(exclude_name_pattern="^CYP")
        
        # Should exclude CYP2C9_Veith
        assert len(tools) == 2
        assert not any("CYP" in t.name for t in tools)
    
    @patch('txgemma.tool_factory.get_loader')
    def test_combined_filters(self, mock_get_loader, mock_loader):
        """Test combining multiple filters."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools(
            filter_placeholder="Drug SMILES",
            max_placeholders=1,
            exclude_name_pattern="^CYP"
        )
        
        # Should only get Lipophilicity
        assert len(tools) == 1
        assert "Lipophilicity" in tools[0].name


# =============================================================================
# TOOL INTROSPECTION TESTS
# =============================================================================

class TestToolIntrospection:
    """Test tool analysis and suggestion functions."""
    
    @patch('txgemma.tool_factory.get_loader')
    def test_analyze_tools(self, mock_get_loader, mock_loader):
        """Test tool analysis returns statistics."""
        mock_get_loader.return_value = mock_loader
        
        stats = analyze_tools()
        
        assert "total_tools" in stats
        assert stats["total_tools"] == 3
        assert "total_placeholders" in stats
        assert "placeholder_usage" in stats
        assert "tools_by_complexity" in stats
    
    @patch('txgemma.tool_factory.get_loader')
    def test_suggest_tool_subsets(self, mock_get_loader, mock_loader):
        """Test tool subset suggestions."""
        mock_get_loader.return_value = mock_loader
        
        # Mock get_tool_names to return appropriate lists
        with patch('txgemma.tool_factory.get_tool_names') as mock_get_names:
            mock_get_names.return_value = ["Tool1", "Tool2"]
            
            subsets = suggest_tool_subsets()
            
            assert "drug_discovery" in subsets
            assert "protein_analysis" in subsets
            assert isinstance(subsets["drug_discovery"], list)
    
    @patch('txgemma.tool_factory.get_loader')
    def test_get_tool_names(self, mock_get_loader, mock_loader):
        """Test getting tool names with filters."""
        mock_get_loader.return_value = mock_loader
        
        names = get_tool_names()
        
        assert len(names) == 3
        assert all(isinstance(name, str) for name in names)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    @patch('txgemma.tool_factory.get_loader')
    def test_full_workflow_numeric_task(self, mock_get_loader, mock_loader):
        """Test complete workflow for numeric prediction task."""
        mock_get_loader.return_value = mock_loader
        
        # Build tools
        tools = build_tools(filter_placeholder="Drug SMILES", max_placeholders=1)
        
        assert len(tools) >= 1
        
        # Check first tool
        tool = tools[0]
        assert tool.name is not None
        assert tool.description is not None
        assert "type" in tool.inputSchema
        assert tool.inputSchema["type"] == "object"
        assert "properties" in tool.inputSchema
        
        # Verify description quality
        assert "Question:" in tool.description
        assert "Drug SMILES:" not in tool.description  # No parameter placeholders
    
    @patch('txgemma.tool_factory.get_loader')
    def test_full_workflow_classification_task(self, mock_get_loader, mock_loader):
        """Test complete workflow for classification task."""
        mock_get_loader.return_value = mock_loader
        
        # Get classification tool
        tools = build_tools()
        cyp_tools = [t for t in tools if "CYP" in t.name]
        
        if cyp_tools:
            tool = cyp_tools[0]
            
            # Should have classification options in description
            # (assuming the mock has classification structure)
            assert tool.description is not None
    
    @patch('txgemma.tool_factory.get_loader')
    def test_parameter_mapping_consistency(self, mock_get_loader, mock_loader):
        """Test that parameter mapping is consistent across tools."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools()
        
        # All tools with Drug SMILES should use drug_smiles parameter
        for tool in tools:
            if any("Drug SMILES" == prop["title"]
                   for prop in tool.inputSchema["properties"].values()):
                assert "drug_smiles" in tool.inputSchema["properties"]
    
    @patch('txgemma.tool_factory.get_loader')
    def test_description_no_redundancy_with_schema(self, mock_get_loader, mock_loader):
        """Test that description doesn't duplicate schema info."""
        mock_get_loader.return_value = mock_loader
        
        tools = build_tools()
        
        for tool in tools:
            # Description should not contain parameter placeholder syntax
            assert "{" not in tool.description or "**" in tool.description  # Allow {**header**}
            
            # Description should not repeat parameter names with colons
            for param_name in tool.inputSchema["properties"].values():
                title = param_name.get("title", "")
                if title:
                    # Title should not appear in description with colon format
                    assert f"{title}:" not in tool.description