"""
Tests for txgemma.tool_factory module.

Tests tool building, filtering, and introspection with proper mocking.
"""

from unittest.mock import Mock, patch

import pytest

from txgemma.tool_factory import (
    analyze_tools,
    build_tool_from_template,
    build_tools,
    get_placeholder_description,
    get_placeholder_pattern,
    get_placeholder_type,
    get_tool_names,
    suggest_tool_subsets,
)


# -------------------------
# Test Fixtures
# -------------------------

def create_mock_template(name: str, placeholders: list[str], description: str = None):
    """
    Create a properly structured mock PromptTemplate matching the real interface.
    
    Based on actual PromptTemplate from prompts.py:
    - name: str
    - placeholders: list[str] 
    - metadata: dict
    - has_placeholder(str) -> bool
    - placeholder_count() -> int
    - get_description() -> str
    """
    template = Mock()
    template.name = name
    template.placeholders = placeholders
    template.metadata = {"description": description} if description else {}
    
    # Mock methods to match actual interface
    def mock_has_placeholder(ph: str) -> bool:
        return ph in placeholders
    
    def mock_placeholder_count() -> int:
        return len(placeholders)
    
    def mock_get_description() -> str:
        if description:
            return description
        return f"TxGemma prediction task: {name}"
    
    template.has_placeholder = Mock(side_effect=mock_has_placeholder)
    template.placeholder_count = Mock(side_effect=mock_placeholder_count)
    template.get_description = Mock(side_effect=mock_get_description)
    template.required_inputs = set(placeholders)
    
    return template


@pytest.fixture
def mock_template_simple():
    """Create a simple mock template with one placeholder."""
    return create_mock_template(
        name="test_tool",
        placeholders=["Drug SMILES"],
        description="Test tool description"
    )


@pytest.fixture
def mock_template_complex():
    """Create a complex mock template with multiple placeholders."""
    return create_mock_template(
        name="complex_tool",
        placeholders=["Drug SMILES", "Target sequence", "Dose"],
        description="Complex tool description"
    )


@pytest.fixture
def mock_loader_with_templates():
    """
    Create a mock loader with sample templates.
    
    Mimics the actual PromptLoader interface from prompts.py:
    - all() -> dict[str, PromptTemplate]
    - filter_by_placeholder(str, exact=bool) -> dict[str, PromptTemplate]
    - filter_by_placeholders(list[str], match_all=bool) -> dict[str, PromptTemplate]
    - placeholder_stats() -> dict[str, int]
    - most_common_placeholders(int) -> list[tuple[str, int]]
    """
    mock_loader = Mock()
    
    # Create realistic templates
    template1 = create_mock_template(
        "tdc_Tool1_predict",
        ["Drug SMILES"],
        "Tool 1"
    )
    
    template2 = create_mock_template(
        "tdc_Tool2_predict",
        ["Target sequence"],
        "Tool 2"
    )
    
    template3 = create_mock_template(
        "ToxCast_Tool3_predict",
        ["Drug SMILES"],
        "ToxCast Tool 3"
    )
    
    template4 = create_mock_template(
        "tdc_Complex_predict",
        ["Drug SMILES", "Target sequence", "Dose"],
        "Complex tool"
    )
    
    template5 = create_mock_template(
        "ToxCast_Tool5_predict",
        ["Drug SMILES"],
        "ToxCast Tool 5"
    )
    
    all_templates = {
        "tdc_Tool1_predict": template1,
        "tdc_Tool2_predict": template2,
        "ToxCast_Tool3_predict": template3,
        "tdc_Complex_predict": template4,
        "ToxCast_Tool5_predict": template5,
    }
    
    # Mock loader methods
    mock_loader.all.return_value = all_templates
    
    # Mock filter_by_placeholder - exact match
    def mock_filter_by_placeholder(placeholder: str, exact: bool = True):
        if exact:
            return {
                name: tmpl 
                for name, tmpl in all_templates.items()
                if placeholder in tmpl.placeholders
            }
        else:
            # Fuzzy match
            placeholder_lower = placeholder.lower()
            return {
                name: tmpl
                for name, tmpl in all_templates.items()
                if any(placeholder_lower in ph.lower() for ph in tmpl.placeholders)
            }
    
    mock_loader.filter_by_placeholder = Mock(side_effect=mock_filter_by_placeholder)
    
    # Mock filter_by_placeholders
    def mock_filter_by_placeholders(placeholders: list[str], match_all: bool = True):
        if match_all:
            return {
                name: tmpl
                for name, tmpl in all_templates.items()
                if all(ph in tmpl.placeholders for ph in placeholders)
            }
        else:
            return {
                name: tmpl
                for name, tmpl in all_templates.items()
                if any(ph in tmpl.placeholders for ph in placeholders)
            }
    
    mock_loader.filter_by_placeholders = Mock(side_effect=mock_filter_by_placeholders)
    
    # Mock placeholder_stats
    def mock_placeholder_stats():
        stats = {}
        for tmpl in all_templates.values():
            for ph in tmpl.placeholders:
                stats[ph] = stats.get(ph, 0) + 1
        return stats
    
    mock_loader.placeholder_stats = Mock(side_effect=mock_placeholder_stats)
    
    # Mock most_common_placeholders
    def mock_most_common_placeholders(top_n: int = 10):
        stats = mock_placeholder_stats()
        return sorted(stats.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    mock_loader.most_common_placeholders = Mock(side_effect=mock_most_common_placeholders)
    
    return mock_loader


# -------------------------
# Placeholder Metadata Tests
# -------------------------

class TestPlaceholderMetadata:
    """Test placeholder metadata functions."""

    def test_get_placeholder_type_string(self):
        """Test that most placeholders default to string."""
        assert get_placeholder_type("Drug SMILES") == "string"
        assert get_placeholder_type("Target sequence") == "string"
        assert get_placeholder_type("Indication") == "string"

    def test_get_placeholder_type_integer(self):
        """Test integer type detection."""
        assert get_placeholder_type("Trial count") == "integer"
        assert get_placeholder_type("Number of patients") == "integer"
        assert get_placeholder_type("Index") == "integer"
        assert get_placeholder_type("Quantity") == "integer"

    def test_get_placeholder_type_number(self):
        """Test number (float) type detection."""
        assert get_placeholder_type("Dose") == "number"
        assert get_placeholder_type("Concentration") == "number"
        assert get_placeholder_type("Score value") == "number"

    def test_get_placeholder_type_boolean(self):
        """Test boolean type detection."""
        assert get_placeholder_type("Is active") == "boolean"
        assert get_placeholder_type("Has toxicity") == "boolean"
        assert get_placeholder_type("Can bind") == "boolean"
        assert get_placeholder_type("Should proceed") == "boolean"

    def test_get_placeholder_description_known(self):
        """Test descriptions for known placeholders."""
        desc = get_placeholder_description("Drug SMILES")
        assert "SMILES" in desc
        assert "drug" in desc.lower()

        desc = get_placeholder_description("Target sequence")
        assert "amino acid" in desc.lower()
        assert "protein" in desc.lower()
        
        desc = get_placeholder_description("Cell line")
        assert "Cell line" in desc

    def test_get_placeholder_description_unknown(self):
        """Test fallback description for unknown placeholders."""
        desc = get_placeholder_description("Custom Parameter")
        assert "Custom Parameter" in desc
        assert "Input parameter" in desc

    def test_get_placeholder_description_with_usage(self):
        """Test description includes usage count."""
        desc = get_placeholder_description("Drug SMILES", usage_count=15)
        assert "15 tools" in desc
        
    def test_get_placeholder_description_with_usage_count_1(self):
        """Test that usage count of 1 doesn't add extra text."""
        desc = get_placeholder_description("Drug SMILES", usage_count=1)
        assert "tools" not in desc

    def test_get_placeholder_pattern_smiles(self):
        """Test SMILES validation pattern."""
        pattern = get_placeholder_pattern("Drug SMILES")
        assert pattern is not None
        assert "A-Za-z0-9" in pattern
        
        pattern = get_placeholder_pattern("Molecule SMILES")
        assert pattern is not None

    def test_get_placeholder_pattern_sequence(self):
        """Test amino acid sequence pattern."""
        pattern = get_placeholder_pattern("Target sequence")
        assert pattern is not None
        assert "ACDEFGHIKLMNPQRSTVWY" in pattern

        pattern = get_placeholder_pattern("Epitope amino acid sequence")
        assert pattern is not None

    def test_get_placeholder_pattern_phase(self):
        """Test trial phase pattern."""
        pattern = get_placeholder_pattern("Trial phase")
        assert pattern is not None
        assert "[1-3]" in pattern
        
        pattern = get_placeholder_pattern("Phase")
        assert pattern is not None

    def test_get_placeholder_pattern_none(self):
        """Test that some placeholders have no pattern."""
        assert get_placeholder_pattern("Indication") is None
        assert get_placeholder_pattern("Disease") is None
        assert get_placeholder_pattern("Cell line") is None


# -------------------------
# Build Tool From Template Tests
# -------------------------

class TestBuildToolFromTemplate:
    """Test building tools from templates."""

    def test_build_tool_simple(self, mock_template_simple):
        """Test building tool with single placeholder."""
        tool = build_tool_from_template(mock_template_simple)

        assert tool.name == "test_tool"
        assert tool.description == "Test tool description"
        assert "Drug SMILES" in tool.inputSchema["properties"]
        assert tool.inputSchema["required"] == ["Drug SMILES"]
        assert tool.inputSchema["additionalProperties"] is False

    def test_build_tool_multiple_placeholders(self, mock_template_complex):
        """Test building tool with multiple placeholders."""
        tool = build_tool_from_template(mock_template_complex)

        assert tool.name == "complex_tool"
        assert len(tool.inputSchema["properties"]) == 3
        assert tool.inputSchema["required"] == ["Drug SMILES", "Target sequence", "Dose"]

        # Check types are inferred correctly
        assert tool.inputSchema["properties"]["Drug SMILES"]["type"] == "string"
        assert tool.inputSchema["properties"]["Target sequence"]["type"] == "string"
        assert tool.inputSchema["properties"]["Dose"]["type"] == "number"

    def test_build_tool_with_patterns(self):
        """Test that patterns are added for validated fields."""
        template = create_mock_template(
            "validated_tool",
            ["Drug SMILES", "Trial phase"],
            "Tool with validation"
        )

        tool = build_tool_from_template(template)

        # SMILES should have pattern
        assert "pattern" in tool.inputSchema["properties"]["Drug SMILES"]

        # Phase should have pattern
        assert "pattern" in tool.inputSchema["properties"]["Trial phase"]

    def test_build_tool_with_stats(self):
        """Test building tool with placeholder statistics."""
        template = create_mock_template(
            "stats_tool",
            ["Drug SMILES"],
            "Tool description"
        )

        stats = {"Drug SMILES": 42}
        tool = build_tool_from_template(template, placeholder_stats=stats)

        # Description should mention usage
        desc = tool.inputSchema["properties"]["Drug SMILES"]["description"]
        assert "42 tools" in desc
    
    def test_build_tool_schema_structure(self, mock_template_simple):
        """Test that generated schema has correct structure."""
        tool = build_tool_from_template(mock_template_simple)
        
        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "additionalProperties" in schema
        assert schema["additionalProperties"] is False


# -------------------------
# Build Tools Tests
# -------------------------

class TestBuildTools:
    """Test build_tools function with various filters."""

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_all(self, mock_get_loader, mock_loader_with_templates):
        """Test building all tools without filters."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools()

        # Should build all 5 templates
        assert len(tools) == 5
        assert mock_loader_with_templates.all.called

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_filter_placeholder(self, mock_get_loader, mock_loader_with_templates):
        """Test filtering by placeholder."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(filter_placeholder="Drug SMILES")

        # Should get tools with Drug SMILES (tool1, tool3, tool4, tool5)
        assert len(tools) == 4
        for tool in tools:
            assert "Drug SMILES" in tool.inputSchema["required"]

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_filter_placeholder_not_found(self, mock_get_loader, mock_loader_with_templates):
        """Test filtering by placeholder that doesn't exist."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(filter_placeholder="Nonexistent Placeholder")

        # Should return empty list
        assert len(tools) == 0

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_max_placeholders(self, mock_get_loader, mock_loader_with_templates):
        """Test filtering by maximum number of placeholders."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(max_placeholders=1)

        # Should only get simple tools (1 placeholder)
        assert len(tools) == 4  # All except complex_tool
        for tool in tools:
            assert len(tool.inputSchema["required"]) <= 1

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_max_placeholders_zero(self, mock_get_loader, mock_loader_with_templates):
        """Test max_placeholders=0 returns nothing."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(max_placeholders=0)

        assert len(tools) == 0

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_combined_filters(self, mock_get_loader, mock_loader_with_templates):
        """Test combining placeholder filter and max_placeholders."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(filter_placeholder="Drug SMILES", max_placeholders=1)

        # Should get simple Drug SMILES tools only (tool1, tool3, tool5)
        assert len(tools) == 3
        for tool in tools:
            assert "Drug SMILES" in tool.inputSchema["required"]
            assert len(tool.inputSchema["required"]) == 1

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_filter_placeholders_match_all(self, mock_get_loader, mock_loader_with_templates):
        """Test filtering by multiple placeholders with match_all=True."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=True
        )

        # Should only get complex tool that has both
        assert len(tools) == 1
        assert tools[0].name == "tdc_Complex_predict"
        assert "Drug SMILES" in tools[0].inputSchema["required"]
        assert "Target sequence" in tools[0].inputSchema["required"]

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_filter_placeholders_match_any(self, mock_get_loader, mock_loader_with_templates):
        """Test filtering by multiple placeholders with match_all=False."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=False
        )

        # Should get all tools with either placeholder (all 5)
        assert len(tools) == 5

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_exact_match_false(self, mock_get_loader, mock_loader_with_templates):
        """Test fuzzy matching with exact_match=False."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            filter_placeholder="smiles",
            exact_match=False
        )

        # Should get tools with any placeholder containing "smiles" (fuzzy)
        # All Drug SMILES tools should match
        assert len(tools) == 4
        for tool in tools:
            # Should have a placeholder containing "smiles" (case insensitive)
            assert any("smiles" in ph.lower() for ph in tool.inputSchema["required"])

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_exclude_complex(self, mock_get_loader, mock_loader_with_templates):
        """Test exclude_complex parameter."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(exclude_complex=True)

        # Should exclude tools with >2 placeholders
        assert len(tools) == 4  # All except complex_tool
        for tool in tools:
            assert len(tool.inputSchema["required"]) <= 2

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_exclude_complex_with_max_placeholders(self, mock_get_loader, mock_loader_with_templates):
        """Test that max_placeholders takes precedence over exclude_complex."""
        mock_get_loader.return_value = mock_loader_with_templates

        # max_placeholders should override exclude_complex
        tools = build_tools(
            max_placeholders=1,
            exclude_complex=True  # This should be ignored
        )

        # Should only get tools with ≤1 placeholder
        assert len(tools) == 4
        for tool in tools:
            assert len(tool.inputSchema["required"]) <= 1


# -------------------------
# Exclude Name Pattern Tests
# -------------------------

class TestExcludeNamePattern:
    """Test excluding tools by name pattern."""

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_prefix_pattern(self, mock_get_loader, mock_loader_with_templates):
        """Test excluding tools with specific prefix."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(exclude_name_pattern="^ToxCast")

        # Should exclude ToxCast_Tool3 and ToxCast_Tool5
        assert len(tools) == 3
        for tool in tools:
            assert not tool.name.startswith("ToxCast")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_substring_pattern(self, mock_get_loader, mock_loader_with_templates):
        """Test excluding by substring match."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(exclude_name_pattern="ToxCast")

        # Should exclude any tool with "ToxCast" in name
        assert len(tools) == 3
        for tool in tools:
            assert "ToxCast" not in tool.name

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_multiple_patterns(self, mock_get_loader, mock_loader_with_templates):
        """Test excluding with alternation pattern."""
        mock_get_loader.return_value = mock_loader_with_templates

        # Exclude ToxCast OR Complex
        tools = build_tools(exclude_name_pattern="(ToxCast|Complex)")

        # Should exclude ToxCast tools and Complex tool
        assert len(tools) == 2
        for tool in tools:
            assert "ToxCast" not in tool.name
            assert "Complex" not in tool.name

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_case_insensitive(self, mock_get_loader, mock_loader_with_templates):
        """Test case-insensitive exclusion."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(exclude_name_pattern="(?i)toxcast")

        # Should exclude ToxCast regardless of case
        assert len(tools) == 3
        for tool in tools:
            assert "toxcast" not in tool.name.lower()

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_with_placeholder_filter(self, mock_get_loader, mock_loader_with_templates):
        """Test combining exclusion with placeholder filter."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            filter_placeholder="Drug SMILES",
            exclude_name_pattern="^ToxCast"
        )

        # Should get Drug SMILES tools excluding ToxCast (tool1, tool4)
        assert len(tools) == 2
        for tool in tools:
            assert "Drug SMILES" in tool.inputSchema["required"]
            assert not tool.name.startswith("ToxCast")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_with_max_placeholders(self, mock_get_loader, mock_loader_with_templates):
        """Test combining exclusion with complexity filter."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            max_placeholders=1,
            exclude_name_pattern="^ToxCast"
        )

        # Should get simple tools excluding ToxCast (tool1, tool2)
        assert len(tools) == 2
        for tool in tools:
            assert len(tool.inputSchema["required"]) <= 1
            assert not tool.name.startswith("ToxCast")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_all_filters_combined(self, mock_get_loader, mock_loader_with_templates):
        """Test all three filters together."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            filter_placeholder="Drug SMILES",
            max_placeholders=1,
            exclude_name_pattern="^ToxCast"
        )

        # Should get simple Drug SMILES tools excluding ToxCast (only tool1)
        assert len(tools) == 1
        tool = tools[0]
        assert "Drug SMILES" in tool.inputSchema["required"]
        assert len(tool.inputSchema["required"]) == 1
        assert not tool.name.startswith("ToxCast")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_with_filter_placeholders(self, mock_get_loader, mock_loader_with_templates):
        """Test exclude_name_pattern with filter_placeholders."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=True,
            exclude_name_pattern="^ToxCast"
        )

        # Should get complex tool only (has both, not ToxCast)
        assert len(tools) == 1
        assert tools[0].name == "tdc_Complex_predict"
        assert not tools[0].name.startswith("ToxCast")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_with_exact_match_false(self, mock_get_loader, mock_loader_with_templates):
        """Test exclude_name_pattern with fuzzy placeholder matching."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            filter_placeholder="smiles",
            exact_match=False,
            exclude_name_pattern="^ToxCast"
        )

        # Should get Drug SMILES tools excluding ToxCast
        assert len(tools) == 2  # tool1 and tool4 (complex)
        for tool in tools:
            assert not tool.name.startswith("ToxCast")
            assert any("smiles" in ph.lower() for ph in tool.inputSchema["required"])

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_with_exclude_complex(self, mock_get_loader, mock_loader_with_templates):
        """Test exclude_name_pattern with exclude_complex."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            exclude_complex=True,
            exclude_name_pattern="^ToxCast"
        )

        # Should get simple tools excluding ToxCast (tool1, tool2)
        assert len(tools) == 2
        for tool in tools:
            assert len(tool.inputSchema["required"]) <= 2
            assert not tool.name.startswith("ToxCast")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_pattern_all_parameters_combined(self, mock_get_loader, mock_loader_with_templates):
        """Test exclude_name_pattern with every other parameter."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=False,  # ANY of these
            exact_match=True,
            max_placeholders=2,
            exclude_name_pattern="^ToxCast"
        )

        # Should get tools with Drug SMILES or Target sequence, ≤2 params, no ToxCast
        assert len(tools) == 2  # tool1 and tool2
        for tool in tools:
            assert len(tool.inputSchema["required"]) <= 2
            assert not tool.name.startswith("ToxCast")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_none_pattern(self, mock_get_loader, mock_loader_with_templates):
        """Test that None pattern excludes nothing."""
        mock_get_loader.return_value = mock_loader_with_templates

        all_tools = build_tools()
        filtered_tools = build_tools(exclude_name_pattern=None)

        assert len(filtered_tools) == len(all_tools)

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_empty_string(self, mock_get_loader, mock_loader_with_templates):
        """Test that empty string pattern excludes nothing."""
        mock_get_loader.return_value = mock_loader_with_templates

        all_tools = build_tools()
        filtered_tools = build_tools(exclude_name_pattern="")

        # Empty string matches everything at position 0, so nothing should be excluded
        assert len(filtered_tools) == len(all_tools)

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_invalid_regex(self, mock_get_loader, mock_loader_with_templates):
        """Test that invalid regex raises ValueError."""
        mock_get_loader.return_value = mock_loader_with_templates

        with pytest.raises(ValueError, match="Invalid regex pattern"):
            build_tools(exclude_name_pattern="[invalid(")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_pattern_logging(self, mock_get_loader, mock_loader_with_templates, caplog):
        """Test that exclusion is logged."""
        import logging
        caplog.set_level(logging.INFO)

        mock_get_loader.return_value = mock_loader_with_templates
        build_tools(exclude_name_pattern="^ToxCast")

        assert "Excluding tools matching pattern: ^ToxCast" in caplog.text
        assert "Excluded 2 tool(s) matching pattern" in caplog.text

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_pattern_excludes_none(self, mock_get_loader, mock_loader_with_templates, caplog):
        """Test logging when pattern excludes nothing."""
        import logging
        caplog.set_level(logging.INFO)

        mock_get_loader.return_value = mock_loader_with_templates
        build_tools(exclude_name_pattern="^NonExistent")

        # Should not log exclusion count if nothing excluded
        assert "Excluded 0 tool(s)" not in caplog.text


# -------------------------
# Pattern Validation Tests
# -------------------------

class TestPatternValidation:
    """Test that generated patterns actually validate correctly."""

    def test_smiles_pattern_valid(self):
        """Test SMILES pattern accepts valid SMILES strings."""
        import re

        pattern = get_placeholder_pattern("Drug SMILES")
        regex = re.compile(pattern)

        # Valid SMILES strings
        assert regex.match("CCO")
        assert regex.match("c1ccccc1")
        assert regex.match("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        assert regex.match("C[C@H](N)C(=O)O")  # Alanine with stereochemistry

    def test_smiles_pattern_invalid(self):
        """Test SMILES pattern rejects invalid strings."""
        import re

        pattern = get_placeholder_pattern("Drug SMILES")
        regex = re.compile(pattern)

        # Invalid SMILES (contains spaces or invalid characters)
        assert not regex.match("CC O")
        assert not regex.match("")
        assert not regex.match("Hello World")

    def test_sequence_pattern_valid(self):
        """Test sequence pattern accepts valid amino acid sequences."""
        import re

        pattern = get_placeholder_pattern("Target sequence")
        regex = re.compile(pattern)

        # Valid sequences (uppercase, valid amino acids)
        assert regex.match("MKTAYIAK")
        assert regex.match("ACDEFGHIKLMNPQRSTVWY")
        assert regex.match("A")  # Single amino acid

    def test_sequence_pattern_invalid(self):
        """Test sequence pattern rejects invalid sequences."""
        import re

        pattern = get_placeholder_pattern("Target sequence")
        regex = re.compile(pattern)

        # Invalid sequences
        assert not regex.match("mktayiak")  # lowercase
        assert not regex.match("MKTAY123")  # contains numbers
        assert not regex.match("MKTAYIAX")  # X not in standard amino acids
        assert not regex.match("")  # empty

    def test_phase_pattern_valid(self):
        """Test phase pattern accepts valid phases."""
        import re

        pattern = get_placeholder_pattern("Trial phase")
        regex = re.compile(pattern)

        assert regex.match("1")
        assert regex.match("2")
        assert regex.match("3")

    def test_phase_pattern_invalid(self):
        """Test phase pattern rejects invalid phases."""
        import re

        pattern = get_placeholder_pattern("Trial phase")
        regex = re.compile(pattern)

        assert not regex.match("0")
        assert not regex.match("4")
        assert not regex.match("Phase 1")
        assert not regex.match("")


# -------------------------
# Get Tool Names Tests
# -------------------------

class TestGetToolNames:
    """Test get_tool_names function."""

    @patch("txgemma.tool_factory.get_loader")
    def test_get_tool_names_all(self, mock_get_loader, mock_loader_with_templates):
        """Test getting all tool names."""
        mock_get_loader.return_value = mock_loader_with_templates

        names = get_tool_names()

        assert len(names) == 5
        assert "tdc_Tool1_predict" in names
        assert "ToxCast_Tool3_predict" in names

    @patch("txgemma.tool_factory.get_loader")
    def test_get_tool_names_with_filter(self, mock_get_loader, mock_loader_with_templates):
        """Test getting tool names with placeholder filter."""
        mock_get_loader.return_value = mock_loader_with_templates

        names = get_tool_names(filter_placeholder="Drug SMILES")

        # Should get templates with Drug SMILES (tool1, tool3, tool4, tool5)
        assert len(names) == 4
        assert "tdc_Tool1_predict" in names
        assert "ToxCast_Tool3_predict" in names
        assert "tdc_Complex_predict" in names
        assert "ToxCast_Tool5_predict" in names
        
        # Mock should have been called with exact=True (default)
        mock_loader_with_templates.filter_by_placeholder.assert_called_with("Drug SMILES")

    @patch("txgemma.tool_factory.get_loader")
    def test_get_tool_names_with_multiple_filters(self, mock_get_loader, mock_loader_with_templates):
        """Test getting tool names with multiple placeholder filters."""
        mock_get_loader.return_value = mock_loader_with_templates

        names = get_tool_names(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=True
        )

        # Should get only complex tool that has both placeholders
        assert len(names) == 1
        assert "tdc_Complex_predict" in names
        
        mock_loader_with_templates.filter_by_placeholders.assert_called_with(
            ["Drug SMILES", "Target sequence"],
            match_all=True
        )
    
    @patch("txgemma.tool_factory.get_loader")
    def test_get_tool_names_with_match_any(self, mock_get_loader, mock_loader_with_templates):
        """Test getting tool names with match_all=False."""
        mock_get_loader.return_value = mock_loader_with_templates

        names = get_tool_names(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=False
        )

        # Should get any tool that has either placeholder
        assert len(names) == 5  # All templates have at least one of these
        
        mock_loader_with_templates.filter_by_placeholders.assert_called_with(
            ["Drug SMILES", "Target sequence"],
            match_all=False
        )


# -------------------------
# Analyze Tools Tests
# -------------------------

class TestAnalyzeTools:
    """Test analyze_tools function."""

    @patch("txgemma.tool_factory.get_loader")
    def test_analyze_tools_structure(self, mock_get_loader, mock_loader_with_templates):
        """Test that analyze_tools returns correct structure."""
        mock_loader_with_templates.placeholder_stats.return_value = {
            "Drug SMILES": 3,
            "Target sequence": 2
        }
        mock_loader_with_templates.most_common_placeholders.return_value = [
            ("Drug SMILES", 3),
            ("Target sequence", 2)
        ]
        
        mock_get_loader.return_value = mock_loader_with_templates

        analysis = analyze_tools()

        assert "total_tools" in analysis
        assert "total_placeholders" in analysis
        assert "placeholder_usage" in analysis
        assert "tools_by_complexity" in analysis
        assert "most_common_placeholders" in analysis
        assert "simple_tools" in analysis
        assert "complex_tools" in analysis

    @patch("txgemma.tool_factory.get_loader")
    def test_analyze_tools_counts(self, mock_get_loader, mock_loader_with_templates):
        """Test that analyze_tools computes correct counts."""
        mock_loader_with_templates.placeholder_stats.return_value = {
            "Drug SMILES": 4,
            "Target sequence": 2,
            "Dose": 1
        }
        mock_loader_with_templates.most_common_placeholders.return_value = [
            ("Drug SMILES", 4),
            ("Target sequence", 2)
        ]
        
        mock_get_loader.return_value = mock_loader_with_templates

        analysis = analyze_tools()

        assert analysis["total_tools"] == 5
        assert analysis["total_placeholders"] == 3
        assert analysis["simple_tools"] == 4  # 4 tools with ≤2 placeholders
        assert analysis["complex_tools"] == 1  # 1 tool with >2 placeholders


# -------------------------
# Suggest Tool Subsets Tests
# -------------------------

class TestSuggestToolSubsets:
    """Test suggest_tool_subsets function."""

    @patch("txgemma.tool_factory.get_tool_names")
    def test_suggest_tool_subsets_structure(self, mock_get_names):
        """Test that suggest_tool_subsets returns expected keys."""
        mock_get_names.return_value = []

        subsets = suggest_tool_subsets()

        assert "drug_discovery" in subsets
        assert "protein_analysis" in subsets
        assert "simple_predictions" in subsets
        assert "drug_target_interaction" in subsets

    @patch("txgemma.tool_factory.get_tool_names")
    def test_suggest_tool_subsets_calls_get_tool_names(self, mock_get_names):
        """Test that suggest_tool_subsets calls get_tool_names correctly."""
        mock_get_names.return_value = ["tool1", "tool2"]

        suggest_tool_subsets()

        # Should call get_tool_names 4 times with different filters
        assert mock_get_names.call_count == 4


# -------------------------
# Edge Cases and Error Handling
# -------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_empty_loader(self, mock_get_loader):
        """Test building tools with empty template set."""
        mock_loader = Mock()
        mock_loader.all.return_value = {}
        mock_get_loader.return_value = mock_loader

        tools = build_tools()

        assert len(tools) == 0

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_template_with_no_placeholders(self, mock_get_loader):
        """Test building tool with template that has no placeholders."""
        mock_loader = Mock()
        
        template = create_mock_template(
            "no_placeholder_tool",
            [],  # No placeholders
            "No placeholder tool"
        )
        
        mock_loader.all.return_value = {"no_placeholder_tool": template}
        mock_get_loader.return_value = mock_loader

        tools = build_tools()

        assert len(tools) == 1
        assert len(tools[0].inputSchema["required"]) == 0

    @patch("txgemma.tool_factory.get_loader")
    def test_build_tools_negative_max_placeholders(self, mock_get_loader, mock_loader_with_templates):
        """Test that negative max_placeholders returns nothing."""
        mock_get_loader.return_value = mock_loader_with_templates

        tools = build_tools(max_placeholders=-1)

        assert len(tools) == 0

    def test_build_tool_from_template_with_none_stats(self, mock_template_simple):
        """Test building tool with None placeholder_stats."""
        tool = build_tool_from_template(mock_template_simple, placeholder_stats=None)

        assert tool is not None
        assert tool.name == "test_tool"

    def test_build_tool_from_template_with_empty_stats(self, mock_template_simple):
        """Test building tool with empty placeholder_stats."""
        tool = build_tool_from_template(mock_template_simple, placeholder_stats={})

        assert tool is not None
        # Description should not include usage count
        desc = tool.inputSchema["properties"]["Drug SMILES"]["description"]
        assert "tools" not in desc


# -------------------------
# Integration-like Tests
# -------------------------

class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    @patch("txgemma.tool_factory.get_loader")
    def test_build_focused_drug_discovery_toolset(self, mock_get_loader, mock_loader_with_templates):
        """Test building a focused toolset for drug discovery."""
        mock_get_loader.return_value = mock_loader_with_templates

        # Get simple Drug SMILES tools, excluding toxicity testing
        tools = build_tools(
            filter_placeholder="Drug SMILES",
            max_placeholders=2,
            exclude_name_pattern="^ToxCast"
        )

        # Should be curated, focused set
        assert len(tools) > 0
        for tool in tools:
            # Must have Drug SMILES
            assert "Drug SMILES" in tool.inputSchema["required"]
            # Must be relatively simple
            assert len(tool.inputSchema["required"]) <= 2
            # Must not be ToxCast
            assert not tool.name.startswith("ToxCast")

    @patch("txgemma.tool_factory.get_loader")
    def test_exclude_multiple_unwanted_categories(self, mock_get_loader, mock_loader_with_templates):
        """Test excluding multiple categories of tools."""
        mock_get_loader.return_value = mock_loader_with_templates

        # Exclude both ToxCast and Complex tools
        tools = build_tools(exclude_name_pattern="(ToxCast|Complex)")

        assert len(tools) == 2  # Only tool1 and tool2
        for tool in tools:
            assert "ToxCast" not in tool.name
            assert "Complex" not in tool.name