"""
Tests for txgemma.tool_factory module.

Tests tool building, filtering, and introspection.
"""

import pytest
from unittest.mock import Mock, patch

from txgemma.tool_factory import (
    get_placeholder_type,
    get_placeholder_description,
    get_placeholder_pattern,
    build_tool_from_template,
    build_tools,
    get_tool_names,
    analyze_tools,
    suggest_tool_subsets,
)


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
    
    def test_get_placeholder_description_known(self):
        """Test descriptions for known placeholders."""
        desc = get_placeholder_description("Drug SMILES")
        assert "SMILES" in desc
        assert "drug" in desc.lower()
        
        desc = get_placeholder_description("Target sequence")
        assert "amino acid" in desc.lower()
        assert "protein" in desc.lower()
    
    def test_get_placeholder_description_unknown(self):
        """Test fallback description for unknown placeholders."""
        desc = get_placeholder_description("Custom Parameter")
        assert "Custom Parameter" in desc
        assert "Input parameter" in desc
    
    def test_get_placeholder_description_with_usage(self):
        """Test description includes usage count."""
        desc = get_placeholder_description("Drug SMILES", usage_count=15)
        assert "15 tools" in desc
    
    def test_get_placeholder_pattern_smiles(self):
        """Test SMILES validation pattern."""
        pattern = get_placeholder_pattern("Drug SMILES")
        assert pattern is not None
        assert "A-Za-z0-9" in pattern
    
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
    
    def test_get_placeholder_pattern_none(self):
        """Test that some placeholders have no pattern."""
        pattern = get_placeholder_pattern("Indication")
        assert pattern is None


class TestBuildToolFromTemplate:
    """Test building tools from templates."""
    
    def test_build_tool_simple(self):
        """Test building tool with single placeholder."""
        # Mock template
        template = Mock()
        template.name = "test_tool"
        template.placeholders = ["Drug SMILES"]
        template.get_description.return_value = "Test description"
        
        tool = build_tool_from_template(template)
        
        assert tool.name == "test_tool"
        assert tool.description == "Test description"
        assert "Drug SMILES" in tool.inputSchema["properties"]
        assert tool.inputSchema["required"] == ["Drug SMILES"]
    
    def test_build_tool_multiple_placeholders(self):
        """Test building tool with multiple placeholders."""
        template = Mock()
        template.name = "complex_tool"
        template.placeholders = ["Drug SMILES", "Target sequence", "Dose"]
        template.get_description.return_value = "Complex tool"
        
        tool = build_tool_from_template(template)
        
        assert len(tool.inputSchema["properties"]) == 3
        assert tool.inputSchema["required"] == ["Drug SMILES", "Target sequence", "Dose"]
        
        # Check types are inferred correctly
        assert tool.inputSchema["properties"]["Drug SMILES"]["type"] == "string"
        assert tool.inputSchema["properties"]["Target sequence"]["type"] == "string"
        assert tool.inputSchema["properties"]["Dose"]["type"] == "number"
    
    def test_build_tool_with_patterns(self):
        """Test that patterns are added for validated fields."""
        template = Mock()
        template.name = "validated_tool"
        template.placeholders = ["Drug SMILES", "Trial phase"]
        template.get_description.return_value = "Tool with validation"
        
        tool = build_tool_from_template(template)
        
        # SMILES should have pattern
        assert "pattern" in tool.inputSchema["properties"]["Drug SMILES"]
        
        # Phase should have pattern
        assert "pattern" in tool.inputSchema["properties"]["Trial phase"]
    
    def test_build_tool_with_stats(self):
        """Test building tool with placeholder statistics."""
        template = Mock()
        template.name = "stats_tool"
        template.placeholders = ["Drug SMILES"]
        template.get_description.return_value = "Tool description"
        
        stats = {"Drug SMILES": 42}
        tool = build_tool_from_template(template, placeholder_stats=stats)
        
        # Description should mention usage
        desc = tool.inputSchema["properties"]["Drug SMILES"]["description"]
        assert "42 tools" in desc


class TestBuildTools:
    """Test build_tools function with various filters."""
    
    @patch('txgemma.tool_factory.get_loader')
    def test_build_tools_all(self, mock_get_loader):
        """Test building all tools without filters."""
        # Mock loader
        mock_loader = Mock()
        
        # Create mock templates
        template1 = Mock()
        template1.name = "tool1"
        template1.placeholders = ["Drug SMILES"]
        template1.placeholder_count.return_value = 1
        template1.get_description.return_value = "Tool 1"
        
        template2 = Mock()
        template2.name = "tool2"
        template2.placeholders = ["Target sequence"]
        template2.placeholder_count.return_value = 1
        template2.get_description.return_value = "Tool 2"
        
        mock_loader.all.return_value = {
            "tool1": template1,
            "tool2": template2
        }
        mock_loader.placeholder_stats.return_value = {}
        
        mock_get_loader.return_value = mock_loader
        
        # Build tools
        tools = build_tools()
        
        assert len(tools) == 2
        # loader.all() is called multiple times (for filtering and logging)
        assert mock_loader.all.called
    
    @patch('txgemma.tool_factory.get_loader')
    def test_build_tools_filter_single_placeholder(self, mock_get_loader):
        """Test filtering by single placeholder."""
        mock_loader = Mock()
        
        template1 = Mock()
        template1.name = "smiles_tool"
        template1.placeholders = ["Drug SMILES"]
        template1.placeholder_count.return_value = 1
        template1.get_description.return_value = "SMILES tool"
        
        mock_loader.filter_by_placeholder.return_value = {"smiles_tool": template1}
        mock_loader.placeholder_stats.return_value = {}
        mock_loader.all.return_value = {"smiles_tool": template1, "other": Mock()}
        
        mock_get_loader.return_value = mock_loader
        
        # Build with filter
        tools = build_tools(filter_placeholder="Drug SMILES")
        
        assert len(tools) == 1
        assert tools[0].name == "smiles_tool"
        mock_loader.filter_by_placeholder.assert_called_once_with("Drug SMILES", exact=True)
    
    @patch('txgemma.tool_factory.get_loader')
    def test_build_tools_filter_multiple_placeholders(self, mock_get_loader):
        """Test filtering by multiple placeholders."""
        mock_loader = Mock()
        
        template = Mock()
        template.name = "interaction_tool"
        template.placeholders = ["Drug SMILES", "Target sequence"]
        template.placeholder_count.return_value = 2
        template.get_description.return_value = "Interaction tool"
        
        mock_loader.filter_by_placeholders.return_value = {"interaction_tool": template}
        mock_loader.placeholder_stats.return_value = {}
        mock_loader.all.return_value = {"interaction_tool": template}
        
        mock_get_loader.return_value = mock_loader
        
        # Build with multiple filters
        tools = build_tools(
            filter_placeholders=["Drug SMILES", "Target sequence"],
            match_all=True
        )
        
        assert len(tools) == 1
        mock_loader.filter_by_placeholders.assert_called_once_with(
            ["Drug SMILES", "Target sequence"],
            match_all=True
        )
    
    @patch('txgemma.tool_factory.get_loader')
    def test_build_tools_max_placeholders(self, mock_get_loader):
        """Test filtering by maximum placeholder count."""
        mock_loader = Mock()
        
        # Simple tool (1 placeholder)
        simple = Mock()
        simple.name = "simple"
        simple.placeholders = ["Drug SMILES"]
        simple.placeholder_count.return_value = 1
        simple.get_description.return_value = "Simple"
        
        # Complex tool (3 placeholders)
        complex_tool = Mock()
        complex_tool.name = "complex"
        complex_tool.placeholders = ["A", "B", "C"]
        complex_tool.placeholder_count.return_value = 3
        complex_tool.get_description.return_value = "Complex"
        
        mock_loader.all.return_value = {
            "simple": simple,
            "complex": complex_tool
        }
        mock_loader.placeholder_stats.return_value = {}
        
        mock_get_loader.return_value = mock_loader
        
        # Build with max filter
        tools = build_tools(max_placeholders=2)
        
        # Should only get simple tool
        assert len(tools) == 1
        assert tools[0].name == "simple"
    
    @patch('txgemma.tool_factory.get_loader')
    def test_build_tools_fuzzy_match(self, mock_get_loader):
        """Test fuzzy placeholder matching."""
        mock_loader = Mock()
        
        template = Mock()
        template.name = "seq_tool"
        template.placeholders = ["Target sequence"]
        template.placeholder_count.return_value = 1
        template.get_description.return_value = "Sequence tool"
        
        mock_loader.filter_by_placeholder.return_value = {"seq_tool": template}
        mock_loader.placeholder_stats.return_value = {}
        mock_loader.all.return_value = {"seq_tool": template}
        
        mock_get_loader.return_value = mock_loader
        
        # Build with fuzzy match
        tools = build_tools(filter_placeholder="sequence", exact_match=False)
        
        mock_loader.filter_by_placeholder.assert_called_once_with("sequence", exact=False)


class TestGetToolNames:
    """Test get_tool_names function."""
    
    @patch('txgemma.tool_factory.get_loader')
    def test_get_tool_names_all(self, mock_get_loader):
        """Test getting all tool names."""
        mock_loader = Mock()
        mock_loader.all.return_value = {
            "tool1": Mock(),
            "tool2": Mock(),
            "tool3": Mock()
        }
        mock_get_loader.return_value = mock_loader
        
        names = get_tool_names()
        
        assert len(names) == 3
        assert "tool1" in names
        assert "tool2" in names
        assert "tool3" in names
    
    @patch('txgemma.tool_factory.get_loader')
    def test_get_tool_names_filtered(self, mock_get_loader):
        """Test getting filtered tool names."""
        mock_loader = Mock()
        mock_loader.filter_by_placeholder.return_value = {
            "smiles_tool": Mock()
        }
        mock_get_loader.return_value = mock_loader
        
        names = get_tool_names(filter_placeholder="Drug SMILES")
        
        assert len(names) == 1
        assert "smiles_tool" in names


class TestAnalyzeTools:
    """Test tool analysis function."""
    
    @patch('txgemma.tool_factory.get_loader')
    def test_analyze_tools(self, mock_get_loader):
        """Test tool analysis returns correct statistics."""
        mock_loader = Mock()
        
        # Create mock templates
        template1 = Mock()
        template1.placeholder_count.return_value = 1
        
        template2 = Mock()
        template2.placeholder_count.return_value = 2
        
        template3 = Mock()
        template3.placeholder_count.return_value = 1
        
        mock_loader.all.return_value = {
            "tool1": template1,
            "tool2": template2,
            "tool3": template3
        }
        
        mock_loader.placeholder_stats.return_value = {
            "Drug SMILES": 3,
            "Target sequence": 1
        }
        
        mock_loader.most_common_placeholders.return_value = [
            ("Drug SMILES", 3),
            ("Target sequence", 1)
        ]
        
        mock_get_loader.return_value = mock_loader
        
        # Analyze
        stats = analyze_tools()
        
        assert stats["total_tools"] == 3
        assert stats["total_placeholders"] == 2
        assert stats["simple_tools"] == 3  # All have ≤2 placeholders
        assert stats["complex_tools"] == 0
        assert 1 in stats["tools_by_complexity"]
        assert 2 in stats["tools_by_complexity"]


class TestSuggestToolSubsets:
    """Test tool subset suggestions."""
    
    @patch('txgemma.tool_factory.get_tool_names')
    def test_suggest_tool_subsets(self, mock_get_tool_names):
        """Test that subset suggestions call correct filters."""
        # Mock different results for different calls
        mock_get_tool_names.side_effect = [
            ["tool1", "tool2"],  # drug_discovery
            ["tool3"],           # protein_analysis
            ["tool1"],           # simple_predictions
            ["tool4"],           # drug_target_interaction
        ]
        
        subsets = suggest_tool_subsets()
        
        assert "drug_discovery" in subsets
        assert "protein_analysis" in subsets
        assert "simple_predictions" in subsets
        assert "drug_target_interaction" in subsets
        
        # Verify correct calls were made
        assert mock_get_tool_names.call_count == 4


class TestPlaceholderPatternValidation:
    """Test that patterns actually validate correctly."""
    
    def test_smiles_pattern_valid(self):
        """Test SMILES pattern accepts valid SMILES."""
        import re
        pattern = get_placeholder_pattern("Drug SMILES")
        regex = re.compile(pattern)
        
        # Valid SMILES examples
        assert regex.match("CC(=O)O")  # Acetic acid
        assert regex.match("c1ccccc1")  # Benzene
        assert regex.match("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
    
    def test_smiles_pattern_invalid(self):
        """Test SMILES pattern rejects invalid strings."""
        import re
        pattern = get_placeholder_pattern("Drug SMILES")
        regex = re.compile(pattern)
        
        # Invalid SMILES
        assert not regex.match("Hello World")
        assert not regex.match("123 456")
    
    def test_sequence_pattern_valid(self):
        """Test sequence pattern accepts valid amino acid sequences."""
        import re
        pattern = get_placeholder_pattern("Target sequence")
        regex = re.compile(pattern)
        
        # Valid sequences
        assert regex.match("MKTAYIAK")
        assert regex.match("ACDEFGHIKLMNPQRSTVWY")
    
    def test_sequence_pattern_invalid(self):
        """Test sequence pattern rejects invalid sequences."""
        import re
        pattern = get_placeholder_pattern("Target sequence")
        regex = re.compile(pattern)
        
        # Invalid sequences (lowercase, numbers, invalid letters)
        assert not regex.match("mktayiak")
        assert not regex.match("MKTAY123")
        assert not regex.match("MKTAYIAX")  # X not in valid amino acids
    
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


class TestBuildToolsErrorHandling:
    """Test error handling in tool building."""
    
    @patch('txgemma.tool_factory.get_loader')
    @patch('txgemma.tool_factory.logger')
    def test_build_tools_handles_errors(self, mock_logger, mock_get_loader):
        """Test that build_tools continues on individual tool errors."""
        mock_loader = Mock()
        
        # Good template
        good_template = Mock()
        good_template.name = "good_tool"
        good_template.placeholders = ["Drug SMILES"]
        good_template.placeholder_count.return_value = 1
        good_template.get_description.return_value = "Good tool"
        
        # Bad template that raises error
        bad_template = Mock()
        bad_template.name = "bad_tool"
        bad_template.placeholders = ["Something"]
        bad_template.placeholder_count.return_value = 1
        bad_template.get_description.side_effect = RuntimeError("Template error")
        
        mock_loader.all.return_value = {
            "good_tool": good_template,
            "bad_tool": bad_template
        }
        mock_loader.placeholder_stats.return_value = {}
        
        mock_get_loader.return_value = mock_loader
        
        # Should build good tool, skip bad one
        tools = build_tools()
        
        assert len(tools) == 1
        assert tools[0].name == "good_tool"
        
        # Should log error
        assert mock_logger.error.called