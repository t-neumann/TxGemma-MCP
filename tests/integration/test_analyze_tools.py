"""
Tests for scripts/analyze_tools.py

Tests the CLI tool for analyzing and filtering TxGemma tools.
Uses mocked components for fast execution without HuggingFace/GPU.

"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from analyze_tools import main, print_section, print_template_details  # noqa: E402

pytestmark = [pytest.mark.integration]

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_loader():
    """Create a mock loader with sample data."""
    loader = MagicMock()

    # Mock templates
    template1 = Mock()
    template1.name = "tdc_Tool1_predict"
    template1.placeholders = ["Drug SMILES"]
    template1.placeholder_count.return_value = 1
    template1.get_description.return_value = "Tool 1 description"
    template1.template = "Instruction: Test\nContext: Context\nQuestion: {Drug SMILES}"
    template1.metadata = {"category": "test"}

    template2 = Mock()
    template2.name = "ToxCast_Tool2_predict"
    template2.placeholders = ["Drug SMILES"]
    template2.placeholder_count.return_value = 1
    template2.get_description.return_value = "ToxCast Tool 2"
    template2.template = "Instruction: ToxCast test"
    template2.metadata = {}

    loader.get.side_effect = lambda name: {
        "tdc_Tool1_predict": template1,
        "ToxCast_Tool2_predict": template2,
    }.get(name, Mock())

    loader.all.return_value = {
        "tdc_Tool1_predict": template1,
        "ToxCast_Tool2_predict": template2,
    }

    loader.placeholder_stats.return_value = {
        "Drug SMILES": 2,
        "Target sequence": 1,
    }

    loader.placeholder_usage.return_value = {"tdc_Tool1_predict", "ToxCast_Tool2_predict"}

    loader.most_common_placeholders.return_value = [
        ("Drug SMILES", 2),
        ("Target sequence", 1),
    ]

    loader.source = "test_source"
    loader.__len__ = Mock(return_value=2)

    return loader


@pytest.fixture
def mock_tools():
    """Create mock MCP tools."""
    tool1 = Mock()
    tool1.name = "tdc_Tool1_predict"
    tool1.description = "Tool 1 description"
    tool1.inputSchema = {
        "required": ["Drug SMILES"],
        "properties": {"Drug SMILES": {"type": "string", "description": "SMILES string"}},
    }

    tool2 = Mock()
    tool2.name = "ToxCast_Tool2_predict"
    tool2.description = "ToxCast Tool 2"
    tool2.inputSchema = {
        "required": ["Drug SMILES"],
        "properties": {"Drug SMILES": {"type": "string", "description": "SMILES string"}},
    }

    return [tool1, tool2]


# -------------------------
# Helper Function Tests
# -------------------------


class TestHelperFunctions:
    """Test helper functions."""

    def test_print_section(self, capsys):
        """Test print_section formats correctly."""
        print_section("Test Section")
        captured = capsys.readouterr()

        assert "Test Section" in captured.out
        assert "=" * 70 in captured.out

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    def test_print_template_details(self, mock_build_tools, mock_get_loader, mock_loader, capsys):
        """Test print_template_details shows template info."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = []

        print_template_details("tdc_Tool1_predict")
        captured = capsys.readouterr()

        assert "tdc_Tool1_predict" in captured.out
        assert "Drug SMILES" in captured.out
        assert "Tool 1 description" in captured.out

    @patch("analyze_tools.get_loader")
    def test_print_template_details_not_found(self, mock_get_loader, mock_loader, capsys):
        """Test print_template_details handles missing template."""
        mock_loader.get.side_effect = KeyError("Template not found")
        mock_get_loader.return_value = mock_loader

        print_template_details("nonexistent")
        captured = capsys.readouterr()

        assert "Error" in captured.out


# -------------------------
# CLI Argument Tests
# -------------------------


class TestCLIArguments:
    """Test CLI argument parsing and behavior."""

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_no_arguments(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test running with no arguments shows all tools."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = mock_tools
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {
            "drug_discovery": ["tdc_Tool1_predict"],
        }

        with patch.object(sys, "argv", ["analyze_tools.py"]):
            main()

        captured = capsys.readouterr()
        assert "tdc_Tool1_predict" in captured.out
        assert "ToxCast_Tool2_predict" in captured.out

    @patch("analyze_tools.get_loader")
    def test_list_placeholders(self, mock_get_loader, mock_loader, capsys):
        """Test --list-placeholders flag."""
        mock_get_loader.return_value = mock_loader

        with patch.object(sys, "argv", ["analyze_tools.py", "--list-placeholders"]):
            main()

        captured = capsys.readouterr()
        assert "Drug SMILES" in captured.out
        assert "Target sequence" in captured.out
        assert "(2 tools)" in captured.out

    @patch("analyze_tools.get_loader")
    def test_list_placeholders_json(self, mock_get_loader, mock_loader, capsys):
        """Test --list-placeholders with --json."""
        mock_get_loader.return_value = mock_loader

        with patch.object(sys, "argv", ["analyze_tools.py", "--list-placeholders", "--json"]):
            main()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["Drug SMILES"] == 2
        assert data["Target sequence"] == 1

    @patch("analyze_tools.get_loader")
    def test_source_flag(self, mock_get_loader, mock_loader, capsys):
        """Test --source flag shows prompt source."""
        mock_get_loader.return_value = mock_loader

        with patch.object(sys, "argv", ["analyze_tools.py", "--source"]):
            main()

        captured = capsys.readouterr()
        assert "test_source" in captured.out
        assert "Total templates: 2" in captured.out

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    def test_template_flag(self, mock_build_tools, mock_get_loader, mock_loader, capsys):
        """Test --template flag shows template details."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = []

        with patch.object(sys, "argv", ["analyze_tools.py", "--template", "tdc_Tool1_predict"]):
            main()

        captured = capsys.readouterr()
        assert "tdc_Tool1_predict" in captured.out
        assert "Drug SMILES" in captured.out


# -------------------------
# Filtering Tests
# -------------------------


class TestFiltering:
    """Test tool filtering via CLI."""

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_placeholder_filter(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test --placeholder filter."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = [mock_tools[0]]  # Only tool1
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {}

        with patch.object(sys, "argv", ["analyze_tools.py", "--placeholder", "Drug SMILES"]):
            main()

        # Verify build_tools was called with correct arguments
        mock_build_tools.assert_called_once()
        call_kwargs = mock_build_tools.call_args[1]
        assert call_kwargs["filter_placeholder"] == "Drug SMILES"
        assert call_kwargs["exact_match"] is True

        captured = capsys.readouterr()
        assert "using 'Drug SMILES' (exact match)" in captured.out

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_placeholder_fuzzy_filter(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test --placeholder with --fuzzy."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = mock_tools
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {}

        with patch.object(sys, "argv", ["analyze_tools.py", "--placeholder", "smiles", "--fuzzy"]):
            main()

        # Verify build_tools was called with exact_match=False
        call_kwargs = mock_build_tools.call_args[1]
        assert call_kwargs["filter_placeholder"] == "smiles"
        assert call_kwargs["exact_match"] is False

        captured = capsys.readouterr()
        assert "fuzzy match" in captured.out

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_simple_filter(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test --simple filter."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = mock_tools
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {}

        with patch.object(sys, "argv", ["analyze_tools.py", "--simple"]):
            main()

        # Verify build_tools was called with max_placeholders=2
        call_kwargs = mock_build_tools.call_args[1]
        assert call_kwargs["max_placeholders"] == 2

        captured = capsys.readouterr()
        assert "simple" in captured.out

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_exclude_filter(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test --exclude filter."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = [mock_tools[0]]  # Exclude ToxCast
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {}

        with patch.object(sys, "argv", ["analyze_tools.py", "--exclude", "^ToxCast"]):
            main()

        # Verify build_tools was called with exclude_name_pattern
        call_kwargs = mock_build_tools.call_args[1]
        assert call_kwargs["exclude_name_pattern"] == "^ToxCast"

        captured = capsys.readouterr()
        assert "excluding pattern '^ToxCast'" in captured.out

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_combined_filters(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test combining multiple filters."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = [mock_tools[0]]
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {}

        with patch.object(
            sys,
            "argv",
            [
                "analyze_tools.py",
                "--placeholder",
                "Drug SMILES",
                "--simple",
                "--exclude",
                "^ToxCast",
            ],
        ):
            main()

        # Verify all filters were applied
        call_kwargs = mock_build_tools.call_args[1]
        assert call_kwargs["filter_placeholder"] == "Drug SMILES"
        assert call_kwargs["max_placeholders"] == 2
        assert call_kwargs["exclude_name_pattern"] == "^ToxCast"

        captured = capsys.readouterr()
        assert "Drug SMILES" in captured.out
        assert "simple" in captured.out
        assert "excluding pattern" in captured.out

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_placeholders_match_all(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test --placeholders with default match_all=True."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = []
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {}

        with patch.object(
            sys, "argv", ["analyze_tools.py", "--placeholders", "Drug SMILES", "Target sequence"]
        ):
            main()

        # Verify build_tools was called with match_all=True
        call_kwargs = mock_build_tools.call_args[1]
        assert call_kwargs["filter_placeholders"] == ["Drug SMILES", "Target sequence"]
        assert call_kwargs["match_all"] is True

        captured = capsys.readouterr()
        assert "ALL of:" in captured.out

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_placeholders_match_any(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test --placeholders with --any."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = mock_tools
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {}

        with patch.object(
            sys,
            "argv",
            ["analyze_tools.py", "--placeholders", "Drug SMILES", "Target sequence", "--any"],
        ):
            main()

        # Verify build_tools was called with match_all=False
        call_kwargs = mock_build_tools.call_args[1]
        assert call_kwargs["filter_placeholders"] == ["Drug SMILES", "Target sequence"]
        assert call_kwargs["match_all"] is False

        captured = capsys.readouterr()
        assert "ANY of:" in captured.out


# -------------------------
# Output Format Tests
# -------------------------


class TestOutputFormats:
    """Test different output formats."""

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_json_output(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test --json output format."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = mock_tools
        mock_analyze.return_value = {}
        mock_subsets.return_value = {}

        with patch.object(sys, "argv", ["analyze_tools.py", "--json"]):
            main()

        captured = capsys.readouterr()
        # Should be valid JSON
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "tdc_Tool1_predict"
        assert data[0]["parameters"] == ["Drug SMILES"]

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_verbose_output(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test --verbose output includes details."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = mock_tools
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2)],
        }
        mock_subsets.return_value = {
            "drug_discovery": ["tdc_Tool1_predict", "ToxCast_Tool2_predict"],
        }

        with patch.object(sys, "argv", ["analyze_tools.py", "--verbose"]):
            main()

        captured = capsys.readouterr()
        # Verbose should show parameter details
        assert "Details:" in captured.out
        assert "string" in captured.out


# -------------------------
# Error Handling Tests
# -------------------------


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    def test_invalid_regex(self, mock_build_tools, mock_get_loader, mock_loader, capsys):
        """Test invalid regex pattern in --exclude."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.side_effect = ValueError("Invalid regex pattern")

        with patch.object(sys, "argv", ["analyze_tools.py", "--exclude", "[invalid("]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_keyboard_interrupt(self):
        """Test keyboard interrupt handling.

        NOTE: KeyboardInterrupt is caught in the if __name__ == "__main__" block,
        not inside main(). Testing this properly would require subprocess testing
        or restructuring the code. This test documents the limitation.

        The actual KeyboardInterrupt handling is in analyze_tools.py:373-375:
            except KeyboardInterrupt:
                print("\\n\\nInterrupted by user")
                sys.exit(1)

        For now, we skip testing this edge case as it would require:
        - subprocess.run() to test the actual script execution
        - or restructuring to move exception handling into main()
        """
        pytest.skip("KeyboardInterrupt handling requires subprocess testing")

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    def test_general_exception(self, mock_build_tools, mock_get_loader, mock_loader, capsys):
        """Test handling of general exceptions.

        NOTE: General exception handling is in the if __name__ == "__main__" block,
        not inside main(). Testing this would require subprocess testing.
        """
        pytest.skip("General exception handling requires subprocess testing")

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    def test_verbose_exception(self, mock_build_tools, mock_get_loader, mock_loader, capsys):
        """Test verbose mode shows traceback.

        NOTE: Same as test_general_exception - requires subprocess testing.
        """
        pytest.skip("Verbose exception handling requires subprocess testing")


# -------------------------
# Integration Tests
# -------------------------


class TestIntegration:
    """Integration-like tests for realistic scenarios."""

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_realistic_workflow(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test a realistic workflow: filter, exclude, and get JSON output."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = [mock_tools[0]]
        mock_analyze.return_value = {}
        mock_subsets.return_value = {}

        with patch.object(
            sys,
            "argv",
            [
                "analyze_tools.py",
                "--placeholder",
                "Drug SMILES",
                "--simple",
                "--exclude",
                "^ToxCast",
                "--json",
            ],
        ):
            main()

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        # Should only have non-ToxCast tool
        assert len(data) == 1
        assert data[0]["name"] == "tdc_Tool1_predict"
        assert "ToxCast" not in data[0]["name"]

    @patch("analyze_tools.get_loader")
    @patch("analyze_tools.build_tools")
    @patch("analyze_tools.analyze_tools")
    @patch("analyze_tools.suggest_tool_subsets")
    def test_statistics_display(
        self,
        mock_subsets,
        mock_analyze,
        mock_build_tools,
        mock_get_loader,
        mock_loader,
        mock_tools,
        capsys,
    ):
        """Test that statistics are displayed correctly."""
        mock_get_loader.return_value = mock_loader
        mock_build_tools.return_value = [mock_tools[0]]
        mock_analyze.return_value = {
            "total_tools": 2,
            "total_placeholders": 2,
            "simple_tools": 2,
            "complex_tools": 0,
            "tools_by_complexity": {1: 2},
            "most_common_placeholders": [("Drug SMILES", 2), ("Target sequence", 1)],
        }
        mock_subsets.return_value = {
            "drug_discovery": ["tdc_Tool1_predict"],
        }

        with patch.object(sys, "argv", ["analyze_tools.py", "--exclude", "^ToxCast"]):
            main()

        captured = capsys.readouterr()

        # Should show both total and filtered counts
        assert "Total tools available:" in captured.out
        assert "Tools after filtering:" in captured.out
        assert "Unique placeholders:" in captured.out
        assert "Tool Statistics" in captured.out
        assert "Suggested Tool Subsets" in captured.out
