"""
Tests for txgemma.prompts module.
"""

import json
import pytest
from pathlib import Path

from txgemma.prompts import (
    PromptTemplate,
    PromptLoader,
    get_loader,
    PLACEHOLDER_REGEX,
)


# =============================================================================
# PromptTemplate Tests
# =============================================================================

class TestPromptTemplate:
    """Tests for PromptTemplate class."""
    
    def test_init_basic(self):
        """Test basic template initialization."""
        template = PromptTemplate(
            name="test_tool",
            template="Question: What is {param1}?"
        )
        
        assert template.name == "test_tool"
        assert template.template == "Question: What is {param1}?"
        assert template.metadata == {}
        assert template.placeholders == ["param1"]
    
    def test_init_with_metadata(self):
        """Test template with metadata."""
        metadata = {"category": "test", "version": "1.0"}
        template = PromptTemplate(
            name="test_tool",
            template="Question: {input}",
            metadata=metadata
        )
        
        assert template.metadata == metadata
        assert template.metadata["category"] == "test"
    
    def test_extract_placeholders_multiple(self):
        """Test placeholder extraction with multiple placeholders."""
        template = PromptTemplate(
            "test",
            "Drug: {Drug SMILES}, Target: {Target sequence}, Phase: {Trial phase}"
        )
        
        assert template.placeholders == ["Drug SMILES", "Target sequence", "Trial phase"]
        assert template.placeholder_count() == 3
    
    def test_extract_placeholders_duplicates(self):
        """Test that duplicate placeholders are handled correctly."""
        template = PromptTemplate(
            "test",
            "First {Drug SMILES}, second {Drug SMILES}, third {Other}"
        )
        
        # Should preserve order and remove duplicates
        assert template.placeholders == ["Drug SMILES", "Other"]
        assert template.placeholder_count() == 2
    
    def test_extract_placeholders_none(self):
        """Test template with no placeholders."""
        template = PromptTemplate(
            "test",
            "This template has no placeholders"
        )
        
        assert template.placeholders == []
        assert template.placeholder_count() == 0
    
    def test_required_inputs(self):
        """Test required_inputs property."""
        template = PromptTemplate(
            "test",
            "{Drug SMILES} and {Target sequence}"
        )
        
        required = template.required_inputs
        assert isinstance(required, set)
        assert required == {"Drug SMILES", "Target sequence"}
    
    def test_has_placeholder(self):
        """Test has_placeholder method."""
        template = PromptTemplate(
            "test",
            "{Drug SMILES} and {Target sequence}"
        )
        
        assert template.has_placeholder("Drug SMILES")
        assert template.has_placeholder("Target sequence")
        assert not template.has_placeholder("Other")
    
    def test_format_success(self):
        """Test successful template formatting."""
        template = PromptTemplate(
            "test",
            "Drug: {Drug SMILES}, Target: {Target sequence}"
        )
        
        result = template.format(
            **{"Drug SMILES": "CC(=O)O", "Target sequence": "MKTAYIAK"}
        )
        
        assert "CC(=O)O" in result
        assert "MKTAYIAK" in result
    
    def test_format_missing_placeholder(self):
        """Test formatting with missing placeholder."""
        template = PromptTemplate(
            "test",
            "{Drug SMILES} and {Target sequence}"
        )
        
        with pytest.raises(ValueError, match="Missing required placeholders"):
            template.format(**{"Drug SMILES": "CC(=O)O"})
    
    def test_get_description_from_metadata(self):
        """Test description from metadata."""
        template = PromptTemplate(
            "test",
            "Question: {input}",
            metadata={"description": "Test description"}
        )
        
        assert template.get_description() == "Test description"
    
    def test_get_description_from_context(self):
        """Test description extraction from Context line."""
        template = PromptTemplate(
            "test",
            "Context: This is the context.\nQuestion: {input}"
        )
        
        assert template.get_description() == "This is the context."
    
    def test_get_description_fallback(self):
        """Test fallback description."""
        template = PromptTemplate(
            "test_tool",
            "Question: {input}"
        )
        
        assert "test_tool" in template.get_description()
    
    def test_to_metadata(self):
        """Test metadata export."""
        template = PromptTemplate(
            "test",
            "{Drug SMILES} and {Target sequence}",
            metadata={"category": "test"}
        )
        
        meta = template.to_metadata()
        
        assert meta["name"] == "test"
        assert "description" in meta
        assert meta["placeholder_count"] == 2
        assert set(meta["required_inputs"]) == {"Drug SMILES", "Target sequence"}
    
    def test_str_representation(self):
        """Test string representation."""
        template = PromptTemplate(
            "test",
            "{Drug SMILES}",
            metadata={"description": "Test description"}
        )
        
        str_repr = str(template)
        
        assert "test" in str_repr
        assert "Drug SMILES" in str_repr
        assert "placeholders=1" in str_repr
    
    def test_repr_representation(self):
        """Test repr representation."""
        template = PromptTemplate(
            "test",
            "{Drug SMILES}",
            metadata={"category": "test"}
        )
        
        repr_str = repr(template)
        
        assert "PromptTemplate" in repr_str
        assert "'test'" in repr_str
        assert "Drug SMILES" in repr_str


# =============================================================================
# PromptLoader Tests
# =============================================================================

class TestPromptLoader:
    """Tests for PromptLoader class."""
    
    def test_init_defaults(self):
        """Test loader initialization with defaults."""
        loader = PromptLoader()
        
        assert loader.hf_repo == "google/txgemma-2b-predict"
        assert loader.filename == "tdc_prompts.json"
        assert loader.local_override is None
        assert not loader._loaded
    
    def test_init_custom(self):
        """Test loader with custom settings."""
        loader = PromptLoader(
            hf_repo="custom/repo",
            filename="custom.json",
            local_override=Path("test.json")
        )
        
        assert loader.hf_repo == "custom/repo"
        assert loader.filename == "custom.json"
        assert loader.local_override == Path("test.json")
    
    def test_load_from_local_simple_format(self, tmp_path):
        """Test loading from local file with simple format."""
        prompts_file = tmp_path / "test_prompts.json"
        prompts_file.write_text(json.dumps({
            "test_tool": "Question: What is {Drug SMILES}?"
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        loader.load()
        
        assert loader._loaded
        assert len(loader) == 1
        assert "test_tool" in loader
    
    def test_load_from_local_rich_format(self, tmp_path):
        """Test loading from local file with rich format."""
        prompts_file = tmp_path / "test_prompts.json"
        prompts_file.write_text(json.dumps({
            "test_tool": {
                "template": "Question: {Drug SMILES}",
                "metadata": {"category": "test"}
            }
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        loader.load()
        
        template = loader.get("test_tool")
        assert template.metadata["category"] == "test"
    
    def test_load_from_local_mixed_format(self, tmp_path):
        """Test loading with mixed simple and rich formats."""
        prompts_file = tmp_path / "test_prompts.json"
        prompts_file.write_text(json.dumps({
            "simple": "Question: {input}",
            "rich": {
                "template": "Question: {input}",
                "metadata": {"key": "value"}
            }
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        loader.load()
        
        assert len(loader) == 2
        assert loader.get("simple").metadata == {}
        assert loader.get("rich").metadata["key"] == "value"
    
    def test_load_file_not_found(self, tmp_path):
        """Test error when local file doesn't exist."""
        loader = PromptLoader(local_override=tmp_path / "nonexistent.json")
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_load_invalid_json(self, tmp_path):
        """Test error with invalid JSON."""
        prompts_file = tmp_path / "bad.json"
        prompts_file.write_text("{ invalid json }")
        
        loader = PromptLoader(local_override=prompts_file)
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load()
    
    def test_load_invalid_top_level_type(self, tmp_path):
        """Test error when top-level JSON is not a dict."""
        prompts_file = tmp_path / "bad.json"
        prompts_file.write_text(json.dumps(["not", "a", "dict"]))
        
        loader = PromptLoader(local_override=prompts_file)
        
        with pytest.raises(ValueError, match="must be a dictionary"):
            loader.load()
    
    def test_load_missing_template_field(self, tmp_path):
        """Test error when rich format missing template field."""
        prompts_file = tmp_path / "bad.json"
        prompts_file.write_text(json.dumps({
            "tool": {"metadata": {"key": "value"}}
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        
        with pytest.raises(ValueError, match="missing 'template' field"):
            loader.load()
    
    def test_load_invalid_prompt_type(self, tmp_path):
        """Test error with invalid prompt content type."""
        prompts_file = tmp_path / "bad.json"
        prompts_file.write_text(json.dumps({
            "tool": 123  # Invalid type
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        
        with pytest.raises(ValueError, match="expected str or dict"):
            loader.load()
    
    def test_load_only_once(self, tmp_path):
        """Test that load() is idempotent."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({"tool": "Question: {input}"}))
        
        loader = PromptLoader(local_override=prompts_file)
        
        loader.load()
        first_templates = dict(loader._templates)
        
        loader.load()  # Should not reload
        second_templates = dict(loader._templates)
        
        assert first_templates == second_templates
    
    def test_reload(self, tmp_path):
        """Test reload functionality."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({"tool1": "Question: {input}"}))
        
        loader = PromptLoader(local_override=prompts_file)
        loader.load()
        
        assert len(loader) == 1
        
        # Update file
        prompts_file.write_text(json.dumps({
            "tool1": "Updated: {input}",
            "tool2": "New: {input}"
        }))
        
        loader.reload()
        
        assert len(loader) == 2
        assert "tool2" in loader
    
    def test_get_existing(self, tmp_path):
        """Test getting an existing template."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({"tool": "Question: {input}"}))
        
        loader = PromptLoader(local_override=prompts_file)
        template = loader.get("tool")
        
        assert template.name == "tool"
    
    def test_get_nonexistent(self, tmp_path):
        """Test getting a non-existent template."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({"tool": "Question: {input}"}))
        
        loader = PromptLoader(local_override=prompts_file)
        
        with pytest.raises(KeyError, match="not found"):
            loader.get("nonexistent")
    
    def test_has_template(self, tmp_path):
        """Test has_template method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({"tool": "Question: {input}"}))
        
        loader = PromptLoader(local_override=prompts_file)
        
        assert loader.has_template("tool")
        assert not loader.has_template("nonexistent")
    
    def test_contains_operator(self, tmp_path):
        """Test 'in' operator."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({"tool": "Question: {input}"}))
        
        loader = PromptLoader(local_override=prompts_file)
        
        assert "tool" in loader
        assert "nonexistent" not in loader
    
    def test_len(self, tmp_path):
        """Test len() function."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "Question: {input}",
            "tool2": "Question: {input}",
            "tool3": "Question: {input}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        
        assert len(loader) == 3
    
    def test_all(self, tmp_path):
        """Test all() method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "Q1",
            "tool2": "Q2"
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        templates = loader.all()
        
        assert isinstance(templates, dict)
        assert len(templates) == 2
        assert all(isinstance(t, PromptTemplate) for t in templates.values())
    
    def test_list(self, tmp_path):
        """Test list() method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "Q1",
            "tool2": "Q2"
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        names = loader.list()
        
        assert isinstance(names, list)
        assert set(names) == {"tool1", "tool2"}
    
    def test_placeholder_index(self, tmp_path):
        """Test that placeholder index is built correctly."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "Question: {Drug SMILES}",
            "tool2": "Question: {Drug SMILES} and {Target sequence}",
            "tool3": "Question: {Target sequence}"
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        loader.load()
        
        # Drug SMILES should be in tool1 and tool2
        assert loader._placeholder_index["Drug SMILES"] == {"tool1", "tool2"}
        
        # Target sequence should be in tool2 and tool3
        assert loader._placeholder_index["Target sequence"] == {"tool2", "tool3"}
    
    def test_all_placeholders(self, tmp_path):
        """Test all_placeholders method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Drug SMILES}",
            "tool2": "{Target sequence}",
            "tool3": "{Drug SMILES} and {Indication}"
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        placeholders = loader.all_placeholders()
        
        assert placeholders == {"Drug SMILES", "Target sequence", "Indication"}
    
    def test_placeholder_usage(self, tmp_path):
        """Test placeholder_usage method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Drug SMILES}",
            "tool2": "{Drug SMILES} and {Target sequence}",
            "tool3": "{Other}"
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        
        usage = loader.placeholder_usage("Drug SMILES")
        assert usage == {"tool1", "tool2"}
        
        usage = loader.placeholder_usage("Target sequence")
        assert usage == {"tool2"}
        
        usage = loader.placeholder_usage("Nonexistent")
        assert usage == set()
    
    def test_placeholder_stats(self, tmp_path):
        """Test placeholder_stats method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Drug SMILES}",
            "tool2": "{Drug SMILES} and {Target sequence}",
            "tool3": "{Drug SMILES}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        stats = loader.placeholder_stats()
        
        assert stats["Drug SMILES"] == 3
        assert stats["Target sequence"] == 1
    
    def test_most_common_placeholders(self, tmp_path):
        """Test most_common_placeholders method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{A}",
            "tool2": "{A} and {B}",
            "tool3": "{A} and {C}",
            "tool4": "{B} and {C}",
            "tool5": "{C}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        common = loader.most_common_placeholders(top_n=2)
        
        # A appears 3 times, C appears 3 times, B appears 2 times
        assert len(common) == 2
        assert common[0][1] == 3  # Most common has count 3
        assert common[1][1] in [2, 3]  # Second has count 2 or 3
    
    def test_filter_by_placeholder_exact(self, tmp_path):
        """Test exact placeholder filtering."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Drug SMILES}",
            "tool2": "{Product SMILES}",
            "tool3": "{Drug SMILES} and {Other}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        filtered = loader.filter_by_placeholder("Drug SMILES", exact=True)
        
        assert set(filtered.keys()) == {"tool1", "tool3"}
    
    def test_filter_by_placeholder_fuzzy(self, tmp_path):
        """Test fuzzy placeholder filtering."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Drug SMILES}",
            "tool2": "{Product SMILES}",
            "tool3": "{Target sequence}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        filtered = loader.filter_by_placeholder("smiles", exact=False)
        
        # Should match both Drug SMILES and Product SMILES
        assert set(filtered.keys()) == {"tool1", "tool2"}
    
    def test_filter_by_placeholders_all(self, tmp_path):
        """Test filtering with ALL placeholders required."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Drug SMILES}",
            "tool2": "{Drug SMILES} and {Target sequence}",
            "tool3": "{Target sequence}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        filtered = loader.filter_by_placeholders(
            ["Drug SMILES", "Target sequence"],
            match_all=True
        )
        
        # Only tool2 has both
        assert set(filtered.keys()) == {"tool2"}
    
    def test_filter_by_placeholders_any(self, tmp_path):
        """Test filtering with ANY placeholder matching."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Drug SMILES}",
            "tool2": "{Drug SMILES} and {Target sequence}",
            "tool3": "{Target sequence}",
            "tool4": "{Other}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        filtered = loader.filter_by_placeholders(
            ["Drug SMILES", "Target sequence"],
            match_all=False
        )
        
        # tool1, tool2, tool3 have at least one
        assert set(filtered.keys()) == {"tool1", "tool2", "tool3"}
    
    def test_smiles_prompts(self, tmp_path):
        """Test smiles_prompts convenience method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Drug SMILES}",
            "tool2": "{Target sequence}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        smiles = loader.smiles_prompts()
        
        assert set(smiles.keys()) == {"tool1"}
    
    def test_sequence_prompts(self, tmp_path):
        """Test sequence_prompts convenience method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{Target sequence}",
            "tool2": "{Protein sequence}",
            "tool3": "{Drug SMILES}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        sequences = loader.sequence_prompts()
        
        # Should match anything with "sequence" (case-insensitive)
        assert set(sequences.keys()) == {"tool1", "tool2"}
    
    def test_simple_prompts(self, tmp_path):
        """Test simple_prompts method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{A}",
            "tool2": "{A} and {B}",
            "tool3": "{A}, {B}, and {C}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        simple = loader.simple_prompts(max_placeholders=1)
        
        assert set(simple.keys()) == {"tool1"}
    
    def test_complex_prompts(self, tmp_path):
        """Test complex_prompts method."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({
            "tool1": "{A}",
            "tool2": "{A} and {B}",
            "tool3": "{A}, {B}, and {C}",
            "tool4": "{A}, {B}, {C}, and {D}",
        }))
        
        loader = PromptLoader(local_override=prompts_file)
        complex_tools = loader.complex_prompts(min_placeholders=3)
        
        assert set(complex_tools.keys()) == {"tool3", "tool4"}
    
    def test_source_property(self, tmp_path):
        """Test source property."""
        prompts_file = tmp_path / "test.json"
        prompts_file.write_text(json.dumps({"tool": "Q"}))
        
        loader = PromptLoader(local_override=prompts_file)
        
        assert loader.source is None  # Not loaded yet
        
        loader.load()
        
        assert loader.source is not None
        assert str(prompts_file) in loader.source


# =============================================================================
# Global Loader Tests
# =============================================================================

class TestGlobalLoader:
    """Tests for global loader function."""
    
    def test_get_loader_singleton(self):
        """Test that get_loader returns same instance."""
        loader1 = get_loader()
        loader2 = get_loader()
        
        assert loader1 is loader2
    
    def test_get_loader_default_config(self):
        """Test that global loader has default configuration."""
        loader = get_loader()
        
        assert loader.hf_repo == "google/txgemma-2b-predict"
        assert loader.filename == "tdc_prompts.json"
        assert loader.local_override is None


# =============================================================================
# Regex Tests
# =============================================================================

class TestPlaceholderRegex:
    """Tests for PLACEHOLDER_REGEX."""
    
    def test_simple_placeholder(self):
        """Test matching simple placeholder."""
        text = "Question: {input}"
        matches = PLACEHOLDER_REGEX.findall(text)
        
        assert matches == ["input"]
    
    def test_multiple_placeholders(self):
        """Test matching multiple placeholders."""
        text = "{Drug SMILES} and {Target sequence}"
        matches = PLACEHOLDER_REGEX.findall(text)
        
        assert matches == ["Drug SMILES", "Target sequence"]
    
    def test_placeholder_with_spaces(self):
        """Test placeholder with spaces in name."""
        text = "{Epitope amino acid sequence}"
        matches = PLACEHOLDER_REGEX.findall(text)
        
        assert matches == ["Epitope amino acid sequence"]
    
    def test_no_placeholders(self):
        """Test text with no placeholders."""
        text = "This has no placeholders"
        matches = PLACEHOLDER_REGEX.findall(text)
        
        assert matches == []
    
    def test_nested_braces(self):
        """Test handling of double braces (Python format string escaping)."""
        # {{placeholder}} becomes {placeholder} after one level of escaping
        # The regex will find 'not_a_placeholder' between the inner braces
        text = "Text {placeholder} more text {{not_a_placeholder}}"
        matches = PLACEHOLDER_REGEX.findall(text)
        
        # Both match: 'placeholder' from {placeholder} and 'not_a_placeholder' from {{not_a_placeholder}}
        # This is expected because the regex sees two separate {..} patterns
        assert matches == ["placeholder", "not_a_placeholder"]
    
    def test_only_valid_placeholders(self):
        """Test that only properly formatted placeholders are matched."""
        text = "{Drug SMILES} {Target sequence} {Trial phase}"
        matches = PLACEHOLDER_REGEX.findall(text)
        
        assert len(matches) == 3
        assert "Drug SMILES" in matches
        
    def test_no_nested_braces_in_placeholder(self):
        """Test that placeholders with braces inside are partially matched."""
        # The pattern {outer {inner} outer} contains {inner} which will match
        text = "{outer {inner} outer}"
        matches = PLACEHOLDER_REGEX.findall(text)
        
        # The regex will match 'inner' from the innermost {...}
        assert matches == ["inner"]
    
    def test_proper_placeholder_only(self):
        """Test clean, well-formed placeholders."""
        text = "Question: What is the {Drug SMILES} for {Indication}?"
        matches = PLACEHOLDER_REGEX.findall(text)
        
        assert matches == ["Drug SMILES", "Indication"]