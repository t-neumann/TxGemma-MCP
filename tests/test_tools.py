"""
Tests for TxGemma tools and infrastructure.
"""

import pytest
from pathlib import Path

from txgemma.prompts import PromptTemplate, PromptLoader
from txgemma.tool_factory import build_tool_from_template, build_tools
from txgemma.model import TxGemmaModel

# -----------------------------------------------------------------------------
# Prompt Template Tests
# -----------------------------------------------------------------------------

def test_prompt_template_placeholders():
    """Test placeholder extraction from templates."""
    template = PromptTemplate(
        "test_tool",
        "Question: What is the {property_name} of {drug_smiles}?"
    )
    
    assert set(template.placeholders) == {"property_name", "drug_smiles"}


def test_prompt_template_format():
    """Test template formatting with values."""
    template = PromptTemplate(
        "test_tool",
        "Drug: {drug_smiles}\nProperty: {property_name}"
    )
    
    result = template.format(
        drug_smiles="CC(=O)O",
        property_name="toxicity"
    )
    
    assert "CC(=O)O" in result
    assert "toxicity" in result


def test_prompt_template_missing_placeholder():
    """Test that missing placeholders raise error."""
    template = PromptTemplate(
        "test_tool",
        "Drug: {drug_smiles}\nProperty: {property_name}"
    )
    
    with pytest.raises(ValueError, match="Missing required placeholders"):
        template.format(drug_smiles="CC(=O)O")  # Missing property_name


def test_prompt_template_description():
    """Test description extraction."""
    template = PromptTemplate(
        "test_tool",
        "Context: This predicts drug toxicity.\nQuestion: What is the toxicity?"
    )
    
    description = template.get_description()
    assert "toxicity" in description.lower()


# -----------------------------------------------------------------------------
# Prompt Loader Tests
# -----------------------------------------------------------------------------

def test_prompt_loader_loads_json(tmp_path):
    """Test loading prompts from JSON file."""
    # Create temporary prompts file
    prompts_file = tmp_path / "test_prompts.json"
    prompts_file.write_text('''
    {
        "predict_toxicity": "Question: Predict toxicity of {drug_smiles}",
        "predict_solubility": {
            "template": "Question: Predict solubility of {molecule_smiles}",
            "metadata": {"category": "admet"}
        }
    }
    ''')
    
    loader = PromptLoader(prompts_file)
    loader.load()
    
    assert len(loader.list_templates()) == 2
    assert "predict_toxicity" in loader.list_templates()
    assert "predict_solubility" in loader.list_templates()


def test_prompt_loader_get_template(tmp_path):
    """Test retrieving specific template."""
    prompts_file = tmp_path / "test_prompts.json"
    prompts_file.write_text('''
    {
        "test_tool": "Question: Test {input}"
    }
    ''')
    
    loader = PromptLoader(prompts_file)
    template = loader.get_template("test_tool")
    
    assert template.name == "test_tool"
    assert "input" in template.placeholders


def test_prompt_loader_unknown_template(tmp_path):
    """Test error on unknown template."""
    prompts_file = tmp_path / "test_prompts.json"
    prompts_file.write_text('{}')
    
    loader = PromptLoader(prompts_file)
    
    with pytest.raises(KeyError, match="not found"):
        loader.get_template("nonexistent")


# -----------------------------------------------------------------------------
# Tool Factory Tests
# -----------------------------------------------------------------------------

def test_build_tool_from_template():
    """Test MCP tool creation from template."""
    from txgemma.tool_factory import build_tool_from_template
    
    template = PromptTemplate(
        "predict_toxicity",
        "Context: Predict drug toxicity.\nQuestion: Predict {Drug SMILES} toxicity."
    )
    
    tool = build_tool_from_template(template)
    
    assert tool.name == "predict_toxicity"
    assert "Drug SMILES" in tool.inputSchema["properties"]
    assert "Drug SMILES" in tool.inputSchema["required"]


def test_build_tools_returns_list():
    """Test that build_tools returns a list of tools."""
    from txgemma.tool_factory import build_tools
    
    # This uses the real data/tdc_prompts.json file (or HF download)
    tools = build_tools()
    
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(hasattr(tool, "name") for tool in tools)


def test_build_tools_with_filter():
    """Test building tools with placeholder filter."""
    from txgemma.tool_factory import build_tools
    
    # Build only Drug SMILES tools
    tools = build_tools(filter_placeholder="Drug SMILES")
    
    assert isinstance(tools, list)
    # All tools should have "Drug SMILES" in their parameters
    for tool in tools:
        assert "Drug SMILES" in tool.inputSchema["required"]


def test_build_tools_with_max_placeholders():
    """Test building only simple tools."""
    from txgemma.tool_factory import build_tools
    
    # Build only tools with ≤2 placeholders
    tools = build_tools(max_placeholders=2)
    
    assert isinstance(tools, list)
    # All tools should have ≤2 parameters
    for tool in tools:
        assert len(tool.inputSchema["required"]) <= 2


def test_get_tool_names():
    """Test getting tool names with filter."""
    from txgemma.tool_factory import get_tool_names
    
    names = get_tool_names(filter_placeholder="Drug SMILES")
    
    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(name, str) for name in names)


def test_analyze_tools():
    """Test tool analysis."""
    from txgemma.tool_factory import analyze_tools
    
    stats = analyze_tools()
    
    assert "total_tools" in stats
    assert "total_placeholders" in stats
    assert "placeholder_usage" in stats
    assert stats["total_tools"] > 0


# -----------------------------------------------------------------------------
# Model Tests (require GPU, skipped by default)
# -----------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires GPU and model download")
def test_model_initialization():
    """Test model can be initialized."""
    model = TxGemmaModel()
    assert not model.is_loaded
    assert model.model_name == "google/txgemma-2b-predict"


@pytest.mark.skip(reason="Requires GPU and model download")
def test_model_singleton():
    """Test that model uses singleton pattern."""
    model1 = TxGemmaModel()
    model2 = TxGemmaModel()
    assert model1 is model2


@pytest.mark.skip(reason="Requires GPU and model download")
def test_model_generate():
    """Test model generation."""
    model = TxGemmaModel()
    result = model.generate("Question: What is 2+2?\nAnswer:")
    
    assert isinstance(result, str)
    assert len(result) > 0