# Tool Filtering Guide

TxGemma-MCP supports flexible tool filtering to create focused tool subsets for specific use cases.

## Table of Contents

1. [Why Filter Tools?](#why-filter-tools)
2. [Quick Examples](#quick-examples)
3. [Filtering Methods](#filtering-methods)
4. [CLI Tool Usage](#cli-tool-usage)
5. [Use Cases](#use-cases)
6. [Programmatic Usage](#programmatic-usage)

---

## Why Filter Tools?

**Performance**: Loading only needed tools reduces:
- MCP client initialization time
- Memory usage
- Tool selection complexity for AI models

**Focus**: Specialized tool subsets for:
- Drug discovery workflows (Drug SMILES only)
- Protein analysis (sequence-based tools)
- Simple predictions (single-parameter tools)

**Domain Expertise**: Different scientists need different tools:
- Medicinal chemists → Drug SMILES tools
- Structural biologists → Sequence tools
- Computational chemists → Retrosynthesis tools

---

## Quick Examples

### Load All Tools (Default)

```python
from txgemma.tool_factory import build_tools

tools = build_tools()
# Returns ALL available tools
```

### Load Only Drug SMILES Tools

```python
tools = build_tools(filter_placeholder="Drug SMILES")
# Returns only tools that use Drug SMILES input
```

### Load Drug-Target Interaction Tools

```python
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Target sequence"],
    match_all=True
)
# Returns only tools that use BOTH Drug SMILES AND Target sequence
```

### Load Simple Tools

```python
tools = build_tools(max_placeholders=2)
# Returns only tools with ≤2 input parameters
```

### Fuzzy Matching

```python
tools = build_tools(
    filter_placeholder="sequence",
    exact_match=False
)
# Returns tools with ANY placeholder containing "sequence"
# Matches: "Target sequence", "Protein sequence", "Epitope amino acid sequence", etc.
```

---

## Filtering Methods

### 1. By Single Placeholder

**Exact Match:**
```python
tools = build_tools(filter_placeholder="Drug SMILES")
```

**Fuzzy Match:**
```python
tools = build_tools(
    filter_placeholder="smiles",
    exact_match=False
)
# Matches "Drug SMILES", "Product SMILES", "Molecule SMILES", etc.
```

### 2. By Multiple Placeholders

**ALL Required (AND logic):**
```python
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Target sequence"],
    match_all=True
)
# Tool must use BOTH placeholders
```

**ANY Required (OR logic):**
```python
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Protein sequence"],
    match_all=False
)
# Tool must use AT LEAST ONE placeholder
```

### 3. By Complexity

**Simple Tools:**
```python
tools = build_tools(max_placeholders=1)
# Only tools with exactly 1 parameter
```

**Exclude Complex:**
```python
tools = build_tools(exclude_complex=True)
# Excludes tools with >2 parameters
```

**Complex Tools Only:**
```python
all_tools = build_tools()
complex_tools = [
    t for t in all_tools 
    if len(t.inputSchema["required"]) >= 3
]
```

---

## CLI Tool Usage

The `analyze_tools.py` script provides powerful exploration capabilities.

### Basic Commands

#### List All Placeholders

```bash
python scripts/analyze_tools.py --list-placeholders
```

**Output:**
```
======================================================================
  Available Placeholders
======================================================================

Found 15 unique placeholders

  📌 Drug SMILES                              (12 tools)
  📌 Target sequence                          (5 tools)
  📌 Indication                               (4 tools)
  📌 Trial phase                              (2 tools)
  ...
```

**With verbose details:**
```bash
python scripts/analyze_tools.py --list-placeholders --verbose
```

**Output:**
```
  📌 Drug SMILES                              (12 tools)
     Used in: predict_toxicity, predict_bbb_permeability, predict_solubility
              ... and 9 more
```

#### Show All Tools

```bash
python scripts/analyze_tools.py
```

**Output:**
```
======================================================================
  Tools (all)
======================================================================

Found 15 tools

  📦 predict_toxicity
     Predict toxicity of a drug molecule from its SMILES representation
     Parameters (1): Drug SMILES

  📦 predict_bbb_permeability
     Predict blood-brain barrier permeability for CNS drug development
     Parameters (1): Drug SMILES
  
  ...
```

### Filtering Commands

#### Show Tools Using Specific Placeholder

```bash
python scripts/analyze_tools.py --placeholder "Drug SMILES"
```

**Output:**
```
======================================================================
  Tools (using 'Drug SMILES' (exact match))
======================================================================

Found 12 tools

  📦 predict_toxicity
     Predict toxicity of a drug molecule
     Parameters (1): Drug SMILES
  
  ...
```

#### Fuzzy Search for Placeholders

```bash
python scripts/analyze_tools.py --placeholder "smiles" --fuzzy
```

**Output:**
```
======================================================================
  Tools (using 'smiles' (fuzzy match))
======================================================================

Found 15 tools

  📦 predict_toxicity
     Parameters (1): Drug SMILES

  📦 retrosynthesis
     Parameters (1): Product SMILES
  
  ...
```

#### Show Simple Tools Only

```bash
python scripts/analyze_tools.py --simple
```

**Output:**
```
======================================================================
  Tools (simple (≤2 placeholders))
======================================================================

Found 10 tools

  📦 predict_toxicity
     Parameters (1): Drug SMILES
  
  ...
```

#### Show Complex Tools Only

```bash
python scripts/analyze_tools.py --complex
```

#### Show Tools with Multiple Placeholders (ALL)

```bash
python scripts/analyze_tools.py \
    --placeholders "Drug SMILES" "Target sequence"
```

**Finds tools that use BOTH placeholders.**

#### Show Tools with Multiple Placeholders (ANY)

```bash
python scripts/analyze_tools.py \
    --placeholders "Drug SMILES" "Protein sequence" --any
```

**Finds tools that use AT LEAST ONE of the placeholders.**

### Detailed Tool Information

#### Inspect Specific Template

```bash
python scripts/analyze_tools.py --template "predict_toxicity"
```

**Output:**
```
======================================================================
  Template: predict_toxicity
======================================================================

Description:
  Predict toxicity of a drug molecule from its SMILES representation

Placeholders (1):
  - Drug SMILES

Metadata:
  category: safety
  description: Predict toxicity...

Template Preview:
   1. Instruction: Predict the toxicity of the given drug molecule.
   2. Context: Drug toxicity prediction is critical for early-stage...
   3. Question: Given the drug SMILES '{Drug SMILES}', predict its...
   4. Answer:

Used by:
  Tool name: predict_toxicity
  Description: Predict toxicity of a drug molecule...
  Parameters: Drug SMILES
```

#### Show Prompt Source

```bash
python scripts/analyze_tools.py --source
```

**Output:**
```
======================================================================
  Prompt Source
======================================================================

  Loaded from: HuggingFace: google/txgemma-2b-predict/tdc_prompts.json
  Total templates: 15
```

### Output Formats

#### JSON Output

```bash
python scripts/analyze_tools.py --json
```

**Output:**
```json
[
  {
    "name": "predict_toxicity",
    "description": "Predict toxicity...",
    "parameters": ["Drug SMILES"],
    "parameter_count": 1,
    "properties": {
      "Drug SMILES": {
        "type": "string",
        "description": "SMILES string representation..."
      }
    }
  }
]
```

**Save to file:**
```bash
python scripts/analyze_tools.py --json > tools.json
python scripts/analyze_tools.py --placeholder "Drug SMILES" --json > drug_tools.json
```

#### Verbose Output

```bash
python scripts/analyze_tools.py --simple --verbose
```

**Shows detailed parameter information:**
```
  📦 predict_toxicity
     Predict toxicity of a drug molecule
     Parameters (1): Drug SMILES
     Details:
       - Drug SMILES (string): SMILES string representation of the drug molecule
```

### Common CLI Patterns

```bash
# Quick exploration
python scripts/analyze_tools.py --list-placeholders
python scripts/analyze_tools.py --source

# Find tools for specific use case
python scripts/analyze_tools.py --placeholder "Drug SMILES"
python scripts/analyze_tools.py --simple

# Detailed investigation
python scripts/analyze_tools.py --template "predict_toxicity"
python scripts/analyze_tools.py --placeholder "sequence" --fuzzy --verbose

# Export for documentation
python scripts/analyze_tools.py --json > docs/tools.json
python scripts/analyze_tools.py --list-placeholders --json > docs/placeholders.json
```

---

## Use Cases

### Use Case 1: Drug Discovery Pipeline

**Goal**: Only tools for SMILES-based drug prediction

```python
# In server.py
from txgemma.tool_factory import build_tools

TOOLS = build_tools(filter_placeholder="Drug SMILES")
```

**CLI:**
```bash
python scripts/analyze_tools.py --placeholder "Drug SMILES"
```

**Result**: Tools like:
- `predict_toxicity`
- `predict_bbb_permeability`
- `predict_bioavailability`
- `predict_solubility`

### Use Case 2: Protein Engineering

**Goal**: Only sequence-based tools

```python
TOOLS = build_tools(
    filter_placeholder="sequence",
    exact_match=False
)
```

**CLI:**
```bash
python scripts/analyze_tools.py --placeholder "sequence" --fuzzy
```

**Result**: Tools using any sequence input:
- `predict_drug_target_interaction` (Target sequence)
- `predict_epitope_binding` (Epitope amino acid sequence)
- `predict_protein_property` (Protein sequence)

### Use Case 3: Simple Screening

**Goal**: Quick single-input predictions only

```python
TOOLS = build_tools(max_placeholders=1)
```

**CLI:**
```bash
python scripts/analyze_tools.py --simple
```

**Result**: Only tools with 1 parameter, like:
- `predict_toxicity(Drug SMILES)`
- `predict_solubility(Drug SMILES)`

### Use Case 4: Drug-Target Analysis

**Goal**: Only drug-protein interaction tools

```python
TOOLS = build_tools(
    filter_placeholders=["Drug SMILES", "Target sequence"],
    match_all=True
)
```

**CLI:**
```bash
python scripts/analyze_tools.py \
    --placeholders "Drug SMILES" "Target sequence"
```

**Result**: Tools requiring both drug and target:
- `predict_drug_target_interaction`
- `predict_binding_affinity`

### Use Case 5: Clinical Development

**Goal**: Tools for clinical predictions

```python
TOOLS = build_tools(filter_placeholder="Trial phase")
```

**CLI:**
```bash
python scripts/analyze_tools.py --placeholder "Trial phase"
```

**Result**: Clinical trial tools:
- `predict_clinical_trial_outcome`
- `predict_adverse_events`

---

## Programmatic Usage

### Programmatic Discovery

#### Find All Available Placeholders

```python
from txgemma.prompts import get_loader

loader = get_loader()

# Get all placeholders
placeholders = loader.all_placeholders()
print(placeholders)
# {'Drug SMILES', 'Target sequence', 'Indication', ...}
```

#### Check Placeholder Usage

```python
# Which tools use "Drug SMILES"?
tools = loader.placeholder_usage("Drug SMILES")
print(f"Drug SMILES used in {len(tools)} tools")
# Drug SMILES used in 12 tools

# Get the actual tool names
print(sorted(tools))
# ['predict_bioavailability', 'predict_toxicity', ...]
```

#### Get Usage Statistics

```python
stats = loader.placeholder_stats()
for placeholder, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{placeholder}: {count} tools")
```

**Output:**
```
Drug SMILES: 12 tools
Target sequence: 5 tools
Indication: 4 tools
Trial phase: 2 tools
Cell line: 2 tools
```

#### Find Most Common Placeholders

```python
top = loader.most_common_placeholders(5)
for placeholder, count in top:
    print(f"{placeholder}: {count}")
```

#### Check if Template Exists

```python
# Check without raising error
if loader.has_template("predict_toxicity"):
    template = loader.get("predict_toxicity")

# Or use 'in' operator
if "predict_toxicity" in loader:
    template = loader.get("predict_toxicity")
```

#### Get Template Details

```python
template = loader.get("predict_toxicity")

print(f"Name: {template.name}")
print(f"Placeholders: {template.placeholders}")
print(f"Count: {template.placeholder_count()}")
print(f"Description: {template.get_description()}")

# Format with values
prompt = template.format(**{"Drug SMILES": "CC(=O)O"})
```

### Advanced Filtering Patterns

#### Dynamic Tool Loading

```python
from txgemma.tool_factory import build_tools

class DynamicToolServer:
    def __init__(self):
        self.tool_cache = {}
    
    def get_tools(self, use_case: str):
        if use_case not in self.tool_cache:
            if use_case == "drug_discovery":
                self.tool_cache[use_case] = build_tools(
                    filter_placeholder="Drug SMILES"
                )
            elif use_case == "protein_analysis":
                self.tool_cache[use_case] = build_tools(
                    filter_placeholder="sequence",
                    exact_match=False
                )
        return self.tool_cache[use_case]
```

#### User-Specific Tool Subsets

```python
def get_tools_for_user(user_role: str):
    """Return tools appropriate for user's role."""
    if user_role == "medicinal_chemist":
        return build_tools(filter_placeholder="Drug SMILES")
    elif user_role == "structural_biologist":
        return build_tools(
            filter_placeholder="sequence",
            exact_match=False
        )
    elif user_role == "clinical_researcher":
        return build_tools(filter_placeholder="Trial phase")
    else:
        return build_tools()  # All tools for admins
```

#### Conditional Filtering

```python
import os

# Load different tools based on environment
if os.getenv("ENV") == "production":
    # Production: only well-tested simple tools
    TOOLS = build_tools(max_placeholders=2)
elif os.getenv("ENV") == "development":
    # Development: all tools for testing
    TOOLS = build_tools()
else:
    # Default: drug discovery focus
    TOOLS = build_tools(filter_placeholder="Drug SMILES")
```

---

## Performance Tips

### 1. Load Subsets at Startup

Instead of loading all tools and filtering later:

```python
# ❌ Inefficient - loads and builds all tools
all_tools = build_tools()
drug_tools = [t for t in all_tools if "Drug SMILES" in t.inputSchema["required"]]

# ✅ Efficient - only builds needed tools
drug_tools = build_tools(filter_placeholder="Drug SMILES")
```

### 2. Use `get_tool_names()` for Lightweight Checks

```python
from txgemma.tool_factory import get_tool_names

# Just get names (fast, no tool building)
names = get_tool_names(filter_placeholder="Drug SMILES")
print(f"Found {len(names)} matching tools")

# Only build if needed
if len(names) > 0:
    tools = build_tools(filter_placeholder="Drug SMILES")
```

### 3. Cache Tool Subsets

```python
# Build once, use many times
DRUG_TOOLS = build_tools(filter_placeholder="Drug SMILES")
PROTEIN_TOOLS = build_tools(filter_placeholder="sequence", exact_match=False)

# Use cached subsets
def get_tools_for_use_case(use_case: str):
    if use_case == "drug_discovery":
        return DRUG_TOOLS
    elif use_case == "protein_analysis":
        return PROTEIN_TOOLS
    # ...
```

---

## Configuration Examples

### Configuration 1: Focused Drug Discovery Server

```python
# server.py
TOOLS = build_tools(
    filter_placeholder="Drug SMILES",
    max_placeholders=2,  # Exclude complex multi-input tools
)
# Result: Simple drug prediction tools only
```

### Configuration 2: Comprehensive Research Server

```python
# server.py
TOOLS = build_tools()  # Load everything
# Result: All available tools for maximum flexibility
```

### Configuration 3: Production Screening Server

```python
# server.py
TOOLS = build_tools(
    max_placeholders=1,  # Single-input only
)
# Result: Fast, simple predictions for high-throughput screening
```

### Configuration 4: Multi-Modal Interaction Server

```python
# server.py
TOOLS = build_tools(
    filter_placeholders=["Drug SMILES", "Target sequence"],
    match_all=False,  # Tools using EITHER drug or protein
)
# Result: All molecular and protein analysis tools
```

---

## Summary

### Key Benefits of Filtering

- 🚀 Faster server startup
- 💾 Reduced memory usage
- 🎯 Focused tool selection
- 👥 Role-based access
- 📊 Better organization

### Filtering Dimensions

- By placeholder (exact or fuzzy)
- By multiple placeholders (AND/OR)
- By complexity (parameter count)
- Combinations of above

### Best Practices

1. **Filter at build time, not runtime** - More efficient
2. **Use CLI for exploration** - `analyze_tools.py` is your friend
3. **Use `get_tool_names()` for quick checks** - Lightweight
4. **Cache tool subsets for reuse** - Performance optimization
5. **Match filtering to use case** - Purpose-driven design
6. **Document your filtering choices** - Team clarity

### CLI Quick Reference

```bash
# Exploration
analyze_tools.py --list-placeholders
analyze_tools.py --source
analyze_tools.py --template "tool_name"

# Filtering
analyze_tools.py --placeholder "Drug SMILES"
analyze_tools.py --placeholder "smiles" --fuzzy
analyze_tools.py --placeholders "Drug SMILES" "Target sequence"
analyze_tools.py --simple
analyze_tools.py --complex

# Output
analyze_tools.py --json
analyze_tools.py --verbose
analyze_tools.py --json > output.json
```

---

**For more information, see:**
- `python scripts/analyze_tools.py --help`
- `txgemma/prompts.py` - Loader implementation
- `txgemma/tool_factory.py` - Tool building implementation