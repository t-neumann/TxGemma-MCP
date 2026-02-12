<!--
TxGemma-MCP Tool Filtering Guide
Author: Tobias Neumann
License: MIT
Version: 0.1.1
Date: 2026-02-12
-->

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

### Exclude Tools by Name Pattern

```python
tools = build_tools(exclude_name_pattern="^ToxCast")
# Excludes all tools whose name starts with "ToxCast"
```

### Load Drug-Target Interaction Tools

```python
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Target sequence"],
    match_all=True
)
# Returns only tools that use BOTH Drug SMILES AND Target sequence
```

### Combine Multiple Filters

```python
tools = build_tools(
    filter_placeholder="Drug SMILES",
    max_placeholders=1,
    exclude_name_pattern="^ToxCast"
)
# Returns simple Drug SMILES tools, excluding ToxCast tools
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

### 4. By Name Pattern

Exclude tools by regex pattern matching their names.

**Exclude by Prefix:**
```python
tools = build_tools(exclude_name_pattern="^ToxCast")
# Excludes all tools starting with "ToxCast"
# Example: ToxCast_AR_predict, ToxCast_ER_predict, etc.
```

**Exclude by Substring:**
```python
tools = build_tools(exclude_name_pattern="ToxCast")
# Excludes any tool with "ToxCast" anywhere in name
```

**Exclude Multiple Patterns:**
```python
tools = build_tools(exclude_name_pattern="^(ToxCast|Tox21)")
# Excludes tools starting with "ToxCast" OR "Tox21"
```

**Case-Insensitive Exclusion:**
```python
tools = build_tools(exclude_name_pattern="(?i)toxcast")
# Excludes tools with "toxcast" in any case
```

**Complex Pattern Examples:**
```python
# Exclude all toxicity-related tools
tools = build_tools(exclude_name_pattern="(?i)(tox|toxic)")

# Exclude specific tool families
tools = build_tools(exclude_name_pattern="^(ToxCast|Tox21|AMES)")

# Exclude by suffix
tools = build_tools(exclude_name_pattern="_legacy$")
```

### 5. Combining Filters

All filters can be combined for precise tool selection:

```python
# Drug SMILES tools, simple only, no ToxCast
tools = build_tools(
    filter_placeholder="Drug SMILES",
    max_placeholders=1,
    exclude_name_pattern="^ToxCast"
)

# Drug tools, excluding both ToxCast and Tox21
tools = build_tools(
    filter_placeholder="Drug SMILES",
    exclude_name_pattern="^(ToxCast|Tox21)"
)

# Simple tools, excluding experimental ones
tools = build_tools(
    max_placeholders=2,
    exclude_name_pattern="(?i)(experimental|beta|alpha)"
)
```

**Filter Order:**
Filters are applied in this order:
1. `exclude_name_pattern` - Name-based exclusion (first)
2. `filter_placeholder` - Placeholder filtering
3. `max_placeholders` - Complexity filtering

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

**Fuzzy matching (case-insensitive substring):**
```bash
python scripts/analyze_tools.py --placeholder "smiles" --fuzzy
```

#### Show Tools Using Multiple Placeholders

**Match ALL placeholders (default):**
```bash
python scripts/analyze_tools.py --placeholders "Drug SMILES" "Target sequence"
```

**Match ANY placeholder:**
```bash
python scripts/analyze_tools.py --placeholders "Drug SMILES" "Protein sequence" --any
```

#### Exclude Tools by Name Pattern

**Exclude all ToxCast tools:**
```bash
python scripts/analyze_tools.py --exclude "^ToxCast"
```

**Exclude multiple tool families:**
```bash
python scripts/analyze_tools.py --exclude "^(ToxCast|Tox21)"
```

**Case-insensitive exclusion:**
```bash
python scripts/analyze_tools.py --exclude "(?i)experimental"
```

#### Show Simple Tools Only

```bash
python scripts/analyze_tools.py --simple
```

#### Show Complex Tools Only

```bash
python scripts/analyze_tools.py --complex
```

#### Combine Multiple Filters

**Drug SMILES tools, simple only, no ToxCast:**
```bash
python scripts/analyze_tools.py --placeholder "Drug SMILES" --simple --exclude "^ToxCast"
```

**Fuzzy search with exclusion:**
```bash
python scripts/analyze_tools.py --placeholder "smiles" --fuzzy --exclude "^ToxCast"
```

### Output Format Commands

#### JSON Output

```bash
python scripts/analyze_tools.py --json
```

**Export to file:**
```bash
python scripts/analyze_tools.py --placeholder "Drug SMILES" --json > tools.json
```

#### Verbose Output

```bash
python scripts/analyze_tools.py --verbose
```

Shows detailed parameter information for each tool.

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

**Save to file:**
```bash
python scripts/analyze_tools.py --json > tools.json
python scripts/analyze_tools.py --placeholder "Drug SMILES" --json > drug_tools.json
```

#### Verbose Output

```bash
python scripts/analyze_tools.py --simple --verbose
```

---

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
# In server.py or config
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

### Use Case 2: Exclude Regulatory Assays

**Goal**: Exclude ToxCast and Tox21 regulatory testing tools

```python
TOOLS = build_tools(
    filter_placeholder="Drug SMILES",
    exclude_name_pattern="^(ToxCast|Tox21)"
)
```

**Why?**
- ToxCast/Tox21 tools are highly specialized regulatory assays
- May not be relevant for early-stage discovery
- Reduces tool count for cleaner AI tool selection

**Result**: All Drug SMILES tools except regulatory assays

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

### Use Case 4: Production-Ready Tools Only

**Goal**: Exclude experimental or beta tools

```python
TOOLS = build_tools(
    filter_placeholder="Drug SMILES",
    exclude_name_pattern="(?i)(experimental|beta|alpha|dev)"
)
```

**Why?**
- More stable predictions for production
- Avoid tools still under development
- Consistent API across tool versions

### Use Case 5: Custom Tool Curation

**Goal**: Specific set of high-quality tools

```python
# Include Drug SMILES tools
# Exclude: ToxCast, Tox21, and complex multi-input tools
TOOLS = build_tools(
    filter_placeholder="Drug SMILES",
    max_placeholders=2,
    exclude_name_pattern="^(ToxCast|Tox21)"
)
```

**Result**: Curated, focused drug discovery toolset

---

## Programmatic Usage

### Basic Filtering

```python
from txgemma.tool_factory import build_tools

# Filter by placeholder
drug_tools = build_tools(filter_placeholder="Drug SMILES")

# Filter by complexity
simple_tools = build_tools(max_placeholders=1)

# Exclude by pattern
no_toxcast = build_tools(exclude_name_pattern="^ToxCast")

# Combine filters
focused_tools = build_tools(
    filter_placeholder="Drug SMILES",
    max_placeholders=1,
    exclude_name_pattern="^ToxCast"
)
```

### All Filtering Parameters

The `build_tools()` function supports the following parameters:

#### `filter_placeholder` (str)
Filter by a single placeholder name.

```python
# Exact match (default)
tools = build_tools(filter_placeholder="Drug SMILES")

# Fuzzy match (substring, case-insensitive)
tools = build_tools(
    filter_placeholder="smiles",
    exact_match=False
)
```

#### `filter_placeholders` (list[str])
Filter by multiple placeholders.

```python
# Match ALL placeholders (default)
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Target sequence"],
    match_all=True
)

# Match ANY placeholder
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Protein sequence"],
    match_all=False
)
```

#### `match_all` (bool)
When using `filter_placeholders`, determines if tools must have ALL or ANY of the specified placeholders.

- `True` (default): Tool must have ALL specified placeholders
- `False`: Tool can have ANY of the specified placeholders

```python
# Drug-target interaction tools (need both)
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Target sequence"],
    match_all=True  # Must have BOTH
)

# Any drug or protein tool
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Protein sequence"],
    match_all=False  # Can have EITHER
)
```

#### `exact_match` (bool)
Controls placeholder matching behavior.

- `True` (default): Exact string match
- `False`: Fuzzy substring match (case-insensitive)

```python
# Exact match
tools = build_tools(
    filter_placeholder="Drug SMILES",
    exact_match=True
)
# Matches: "Drug SMILES" only

# Fuzzy match
tools = build_tools(
    filter_placeholder="smiles",
    exact_match=False
)
# Matches: "Drug SMILES", "Product SMILES", "Molecule SMILES", etc.
```

#### `exclude_complex` (bool)
Exclude tools with more than 2 placeholders.

```python
# Only simple tools (≤2 parameters)
tools = build_tools(exclude_complex=True)

# Same as:
tools = build_tools(max_placeholders=2)
```

**Note**: `max_placeholders` takes precedence over `exclude_complex` if both are specified.

#### `max_placeholders` (int)
Maximum number of placeholders per tool.

```python
# Single-parameter tools only
tools = build_tools(max_placeholders=1)

# Up to 2 parameters
tools = build_tools(max_placeholders=2)

# Up to 3 parameters
tools = build_tools(max_placeholders=3)
```

#### `exclude_name_pattern` (str)
Regex pattern to exclude tools by name.

```python
# Exclude by prefix
tools = build_tools(exclude_name_pattern="^ToxCast")

# Exclude multiple families
tools = build_tools(exclude_name_pattern="^(ToxCast|Tox21)")

# Case-insensitive
tools = build_tools(exclude_name_pattern="(?i)experimental")

# Exclude by suffix
tools = build_tools(exclude_name_pattern="_legacy$")
```

### Parameter Combinations

You can combine any parameters for precise filtering:

```python
# Drug SMILES tools, simple only, no ToxCast, fuzzy match
tools = build_tools(
    filter_placeholder="smiles",
    exact_match=False,
    max_placeholders=1,
    exclude_name_pattern="^ToxCast"
)

# Drug-target tools, excluding regulatory assays
tools = build_tools(
    filter_placeholders=["Drug SMILES", "Target sequence"],
    match_all=True,
    exclude_name_pattern="^(ToxCast|Tox21|AMES)"
)

# Any sequence tool, simple only
tools = build_tools(
    filter_placeholder="sequence",
    exact_match=False,
    exclude_complex=True
)
```

### Filter Execution Order

Filters are applied in this order:

1. **Placeholder filtering** (`filter_placeholder` or `filter_placeholders`)
2. **Name exclusion** (`exclude_name_pattern`)
3. **Complexity filtering** (`max_placeholders` or `exclude_complex`)

This order ensures efficient filtering and predictable results.

### Advanced Pattern Matching

```python
# Exclude multiple prefixes
tools = build_tools(
    exclude_name_pattern="^(ToxCast|Tox21|AMES)"
)

# Case-insensitive exclusion
tools = build_tools(
    exclude_name_pattern="(?i)legacy"
)

# Exclude by suffix
tools = build_tools(
    exclude_name_pattern="_v1$"
)

# Complex regex patterns
tools = build_tools(
    exclude_name_pattern="^(ToxCast|Tox21).*_(ER|AR)_"
)
```

### Error Handling

```python
try:
    tools = build_tools(exclude_name_pattern="[invalid(")
except ValueError as e:
    print(f"Invalid regex pattern: {e}")
    # Fall back to no exclusion
    tools = build_tools()
```

### Dynamic Tool Loading

```python
from txgemma.tool_factory import build_tools

class DynamicToolServer:
    def __init__(self):
        self.tool_cache = {}
    
    def get_tools(self, use_case: str):
        if use_case not in self.tool_cache:
            if use_case == "drug_discovery":
                self.tool_cache[use_case] = build_tools(
                    filter_placeholder="Drug SMILES",
                    exclude_name_pattern="^(ToxCast|Tox21)"
                )
            elif use_case == "drug_discovery_full":
                self.tool_cache[use_case] = build_tools(
                    filter_placeholder="Drug SMILES"
                )
            elif use_case == "simple_screening":
                self.tool_cache[use_case] = build_tools(
                    max_placeholders=1,
                    exclude_name_pattern="(?i)(complex|advanced)"
                )
        return self.tool_cache[use_case]
```

### User-Specific Tool Subsets

```python
def get_tools_for_user(user_role: str, include_experimental: bool = False):
    """Return tools appropriate for user's role."""
    
    exclude_pattern = None
    if not include_experimental:
        exclude_pattern = "(?i)(experimental|beta|alpha)"
    
    if user_role == "medicinal_chemist":
        return build_tools(
            filter_placeholder="Drug SMILES",
            exclude_name_pattern=exclude_pattern or "^ToxCast"
        )
    elif user_role == "toxicologist":
        # Toxicologists want ToxCast tools
        return build_tools(
            filter_placeholder="Drug SMILES",
            exclude_name_pattern=exclude_pattern
        )
    elif user_role == "student":
        # Students get simple tools only
        pattern = exclude_pattern or "^ToxCast"
        return build_tools(
            max_placeholders=1,
            exclude_name_pattern=pattern
        )
    else:
        # Admins get all tools
        return build_tools(exclude_name_pattern=exclude_pattern)
```

### Conditional Filtering

```python
import os

# Load different tools based on environment
ENV = os.getenv("ENV", "development")
EXCLUDE_EXPERIMENTAL = os.getenv("EXCLUDE_EXPERIMENTAL", "false").lower() == "true"

if ENV == "production":
    # Production: exclude experimental and ToxCast
    TOOLS = build_tools(
        filter_placeholder="Drug SMILES",
        max_placeholders=2,
        exclude_name_pattern="^(ToxCast|Tox21).*experimental"
    )
elif ENV == "staging":
    # Staging: exclude only experimental
    TOOLS = build_tools(
        filter_placeholder="Drug SMILES",
        exclude_name_pattern="(?i)(experimental|beta)" if EXCLUDE_EXPERIMENTAL else None
    )
else:
    # Development: all tools
    TOOLS = build_tools()
```

### Logging and Debugging

```python
import logging

logging.basicConfig(level=logging.INFO)

# The exclude_name_pattern logs which tools are excluded
tools = build_tools(
    filter_placeholder="Drug SMILES",
    exclude_name_pattern="^ToxCast"
)

# Output:
# INFO:txgemma.tool_factory:Excluding tools matching pattern: ^ToxCast
# INFO:txgemma.tool_factory:Excluded 15 tool(s) matching pattern '^ToxCast'
# INFO:txgemma.tool_factory:Built 25 tools (filter_placeholder=Drug SMILES, max_placeholders=None, exclude_pattern=^ToxCast)
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

### 2. Exclusion is Fast

The `exclude_name_pattern` filter is applied first, before any tool building:

```python
# ✅ Fast - excludes before building tools
tools = build_tools(
    filter_placeholder="Drug SMILES",
    exclude_name_pattern="^ToxCast"  # Applied first
)
```

### 3. Use `get_tool_names()` for Lightweight Checks

```python
from txgemma.tool_factory import get_tool_names

# Just get names (fast, no tool building)
names = get_tool_names(filter_placeholder="Drug SMILES")
print(f"Found {len(names)} matching tools")

# Only build if needed
if len(names) > 0:
    tools = build_tools(filter_placeholder="Drug SMILES")
```

### 4. Cache Tool Subsets

```python
# Build once, use many times
DRUG_TOOLS = build_tools(
    filter_placeholder="Drug SMILES",
    exclude_name_pattern="^ToxCast"
)

DRUG_TOOLS_WITH_TOXCAST = build_tools(
    filter_placeholder="Drug SMILES"
)

# Use cached subsets
def get_tools_for_use_case(use_case: str, include_toxcast: bool = False):
    if use_case == "drug_discovery":
        return DRUG_TOOLS_WITH_TOXCAST if include_toxcast else DRUG_TOOLS
    # ...
```

---

## Configuration Examples

### Configuration 1: Focused Drug Discovery Server

```python
# server.py
TOOLS = build_tools(
    filter_placeholder="Drug SMILES",
    max_placeholders=2,
    exclude_name_pattern="^(ToxCast|Tox21)"
)
# Result: Simple drug prediction tools, excluding regulatory assays
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
    max_placeholders=1,
    exclude_name_pattern="(?i)(experimental|beta|legacy)"
)
# Result: Fast, simple, stable predictions for high-throughput screening
```

### Configuration 4: Regulatory Toxicology Server

```python
# server.py
# Include ONLY ToxCast and Tox21 tools
import re
from txgemma.tool_factory import build_tools

all_tools = build_tools(filter_placeholder="Drug SMILES")
tox_tools = [
    t for t in all_tools 
    if re.match(r"^(ToxCast|Tox21)", t.name)
]
# Result: Only regulatory toxicology assays
```

### Configuration 5: Educational/Training Server

```python
# server.py
TOOLS = build_tools(
    max_placeholders=1,
    exclude_name_pattern="^(ToxCast|Tox21|Advanced)"
)
# Result: Simple, easy-to-understand tools for learning
```

---

## Common Exclusion Patterns

Here are common regex patterns for excluding tools:

```python
# Exclude by prefix
exclude_name_pattern="^ToxCast"           # All ToxCast tools
exclude_name_pattern="^Tox21"             # All Tox21 tools
exclude_name_pattern="^(ToxCast|Tox21)"   # Both families

# Exclude by substring
exclude_name_pattern="legacy"             # Any tool with "legacy"
exclude_name_pattern="(?i)beta"           # Case-insensitive "beta"

# Exclude by suffix
exclude_name_pattern="_v1$"               # Tools ending with "_v1"
exclude_name_pattern="_deprecated$"       # Deprecated tools

# Complex patterns
exclude_name_pattern="^(ToxCast|Tox21).*_(ER|AR)"  # Specific assays
exclude_name_pattern="(?i)(experimental|alpha|beta|dev)"  # Development versions
exclude_name_pattern="^(ToxCast|Tox21|AMES|Bacterial)"  # Multiple prefixes
```

---

## Summary

### Key Benefits of Filtering

- 🚀 Faster server startup
- 💾 Reduced memory usage
- 🎯 Focused tool selection
- 👥 Role-based access
- 📊 Better organization
- 🔒 Exclude unwanted tool categories

### Filtering Dimensions

- **By placeholder** - Filter by input type
- **By complexity** - Filter by parameter count
- **By name pattern** - Exclude by regex (NEW)
- **Combinations** - Use all three together

### Best Practices

1. **Filter at build time, not runtime** - More efficient
2. **Use exclusion patterns for unwanted categories** - Clean tool sets
3. **Test regex patterns before deployment** - Avoid surprises
4. **Use CLI for exploration** - `analyze_tools.py` is your friend
5. **Cache tool subsets for reuse** - Performance optimization
6. **Match filtering to use case** - Purpose-driven design
7. **Document your filtering choices** - Team clarity
8. **Log exclusions in production** - Visibility into what's filtered

### Parameter Reference

```python
build_tools(
    filter_placeholder=None,      # str: Filter by single placeholder
    max_placeholders=None,        # int: Maximum placeholder count
    exclude_name_pattern=None,    # str: Regex pattern to exclude tools
)
```

**Filter Order:**
1. `exclude_name_pattern` (first)
2. `filter_placeholder`
3. `max_placeholders` (last)

### CLI Quick Reference

```bash
# Exploration
analyze_tools.py --list-placeholders
analyze_tools.py --source
analyze_tools.py --template "tool_name"

# Filtering
analyze_tools.py --placeholder "Drug SMILES"
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