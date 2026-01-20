#!/usr/bin/env python3
"""
Analyze available TxGemma tools and explore prompts.

Usage:
    # List all placeholders
    python scripts/analyze_tools.py --list-placeholders
    
    # Show all tools
    python scripts/analyze_tools.py
    
    # Show tools using specific placeholder
    python scripts/analyze_tools.py --placeholder "Drug SMILES"
    
    # Fuzzy search for placeholders
    python scripts/analyze_tools.py --placeholder "smiles" --fuzzy
    
    # Show simple tools only
    python scripts/analyze_tools.py --simple
    
    # Show tools using multiple placeholders (ALL)
    python scripts/analyze_tools.py --placeholders "Drug SMILES" "Target sequence"
    
    # Show tools using multiple placeholders (ANY)
    python scripts/analyze_tools.py --placeholders "Drug SMILES" "Protein sequence" --any
    
    # JSON output
    python scripts/analyze_tools.py --json
    
    # Show specific template details
    python scripts/analyze_tools.py --template "predict_toxicity"
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from txgemma.tool_factory import (
    analyze_tools,
    suggest_tool_subsets,
    get_tool_names,
    build_tools,
)
from txgemma.prompts import get_loader


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_template_details(template_name: str):
    """Print detailed information about a specific template."""
    loader = get_loader()
    
    try:
        template = loader.get(template_name)
    except KeyError as e:
        print(f"Error: {e}")
        return
    
    print_section(f"Template: {template.name}")
    
    print(f"Description:")
    print(f"  {template.get_description()}\n")
    
    print(f"Placeholders ({template.placeholder_count()}):")
    for ph in template.placeholders:
        print(f"  - {ph}")
    
    if template.metadata:
        print(f"\nMetadata:")
        for key, value in template.metadata.items():
            print(f"  {key}: {value}")
    
    print(f"\nTemplate Preview:")
    lines = template.template.split('\n')
    for i, line in enumerate(lines[:10], 1):
        print(f"  {i:2}. {line}")
    if len(lines) > 10:
        print(f"  ... ({len(lines) - 10} more lines)")
    
    print(f"\nUsed by:")
    # Show which tool factory would create from this
    tools = build_tools()
    matching = [t for t in tools if t.name == template_name]
    if matching:
        tool = matching[0]
        print(f"  Tool name: {tool.name}")
        print(f"  Description: {tool.description}")
        print(f"  Parameters: {', '.join(tool.inputSchema['required'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TxGemma MCP tools and prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all placeholders with usage counts
  %(prog)s --list-placeholders
  
  # Show all Drug SMILES tools
  %(prog)s --placeholder "Drug SMILES"
  
  # Fuzzy search for sequence-related tools
  %(prog)s --placeholder "sequence" --fuzzy
  
  # Show simple tools (≤2 parameters)
  %(prog)s --simple
  
  # Show drug-target interaction tools
  %(prog)s --placeholders "Drug SMILES" "Target sequence"
  
  # Show tools with ANY of multiple placeholders
  %(prog)s --placeholders "Drug SMILES" "Protein sequence" --any
  
  # Get template details
  %(prog)s --template "predict_toxicity"
  
  # Export to JSON
  %(prog)s --json > tools.json
        """
    )
    
    parser.add_argument(
        "--placeholder",
        type=str,
        help="Show tools using this placeholder (e.g., 'Drug SMILES')"
    )
    
    parser.add_argument(
        "--placeholders",
        type=str,
        nargs="+",
        help="Show tools using these placeholders"
    )
    
    parser.add_argument(
        "--any",
        action="store_true",
        help="Match ANY placeholder (default: match ALL)"
    )
    
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Use fuzzy matching for placeholder names (case-insensitive substring)"
    )
    
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Show only simple tools (≤2 placeholders)"
    )
    
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Show only complex tools (≥3 placeholders)"
    )
    
    parser.add_argument(
        "--list-placeholders",
        action="store_true",
        help="List all available placeholders with usage counts"
    )
    
    parser.add_argument(
        "--template",
        type=str,
        help="Show details for a specific template"
    )
    
    parser.add_argument(
        "--source",
        action="store_true",
        help="Show where prompts were loaded from"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Set up logging if verbose
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    loader = get_loader()
    
    # Show prompt source
    if args.source:
        print_section("Prompt Source")
        print(f"  Loaded from: {loader.source or 'Not loaded yet'}")
        print(f"  Total templates: {len(loader)}")
        return
    
    # Show template details
    if args.template:
        print_template_details(args.template)
        return
    
    # List placeholders
    if args.list_placeholders:
        print_section("Available Placeholders")
        
        stats = loader.placeholder_stats()
        
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            total_placeholders = len(stats)
            print(f"Found {total_placeholders} unique placeholder{'s' if total_placeholders != 1 else ''}\n")
            
            # Sort by usage count (descending)
            for placeholder, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                tools_using = loader.placeholder_usage(placeholder)
                example_tools = sorted(tools_using)[:3]
                
                print(f"  📌 {placeholder:<40} ({count} tool{'s' if count != 1 else ''})")
                if args.verbose:
                    print(f"     Used in: {', '.join(example_tools)}")
                    if len(tools_using) > 3:
                        print(f"              ... and {len(tools_using) - 3} more")
                print()
        
        return
    
    # Build tools with filters
    if args.placeholder:
        tools = build_tools(
            filter_placeholder=args.placeholder,
            exact_match=not args.fuzzy
        )
        match_type = "fuzzy" if args.fuzzy else "exact"
        filter_desc = f"using '{args.placeholder}' ({match_type} match)"
        
    elif args.placeholders:
        tools = build_tools(
            filter_placeholders=args.placeholders,
            match_all=not args.any,
        )
        match_type = "ANY" if args.any else "ALL"
        filter_desc = f"using {match_type} of: {', '.join(args.placeholders)}"
        
    elif args.simple:
        tools = build_tools(max_placeholders=2)
        filter_desc = "simple (≤2 placeholders)"
        
    elif args.complex:
        tools = build_tools()
        tools = [t for t in tools if len(t.inputSchema["required"]) >= 3]
        filter_desc = "complex (≥3 placeholders)"
        
    else:
        tools = build_tools()
        filter_desc = "all"
    
    # Output results
    if args.json:
        output = []
        for tool in tools:
            output.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema["required"],
                "parameter_count": len(tool.inputSchema["required"]),
                "properties": tool.inputSchema.get("properties", {}),
            })
        print(json.dumps(output, indent=2))
        
    else:
        print_section(f"Tools ({filter_desc})")
        print(f"Found {len(tools)} tool{'s' if len(tools) != 1 else ''}\n")
        
        for tool in sorted(tools, key=lambda t: (len(t.inputSchema["required"]), t.name)):
            params = tool.inputSchema["required"]
            param_count = len(params)
            
            print(f"  📦 {tool.name}")
            print(f"     {tool.description}")
            print(f"     Parameters ({param_count}): {', '.join(params)}")
            
            if args.verbose and param_count > 0:
                print(f"     Details:")
                for param in params:
                    prop = tool.inputSchema.get("properties", {}).get(param, {})
                    param_type = prop.get("type", "unknown")
                    param_desc = prop.get("description", "No description")
                    print(f"       - {param} ({param_type}): {param_desc}")
            
            print()
        
        # Show statistics
        print_section("Tool Statistics")
        stats = analyze_tools()
        
        print(f"  Total tools:              {stats['total_tools']}")
        print(f"  Unique placeholders:      {stats['total_placeholders']}")
        print(f"  Simple tools (≤2 params): {stats['simple_tools']}")
        print(f"  Complex tools (>2 params): {stats['complex_tools']}")
        
        print("\n  Tools by complexity:")
        for count in sorted(stats['tools_by_complexity'].keys()):
            tool_count = stats['tools_by_complexity'][count]
            print(f"    {count} parameter{'s' if count != 1 else ''}:  {tool_count:2} tool{'s' if tool_count != 1 else ''}")
        
        print("\n  Most common placeholders:")
        for placeholder, count in stats['most_common_placeholders'][:5]:
            print(f"    {placeholder:<40} ({count:2} tools)")
        
        # Show suggested subsets
        print_section("Suggested Tool Subsets")
        subsets = suggest_tool_subsets()
        
        for use_case, tool_names in sorted(subsets.items()):
            print(f"  {use_case.replace('_', ' ').title()}:")
            print(f"    {len(tool_names)} tools")
            if tool_names and args.verbose:
                for tool_name in sorted(tool_names):
                    print(f"      - {tool_name}")
            elif tool_names:
                examples = sorted(tool_names)[:3]
                print(f"    Examples: {', '.join(examples)}")
                if len(tool_names) > 3:
                    print(f"              ... and {len(tool_names) - 3} more")
            print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)