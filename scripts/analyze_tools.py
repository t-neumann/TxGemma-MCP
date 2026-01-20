#!/usr/bin/env python3
"""
Analyze available TxGemma tools and suggest subsets.

Usage:
    python scripts/analyze_tools.py
    python scripts/analyze_tools.py --placeholder "Drug SMILES"
    python scripts/analyze_tools.py --simple
"""

import argparse
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TxGemma MCP tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Use fuzzy matching for placeholder names"
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
        help="List all available placeholders"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    loader = get_loader()
    
    # Handle different modes
    if args.list_placeholders:
        print_section("Available Placeholders")
        
        stats = loader.placeholder_stats()
        
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            # Sort by usage count
            for placeholder, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {placeholder:<40} (used in {count} tool{'s' if count != 1 else ''})")
        
        return
    
    # Build tools with filters
    if args.placeholder:
        tools = build_tools(
            filter_placeholder=args.placeholder,
            exact_match=not args.fuzzy
        )
        filter_desc = f"using '{args.placeholder}'"
    elif args.placeholders:
        tools = build_tools(
            filter_placeholders=args.placeholders,
            match_all=not args.any,
        )
        match_type = "ANY" if args.any else "ALL"
        filter_desc = f"using {match_type} of {args.placeholders}"
    elif args.simple:
        tools = build_tools(max_placeholders=2)
        filter_desc = "simple (≤2 placeholders)"
    elif args.complex:
        tools = build_tools(max_placeholders=None)
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
            })
        print(json.dumps(output, indent=2))
    else:
        print_section(f"Tools ({filter_desc})")
        print(f"Found {len(tools)} tool{'s' if len(tools) != 1 else ''}\n")
        
        for tool in tools:
            params = tool.inputSchema["required"]
            print(f"  📦 {tool.name}")
            print(f"     {tool.description}")
            print(f"     Parameters: {', '.join(params)}")
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
            print(f"    {count} parameter{'s' if count != 1 else ''}: {tool_count} tool{'s' if tool_count != 1 else ''}")
        
        print("\n  Most common placeholders:")
        for placeholder, count in stats['most_common_placeholders'][:5]:
            print(f"    {placeholder:<35} (used in {count} tools)")
        
        # Show suggested subsets
        print_section("Suggested Tool Subsets")
        subsets = suggest_tool_subsets()
        
        for use_case, tool_names in subsets.items():
            print(f"  {use_case.replace('_', ' ').title()}:")
            print(f"    {len(tool_names)} tools")
            if tool_names:
                print(f"    Examples: {', '.join(tool_names[:3])}")
                if len(tool_names) > 3:
                    print(f"              ... and {len(tool_names) - 3} more")
            print()


if __name__ == "__main__":
    main()