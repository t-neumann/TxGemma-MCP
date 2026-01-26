"""
Load, validate, and introspect TxGemma / TDC prompt templates.
"""

from __future__ import annotations

import builtins
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# -------------------------
# Constants & Regex
# -------------------------

DEFAULT_HF_REPO = "google/txgemma-2b-predict"
DEFAULT_FILENAME = "tdc_prompts.json"

# Matches placeholders like:
# {Drug SMILES}
# {Epitope amino acid sequence}
PLACEHOLDER_REGEX = re.compile(r"\{([^{}]+)\}")

# -------------------------
# PromptTemplate
# -------------------------


class PromptTemplate:
    """
    Represents a single TxGemma / TDC prompt template.
    """

    def __init__(
        self,
        name: str,
        template: str,
        metadata: dict | None = None,
    ):
        self.name = name
        self.template = template
        self.metadata = metadata or {}

        self.placeholders: list[str] = self._extract_placeholders()

    # ---- Introspection ----

    def _extract_placeholders(self) -> list[str]:
        """
        Extract placeholder variables from template.

        Preserves order and uniqueness.
        """
        matches = PLACEHOLDER_REGEX.findall(self.template)
        return list(dict.fromkeys(matches))

    @property
    def required_inputs(self) -> set[str]:
        """Set of required input variables."""
        return set(self.placeholders)

    def has_placeholder(self, placeholder: str) -> bool:
        """Check if this template requires a specific placeholder."""
        return placeholder in self.required_inputs

    def placeholder_count(self) -> int:
        """Number of unique placeholders in this template."""
        return len(self.placeholders)

    # ---- Rendering ----

    def format(self, **kwargs) -> str:
        """
        Format template with provided values.

        Raises:
            ValueError if required placeholders are missing.
        """
        missing = self.required_inputs - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required placeholders for '{self.name}': {sorted(missing)}")

        return self.template.format(**kwargs)

    # ---- Human-facing descriptions ----

    def get_description(self) -> str:
        """
        Get a concise human-readable description for MCP / tooling.
        """
        if "description" in self.metadata:
            return self.metadata["description"]

        for line in self.template.splitlines():
            if line.strip().startswith("Context:"):
                return line.replace("Context:", "").strip()

        return f"TxGemma prediction task: {self.name}"

    def to_metadata(self) -> dict:
        """
        Export structured metadata (useful for MCP tool schemas).
        """
        return {
            "name": self.name,
            "description": self.get_description(),
            "required_inputs": sorted(self.required_inputs),
            "placeholder_count": self.placeholder_count(),
        }

    def __str__(self) -> str:
        inputs = ", ".join(sorted(self.required_inputs)) or "none"
        desc = self.get_description()

        # Keep description short for logs
        if len(desc) > 80:
            desc = desc[:77] + "..."

        return (
            f"PromptTemplate("
            f"name='{self.name}', "
            f"inputs=[{inputs}], "
            f"placeholders={self.placeholder_count()}, "
            f"description='{desc}'"
            f")"
        )

    def __repr__(self) -> str:
        return (
            f"PromptTemplate("
            f"name={self.name!r}, "
            f"placeholders={self.placeholders!r}, "
            f"metadata_keys={list(self.metadata.keys())!r}, "
            f"template_len={len(self.template)}"
            f")"
        )


# -------------------------
# PromptLoader
# -------------------------


class PromptLoader:
    """
    Load and manage TxGemma / TDC prompt templates.
    """

    def __init__(
        self,
        *,
        hf_repo: str = DEFAULT_HF_REPO,
        filename: str = DEFAULT_FILENAME,
        local_override: Path | None = None,
    ):
        self.hf_repo = hf_repo
        self.filename = filename
        self.local_override = local_override

        self._templates: dict[str, PromptTemplate] = {}
        self._placeholder_index: dict[str, set[str]] = defaultdict(set)
        self._loaded = False
        self._source = None  # Track where prompts were loaded from

    # ---- Loading ----

    def _load_json(self) -> dict:
        """
        Load prompts JSON from local file or Hugging Face.

        Raises:
            FileNotFoundError: If local override doesn't exist
            RuntimeError: If HuggingFace download fails
            ValueError: If JSON is invalid
        """
        if self.local_override:
            if not self.local_override.exists():
                raise FileNotFoundError(f"Local override not found: {self.local_override}")
            path = self.local_override
            self._source = f"local file: {path}"
        else:
            try:
                path = hf_hub_download(
                    repo_id=self.hf_repo,
                    filename=self.filename,
                )
                self._source = f"HuggingFace: {self.hf_repo}/{self.filename}"
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download prompts from HuggingFace "
                    f"({self.hf_repo}/{self.filename}): {e}"
                ) from e

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts file ({path}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to read prompts file ({path}): {e}") from e

        logger.info(f"Loaded {len(data)} prompt definitions from {self._source}")
        return data

    def _build_placeholder_index(self):
        """
        Build reverse index: placeholder -> set of template names that use it.
        """
        self._placeholder_index.clear()
        for name, template in self._templates.items():
            for placeholder in template.placeholders:
                self._placeholder_index[placeholder].add(name)

    def load(self):
        """
        Load prompts from source.

        Raises:
            ValueError: If prompt data is malformed
            RuntimeError: If loading fails
        """
        if self._loaded:
            return

        data = self._load_json()

        # Validate top-level structure
        if not isinstance(data, dict):
            raise ValueError(f"Prompts JSON must be a dictionary, got {type(data).__name__}")

        # Parse each prompt
        for name, content in data.items():
            try:
                if isinstance(content, str):
                    # Simple format: template string only
                    self._templates[name] = PromptTemplate(name, content)

                elif isinstance(content, dict):
                    # Rich format: template + metadata
                    if "template" not in content:
                        raise ValueError(f"Prompt '{name}' missing 'template' field")

                    self._templates[name] = PromptTemplate(
                        name=name,
                        template=content["template"],
                        metadata=content.get("metadata", {}),
                    )
                else:
                    raise ValueError(
                        f"Invalid prompt format for '{name}': "
                        f"expected str or dict, got {type(content).__name__}"
                    )
            except Exception as e:
                logger.error(f"Failed to load prompt '{name}': {e}")
                raise

        # Build reverse index for efficient lookup
        self._build_placeholder_index()

        self._loaded = True
        logger.info(
            f"Successfully loaded {len(self._templates)} templates with "
            f"{len(self._placeholder_index)} unique placeholders"
        )

    def reload(self):
        """
        Reload prompts from source.

        Useful for development when prompts are being updated.
        """
        logger.info("Reloading prompts...")
        self._loaded = False
        self._templates.clear()
        self._placeholder_index.clear()
        self._source = None
        self.load()

    # ---- Accessors ----

    def get(self, name: str) -> PromptTemplate:
        """
        Get a specific template by name.

        Raises:
            KeyError: If template doesn't exist
        """
        self.load()
        if name not in self._templates:
            available = ", ".join(sorted(self._templates.keys())[:5])
            raise KeyError(f"Prompt '{name}' not found. Available prompts include: {available}...")
        return self._templates[name]

    def has_template(self, name: str) -> bool:
        """Check if a template exists without raising an error."""
        self.load()
        return name in self._templates

    def all(self) -> dict[str, PromptTemplate]:
        """Get all templates."""
        self.load()
        return dict(self._templates)

    def list(self) -> builtins.list[str]:
        """List all template names."""
        self.load()
        return list(self._templates.keys())

    def __len__(self) -> int:
        """Return number of loaded templates."""
        self.load()
        return len(self._templates)

    def __contains__(self, name: str) -> bool:
        """Check if template exists (enables 'in' operator)."""
        return self.has_template(name)

    # ---- Placeholder Discovery ----

    def all_placeholders(self) -> set[str]:
        """
        Get set of ALL placeholders used across all templates.

        Useful for understanding the input vocabulary.
        """
        self.load()
        return set(self._placeholder_index.keys())

    def placeholder_usage(self, placeholder: str) -> set[str]:
        """
        Get set of template names that use a specific placeholder.

        Args:
            placeholder: Placeholder name (e.g., "Drug SMILES")

        Returns:
            Set of template names that require this placeholder

        Example:
            >>> loader.placeholder_usage("Drug SMILES")
            {'predict_toxicity', 'predict_bbb_permeability', ...}
        """
        self.load()
        return self._placeholder_index.get(placeholder, set()).copy()

    def placeholder_stats(self) -> dict[str, int]:
        """
        Get usage statistics for all placeholders.

        Returns:
            Dict mapping placeholder -> count of templates using it

        Example:
            >>> loader.placeholder_stats()
            {'Drug SMILES': 15, 'Target sequence': 3, ...}
        """
        self.load()
        return {
            placeholder: len(template_names)
            for placeholder, template_names in self._placeholder_index.items()
        }

    def most_common_placeholders(self, top_n: int = 10) -> builtins.list[tuple[str, int]]:
        """
        Get most commonly used placeholders across templates.

        Args:
            top_n: Number of top placeholders to return

        Returns:
            List of (placeholder, usage_count) tuples, sorted by count descending
        """
        stats = self.placeholder_stats()
        return sorted(stats.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ---- Filtering by Placeholder ----

    def filter_by_placeholder(
        self, placeholder: str, *, exact: bool = True
    ) -> dict[str, PromptTemplate]:
        """
        Return templates that use a specific placeholder.

        Args:
            placeholder: Placeholder to filter by (e.g., "Drug SMILES")
            exact: If True, match exactly. If False, match case-insensitive substring

        Returns:
            Dict of template_name -> PromptTemplate

        Example:
            >>> loader.filter_by_placeholder("Drug SMILES")
            {'predict_toxicity': <PromptTemplate>, ...}

            >>> loader.filter_by_placeholder("smiles", exact=False)
            # Returns all templates with any placeholder containing "smiles"
        """
        self.load()

        if exact:
            template_names = self._placeholder_index.get(placeholder, set())
            return {name: self._templates[name] for name in template_names}
        else:
            # Fuzzy match - case insensitive substring search
            placeholder_lower = placeholder.lower()
            result = {}
            for name, template in self._templates.items():
                for tmpl_placeholder in template.placeholders:
                    if placeholder_lower in tmpl_placeholder.lower():
                        result[name] = template
                        break
            return result

    def filter_by_placeholders(
        self, placeholders: builtins.list[str], *, match_all: bool = True
    ) -> dict[str, PromptTemplate]:
        """
        Filter templates by multiple placeholders.

        Args:
            placeholders: List of placeholder names
            match_all: If True, template must use ALL placeholders.
                      If False, template must use ANY of the placeholders.

        Returns:
            Dict of matching templates

        Example:
            >>> loader.filter_by_placeholders(
            ...     ["Drug SMILES", "Target sequence"],
            ...     match_all=True
            ... )
            # Returns only templates that use BOTH placeholders
        """
        self.load()

        if match_all:
            # Template must have ALL placeholders
            result = {}
            for name, template in self._templates.items():
                if all(ph in template.required_inputs for ph in placeholders):
                    result[name] = template
            return result
        else:
            # Template must have ANY placeholder
            result = {}
            for name, template in self._templates.items():
                if any(ph in template.required_inputs for ph in placeholders):
                    result[name] = template
            return result

    # ---- Convenience Filters (for common use cases) ----

    def smiles_prompts(self) -> dict[str, PromptTemplate]:
        """Convenience: Get all templates that use Drug SMILES."""
        return self.filter_by_placeholder("Drug SMILES")

    def sequence_prompts(self) -> dict[str, PromptTemplate]:
        """Convenience: Get all templates that use protein/target sequences."""
        # Fuzzy match for any sequence-related placeholder
        return self.filter_by_placeholder("sequence", exact=False)

    def simple_prompts(self, max_placeholders: int = 1) -> dict[str, PromptTemplate]:
        """
        Get templates with few placeholders (simpler to use).

        Args:
            max_placeholders: Maximum number of placeholders
        """
        self.load()
        return {
            name: tmpl
            for name, tmpl in self._templates.items()
            if tmpl.placeholder_count() <= max_placeholders
        }

    def complex_prompts(self, min_placeholders: int = 3) -> dict[str, PromptTemplate]:
        """
        Get templates with many placeholders (more complex inputs).

        Args:
            min_placeholders: Minimum number of placeholders
        """
        self.load()
        return {
            name: tmpl
            for name, tmpl in self._templates.items()
            if tmpl.placeholder_count() >= min_placeholders
        }

    @property
    def source(self) -> str | None:
        """Get the source where prompts were loaded from."""
        return self._source


# -------------------------
# Global Loader (optional)
# -------------------------

_default_loader: PromptLoader | None = None


def get_loader() -> PromptLoader:
    """
    Get the global default loader instance.

    Configuration loaded from config.yaml.
    HuggingFace repo automatically derived from predict.model.
    """
    global _default_loader
    if _default_loader is None:
        try:
            # Try to load from config
            from txgemma.config import get_config

            config = get_config()

            prompts_config = config.tools.prompts

            # Check if using local override
            if prompts_config.local_override:
                local_path = Path(prompts_config.local_override)
                _default_loader = PromptLoader(local_override=local_path)
                logger.info(f"Prompts loaded from local file: {local_path}")
            else:
                # Use HuggingFace - derive repo from predict model
                hf_repo = config.predict.model
                _default_loader = PromptLoader(hf_repo=hf_repo, filename=prompts_config.filename)
                logger.info(f"Prompts loaded from HuggingFace: {hf_repo}/{prompts_config.filename}")
        except Exception as e:
            # Fallback to defaults if config not available
            logger.warning(f"Could not load prompts config, using defaults: {e}")
            _default_loader = PromptLoader()
            logger.info(f"Prompts loaded from default: {DEFAULT_HF_REPO}/{DEFAULT_FILENAME}")

    return _default_loader
