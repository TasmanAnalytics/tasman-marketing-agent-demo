"""
Semantic layer parser - loads and compiles SQL from semantic.yml.
"""

import yaml
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path


class SemanticLayer:
    """
    Semantic layer that loads metric definitions and compiles safe SQL queries.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize semantic layer from YAML file.

        Args:
            yaml_path: Path to semantic.yml file
        """
        self.yaml_path = Path(yaml_path)
        self.config = self._load_yaml()
        self.spec_hash = self._compute_hash()

    def _load_yaml(self) -> Dict[str, Any]:
        """Load and parse the semantic YAML file."""
        try:
            with open(self.yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load semantic layer from {self.yaml_path}: {e}")

    def _compute_hash(self) -> str:
        """Compute hash of semantic spec for versioning."""
        content = yaml.dump(self.config, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def get_defaults(self) -> Dict[str, Any]:
        """Get default query parameters."""
        return self.config.get('defaults', {})

    def get_dimensions(self) -> Dict[str, Any]:
        """Get available dimensions."""
        return self.config.get('dimensions', {})

    def get_dimension_names(self) -> List[str]:
        """Get list of dimension names."""
        return list(self.get_dimensions().keys())

    def get_entities(self) -> Dict[str, Any]:
        """Get entity definitions."""
        return self.config.get('entities', {})

    def get_base_queries(self) -> Dict[str, Any]:
        """Get base query definitions."""
        return self.config.get('base_queries', {})

    def get_derived_metrics(self) -> Dict[str, Any]:
        """Get derived metric definitions."""
        return self.config.get('derived_metrics', {})

    def get_metric_definitions(self) -> Dict[str, str]:
        """Get canonical metric formulas."""
        return self.config.get('metric_definitions', {})

    def get_join_rules(self) -> Dict[str, Any]:
        """Get join path rules."""
        return self.config.get('join_rules', {})

    def list_available_metrics(self) -> List[str]:
        """List all available metric names."""
        base = list(self.get_base_queries().keys())
        derived = list(self.get_derived_metrics().keys())
        return sorted(base + derived)

    def compile_query(
        self,
        metric: str,
        window_days: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compile a SQL query for a given metric with parameters.

        Args:
            metric: Metric name (from base_queries or derived_metrics)
            window_days: Time window in days (uses default if None)
            limit: Row limit (uses default if None)

        Returns:
            Dictionary with compiled SQL, description, and metadata

        Raises:
            ValueError: If metric is not defined or compilation fails
        """
        # Get defaults
        defaults = self.get_defaults()
        window_days = window_days or defaults.get('window_days', 90)
        limit = limit or defaults.get('limit', 1000)

        # Check base queries first
        base_queries = self.get_base_queries()
        if metric in base_queries:
            query_def = base_queries[metric]
            sql_template = query_def['sql']
            description = query_def['description']
            outputs = query_def.get('outputs', [])
        else:
            # Check derived metrics
            derived_metrics = self.get_derived_metrics()
            if metric in derived_metrics:
                query_def = derived_metrics[metric]
                sql_template = query_def['sql']
                description = query_def['description']
                outputs = query_def.get('outputs', [])
            else:
                available = self.list_available_metrics()
                raise ValueError(
                    f"Unknown metric '{metric}'. Available metrics: {', '.join(available)}"
                )

        # Compile SQL with parameter substitution
        compiled_sql = sql_template.replace('{{window_days}}', str(window_days))
        compiled_sql = compiled_sql.replace('{{limit}}', str(limit))

        # Generate query ID
        query_id = f"{metric}_{window_days}d_{hashlib.sha256(compiled_sql.encode()).hexdigest()[:8]}"

        return {
            'query_id': query_id,
            'metric': metric,
            'description': description,
            'sql': compiled_sql,
            'window_days': window_days,
            'limit': limit,
            'outputs': outputs,
            'spec_hash': self.spec_hash
        }

    def validate_dimension(self, dimension: str) -> bool:
        """
        Check if a dimension is defined in the semantic layer.

        Args:
            dimension: Dimension name to validate

        Returns:
            True if dimension exists, False otherwise
        """
        return dimension in self.get_dimensions()

    def get_dimension_source(self, dimension: str) -> Optional[str]:
        """
        Get the fully-qualified source for a dimension.

        Args:
            dimension: Dimension name

        Returns:
            Source path (e.g., 'dim_campaigns.channel') or None if not found
        """
        dims = self.get_dimensions()
        if dimension in dims:
            return dims[dimension].get('source')
        return None

    def describe_catalogue(self) -> str:
        """
        Generate a human-readable description of the semantic catalogue.

        Returns:
            Formatted string describing dimensions, metrics, and rules
        """
        lines = ["SEMANTIC CATALOGUE", "=" * 60, ""]

        # Dimensions
        lines.append("Dimensions:")
        for name, info in self.get_dimensions().items():
            lines.append(f"  • {name}: {info['source']}")
            if 'description' in info:
                lines.append(f"    {info['description']}")
        lines.append("")

        # Base queries
        lines.append("Base Queries:")
        for name, info in self.get_base_queries().items():
            lines.append(f"  • {name}: {info['description']}")
        lines.append("")

        # Derived metrics
        lines.append("Derived Metrics:")
        for name, info in self.get_derived_metrics().items():
            lines.append(f"  • {name}: {info['description']}")
        lines.append("")

        # Key join rules
        lines.append("Key Join Rules:")
        for name, info in self.get_join_rules().items():
            lines.append(f"  • {name}: {info['description']}")
        lines.append("")

        lines.append(f"Spec hash: {self.spec_hash}")

        return "\n".join(lines)
