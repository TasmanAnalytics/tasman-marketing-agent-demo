"""
Observability utilities - structured run records for reproducibility.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class RunRecord:
    """
    Structured run record for observability and reproducibility.
    """

    def __init__(self, model_name: str, semantic_spec_hash: str):
        """
        Initialize a new run record.

        Args:
            model_name: LLM model name used
            semantic_spec_hash: Hash of semantic layer spec
        """
        self.run_id = self._generate_run_id()
        self.model_name = model_name
        self.semantic_spec_hash = semantic_spec_hash
        self.start_time = datetime.now().isoformat()
        self.end_time = None

        # Run data
        self.triage_decision = None
        self.semantic_request = None
        self.compiled_queries = []
        self.execution_timings = {}
        self.row_counts = {}
        self.hypothesis_params = None
        self.hypothesis_result = None
        self.narration = None
        self.artifacts = []
        self.test_results = {}
        self.errors = []

    def _generate_run_id(self) -> str:
        """Generate unique run ID from timestamp and random hash."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_hash = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"run_{timestamp}_{random_hash}"

    def record_triage(self, decision: Dict[str, Any]):
        """Record triage decision."""
        self.triage_decision = decision

    def record_semantic_request(self, request: Dict[str, Any]):
        """Record semantic mapping result."""
        self.semantic_request = request

    def record_query(self, query_info: Dict[str, Any]):
        """Record a compiled query."""
        self.compiled_queries.append({
            'query_id': query_info.get('query_id'),
            'metric': query_info.get('metric'),
            'window_days': query_info.get('window_days'),
            'spec_hash': query_info.get('spec_hash')
        })

    def record_execution(self, query_id: str, elapsed_ms: float, row_count: int):
        """Record query execution timing and row count."""
        self.execution_timings[query_id] = elapsed_ms
        self.row_counts[query_id] = row_count

    def record_hypothesis(self, params: Dict[str, Any], result: Dict[str, Any]):
        """Record hypothesis simulation parameters and results."""
        self.hypothesis_params = params
        self.hypothesis_result = result

    def record_narration(self, text: str):
        """Record narration output."""
        self.narration = text

    def add_artifact(self, artifact_type: str, path: str):
        """Add reference to saved artifact."""
        self.artifacts.append({
            'type': artifact_type,
            'path': path
        })

    def record_test(self, test_name: str, passed: bool, details: Optional[str] = None):
        """Record test result."""
        self.test_results[test_name] = {
            'passed': passed,
            'details': details
        }

    def record_error(self, error_type: str, message: str):
        """Record an error that occurred during execution."""
        self.errors.append({
            'type': error_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    def finalize(self):
        """Mark run as complete."""
        self.end_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert run record to dictionary.

        Returns:
            Dictionary representation of run record
        """
        return {
            'run_id': self.run_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'environment': {
                'model_name': self.model_name,
                'semantic_spec_hash': self.semantic_spec_hash
            },
            'triage_decision': self.triage_decision,
            'semantic_request': self.semantic_request,
            'compiled_queries': self.compiled_queries,
            'execution_timings': self.execution_timings,
            'row_counts': self.row_counts,
            'hypothesis': {
                'params': self.hypothesis_params,
                'result': self.hypothesis_result
            },
            'narration': self.narration,
            'artifacts': self.artifacts,
            'test_results': self.test_results,
            'errors': self.errors
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert run record to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, output_dir: str = './outputs'):
        """
        Save run record to JSON file.

        Args:
            output_dir: Directory to save file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / f"{self.run_id}.json"
        with open(file_path, 'w') as f:
            f.write(self.to_json())

        return str(file_path)

    def summary(self) -> str:
        """
        Generate a concise summary of the run.

        Returns:
            Human-readable summary string
        """
        lines = [
            f"Run ID: {self.run_id}",
            f"Model: {self.model_name}",
            f"Queries executed: {len(self.compiled_queries)}",
            f"Total timing: {sum(self.execution_timings.values()):.2f} ms",
            f"Tests passed: {sum(1 for t in self.test_results.values() if t['passed'])}/{len(self.test_results)}",
            f"Errors: {len(self.errors)}"
        ]
        return "\n".join(lines)
