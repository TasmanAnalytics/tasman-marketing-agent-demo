"""LLM clients with caching for OpenAI and Anthropic."""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import openai
import anthropic


class LLMCache:
    """Filesystem cache for LLM responses."""

    def __init__(self, cache_dir: Path):
        """Initialize cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = json.dumps(args, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, *key_parts) -> Optional[str]:
        """Get cached response."""
        key = self._hash_key(*key_parts)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('response')
        return None

    def set(self, response: str, *key_parts) -> None:
        """Cache response."""
        key = self._hash_key(*key_parts)
        cache_file = self.cache_dir / f"{key}.json"

        with open(cache_file, 'w') as f:
            json.dump({
                'key_parts': key_parts,
                'response': response
            }, f, indent=2)


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic."""

    def __init__(
        self,
        provider: str = "openai",
        cache_dir: Optional[Path] = None,
        temperature: float = 0.2,
        max_tokens: int = 512
    ):
        """
        Initialize LLM client.

        Args:
            provider: 'openai' or 'anthropic'
            cache_dir: Directory for filesystem cache
            temperature: Sampling temperature
            max_tokens: Max tokens in response
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = LLMCache(cache_dir) if cache_dir else None

        # Initialize provider-specific clients
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = openai.OpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.content[0].text.strip()

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key_parts: Optional[tuple] = None
    ) -> str:
        """
        Call LLM with caching.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            cache_key_parts: Additional cache key parts

        Returns:
            LLM response
        """
        # Check cache
        if self.cache and cache_key_parts:
            cached = self.cache.get(self.provider, self.model, system_prompt, user_prompt, *cache_key_parts)
            if cached:
                return cached

        # Call LLM
        if self.provider == "openai":
            response = self._call_openai(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            response = self._call_anthropic(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Cache response
        if self.cache and cache_key_parts:
            self.cache.set(response, self.provider, self.model, system_prompt, user_prompt, *cache_key_parts)

        return response


def triage_llm(
    question: str,
    role: Optional[str],
    business_context: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """
    Triage query using LLM.

    Args:
        question: User question
        role: User role
        business_context: Business context
        llm_client: LLM client

    Returns:
        Triage result dict
    """
    system_prompt = """Classify if a question is simple descriptive "search" or requires "analysis".
Return ONLY this JSON:
{"mode": "search"|"analysis", "analysis_type": null|"hypothesis_testing"|"driver_analysis"|"segmentation"|"comparison", "confidence": 0.0-1.0, "reason":"..."}"""

    user_prompt = f"""Role: {role or 'unknown'}
Question: {question}"""

    try:
        response = llm_client.call(
            system_prompt,
            user_prompt,
            cache_key_parts=("triage", question, role or "none")
        )

        # Parse JSON response
        result = json.loads(response)
        result['method'] = 'llm'
        return result

    except Exception as e:
        # Fallback on error
        return {
            "mode": "search",
            "analysis_type": None,
            "confidence": 0.5,
            "reason": f"LLM call failed: {str(e)}",
            "method": "llm_failed"
        }


def text_to_sql_llm(
    question: str,
    role: Optional[str],
    schema_json: str,
    business_context: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """
    Generate SQL using LLM.

    Args:
        question: User question
        role: User role
        schema_json: Schema as JSON string
        business_context: Business context
        llm_client: LLM client

    Returns:
        Dict with sql, confidence, method
    """
    # Build business context summary
    role_config = business_context.get('roles', {}).get(role, {}) if role else {}
    kpis = role_config.get('kpis', [])
    dims = role_config.get('dims', [])
    defaults = business_context.get('defaults', {})
    time_window = defaults.get('time_window_days', 90)
    limit = defaults.get('limit', 1000)

    system_prompt = f"""You convert a business question into a VALID DuckDB SQL query using ONLY the provided schema.
Rules:
- Use only known tables/columns from the schema
- Default to past {time_window} days if no date filter is given (use CURRENT_DATE - INTERVAL X DAY)
- Always add LIMIT {limit} at the end
- Return EXACTLY one fenced SQL block (```sql...```) and NOTHING else
- No explanations, just the SQL"""

    user_prompt = f"""Role: {role or 'unknown'}
Question: {question}

Schema JSON:
{schema_json}

Relevant KPIs for this role: {', '.join(kpis) if kpis else 'N/A'}
Relevant dimensions: {', '.join(dims) if dims else 'N/A'}"""

    try:
        response = llm_client.call(
            system_prompt,
            user_prompt,
            cache_key_parts=("text_to_sql", question, role or "none", schema_json[:100])
        )

        # Extract SQL from fenced code block
        sql = response.strip()
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()

        return {
            'sql': sql,
            'confidence': 0.8,
            'method': 'llm',
            'valid': True,
            'validation_errors': []
        }

    except Exception as e:
        return {
            'sql': None,
            'confidence': 0.0,
            'method': 'llm_failed',
            'valid': False,
            'validation_errors': [f"LLM call failed: {str(e)}"]
        }
