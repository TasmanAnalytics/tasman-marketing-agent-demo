"""
Environment loader - validates required environment variables.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Optional


def load_environment(env_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load and validate environment variables.

    Args:
        env_path: Optional path to .env file. If None, searches parent directories.

    Returns:
        Dictionary with validated environment variables

    Raises:
        ValueError: If required variables are missing
    """
    # Load .env file
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    # Required variables
    required_vars = ['OPENAI_API_KEY']

    # Validate
    missing = []
    env_config = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        else:
            env_config[var] = value

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # Optional variables with defaults
    env_config['OPENAI_MODEL'] = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    env_config['DB_PATH'] = os.getenv('DB_PATH', '../data/synthetic_data.duckdb')

    return env_config


def get_model_name() -> str:
    """Get the OpenAI model name from environment."""
    return os.getenv('OPENAI_MODEL', 'gpt-4o-mini')


def get_db_path() -> str:
    """Get the database path from environment."""
    return os.getenv('DB_PATH', '../data/synthetic_data.duckdb')
