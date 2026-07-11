import yaml
from pathlib import Path
from src.factoryguard.paths import CONFIG_DIR
from src.factoryguard.exceptions import ConfigurationError

def load_yaml(file_path: Path) -> dict:
    """Load a YAML configuration file safely."""
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigurationError(f"Error reading YAML file {file_path}: {e}")

def get_base_config() -> dict:
    return load_yaml(CONFIG_DIR / "base.yaml")

def get_features_config() -> dict:
    return load_yaml(CONFIG_DIR / "features.yaml")

def get_models_config() -> dict:
    return load_yaml(CONFIG_DIR / "models.yaml")
