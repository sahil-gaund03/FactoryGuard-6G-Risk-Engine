import os
from pathlib import Path

# Paths are resolved relative to this file's position (project_root is two levels up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Config Directories
CONFIG_DIR = PROJECT_ROOT / "config"

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
RAW_DATA_PATH = DATA_RAW / "Thales_Group_Manufacturing.csv"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_FEATURES = DATA_DIR / "features"
DATA_PREDICTIONS = DATA_DIR / "predictions"
DATA_REPORTS = DATA_DIR / "reports"

# Model Directories
MODELS_DIR = PROJECT_ROOT / "models"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"

def setup_directories():
    """Ensure that all necessary directories exist in the workspace."""
    directories = [
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        DATA_FEATURES,
        DATA_PREDICTIONS,
        DATA_REPORTS,
        MODELS_DIR,
        REPORTS_DIR,
        REPORTS_DIR / "figures",
        REPORTS_DIR / "tables",
        REPORTS_DIR / "research_paper",
        REPORTS_DIR / "executive_summary",
        REPORTS_DIR / "model_card",
        REPORTS_DIR / "data_quality",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

setup_directories()
