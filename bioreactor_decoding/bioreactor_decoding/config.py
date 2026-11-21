from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"  # DVC metrics: cv_<model>.json, <model>.json (holdout)
CV_DIR = REPORTS_DIR / "cv"  # per-fold CV scores, <model>.csv
HOLDOUT_DIR = REPORTS_DIR / "holdout"  # holdout predictions (no measured values), <model>.csv

# Hyperparameters and pipeline settings (tracked by DVC, see dvc.yaml)
PARAMS_PATH = PROJ_ROOT / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """params.yaml as a dict ({} if the file does not exist)."""
    import yaml

    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
