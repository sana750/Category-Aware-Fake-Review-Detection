from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "processed"
COLAB_DATA_DIR = Path(
    "/content/drive/MyDrive/Category-Aware Fake Review Detection/"
    "data/processed/processed"
)

FAKE_LABEL = 1
GENUINE_LABEL = 0
CLASSIFICATION_THRESHOLD = 0.5

