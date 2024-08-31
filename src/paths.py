from pathlib import Path


PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'
SUBMISSION_DIR = DATA_DIR / 'submission'
MODEL_DIR = PARENT_DIR / 'models'
