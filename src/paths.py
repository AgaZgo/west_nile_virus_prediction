from pathlib import Path
import os


PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
SUBMISSION_DIR = DATA_DIR / 'submission'
MODEL_DIR = PARENT_DIR / 'models'


def make_dir_if_not_exist(directory):
    if not Path(directory).exists():
        os.mkdir(directory)


make_dir_if_not_exist(SUBMISSION_DIR)
make_dir_if_not_exist(MODEL_DIR)
