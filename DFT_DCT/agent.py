import os
from perform import performer

if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("")
    MODEL_PATH = os.path.join(ROOT_DIR)
    performer(MODEL_PATH)