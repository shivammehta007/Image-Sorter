import logging

import torch

LOGGING_LEVEL = logging.INFO

LOGGING_FORMAT = (
    "[%(levelname)s | %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["Needed", "Not Needed"]
TRAINED_MODEL_PATH = "trained_models"
TRAIN_DATA_FOLDER = "Train"
TEST_DATA_FOLDER = "Test"
epochs = 10
learning_rate = 0.001
dropout=0.4
run_name = "test_run1"
regularisation = 0.001

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
