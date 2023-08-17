import os

DATASET = "MR"

MAX_LENGTH = 67 #mean of mean and max values

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

EMB_PATH = os.path.join(BASE_PATH, "embeddings")

DATA_PATH = os.path.join(BASE_PATH, "datasets")

