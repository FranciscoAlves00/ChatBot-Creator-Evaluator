import os

# Define the base directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.path.join(BASE_DIR, "docs")
SMALL_MISTRAL_PATH = os.path.join(
    BASE_DIR, "models/mistral/mistral-7b-instruct-v0.1.Q2_K-001.gguf"
)
BASE_MISTRAL_PATH = os.path.join(
    BASE_DIR, "models/mistral/mistral-7b-instruct-v0.1.Q4_K_M-002.gguf"
)
RERANKER_PATH = os.path.join(BASE_DIR, "models/bge-reranker-base")
BASE_BGE_PATH = os.path.join(BASE_DIR, "models/bge-base-en-v1.5")
LARGE_BGE_PATH = os.path.join(BASE_DIR, "models/bge-large-en-v1.5")
SMALL_BGE_PATH = os.path.join(BASE_DIR, "models/bge-small-en-v1.5")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "ground_truth/gt_dataset")

CHATBOT_URL = "http://localhost:8505"
