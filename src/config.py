import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'lipread_mp4')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'current')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 112
DROPOUT = 0.5

MAX_CLASSES = None  # Set to None to use all classes

BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_lip_reading_model.pt')
CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'word_mappings.json')
