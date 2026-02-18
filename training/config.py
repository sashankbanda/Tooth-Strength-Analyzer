
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')

# Training Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_WORKERS = 0 # Set to 0 for Windows compatibility, otherwise 2 or 4

# Data subdirectories
DETECTION_IMAGES_DIR = os.path.join(DATASET_DIR, 'detection', 'images')
DETECTION_LABELS_PATH = os.path.join(DATASET_DIR, 'detection', 'annotations.json') # COCO format

SEGMENTATION_IMAGES_DIR = os.path.join(DATASET_DIR, 'segmentation', 'images')
SEGMENTATION_MASKS_DIR = os.path.join(DATASET_DIR, 'segmentation', 'masks')

# Model saving
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
