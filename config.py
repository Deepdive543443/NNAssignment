import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Path
CIFAR_ROOT = "CIFAR10"
TRANSFORM = True

#Model
SAVE_MODEL = True
LOAD_MODEL = False
FLOAT16 = False

#Dataset
TEST_SPLIT = None#240
SHUFFLE = True
BATCH_SIZE = 256
NUM_WORKERS = 0
EPOCH = 1000

#Hyper parameters
LEARNING_RATE = 3e-4
# CRITIC = 5
# BETA1 = 0.5
# BETA2 = 0.999

KFOLD_SPLIT = 5
USE_BATCHNORM=True