import math

import torch

WINDOW_SIZE: int = 15
TOTAL_POINTS = 1085913
TRAIN_START = 0
TRAIN_END = math.floor(TOTAL_POINTS * 0.6)
TEST_END = TOTAL_POINTS
VALIDATE_END = math.floor(TOTAL_POINTS * 0.8)
BATCH_SIZE = TOTAL_POINTS // 100 # 100 steps per epoch
HIDDEN_DIM = 51
REPRODUCIBLE = False

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    DTYPE = torch.float32
else:
    DEVICE = torch.device('cpu')
    DTYPE = torch.float32
