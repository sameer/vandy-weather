import math

import torch

WINDOW_SIZE: int = 15
TOTAL_POINTS = 1085913
TRAIN_END = math.floor(TOTAL_POINTS * 0.6)
VALIDATE_END = math.floor(TOTAL_POINTS * 0.8)

CPU = torch.device('cpu')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    DTYPE = torch.float32
else:
    DEVICE = CPU
    DTYPE = torch.float32
