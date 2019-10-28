import torch

WINDOW_SIZE: int = 15

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    DTYPE = torch.float32
else:
    DEVICE = torch.device('cpu')
    DTYPE = torch.float32
