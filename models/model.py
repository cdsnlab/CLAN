from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class ConTF(nn.Module):
    def __init__(self, configs, args):
        super(ConTF, self).__init__()