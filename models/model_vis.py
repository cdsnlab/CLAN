import torch
from torch import nn
from .Transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import seaborn
import matplotlib.pyplot as plt

# Encoders for novelty detection
class ConTF(nn.Module):
    def __init__(self, configs, args):
        super(ConTF, self).__init__()