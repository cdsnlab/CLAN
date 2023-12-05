from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class ConTF(nn.Module):
    def __init__(self, configs, args):
        super(ConTF, self).__init__()
        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=1, batch_first = True)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_t = nn.Linear(configs.TSlength_aligned * configs.input_channels, args.K_shift)

    def forward(self, x_in_t):

        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t.float())
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Shifted transformation classifier"""
        s_time = self.shift_cls_layer_t(h_time)\
    
        return h_time, z_time, s_time
