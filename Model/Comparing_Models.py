# Defines the architecture of benchmark models such as MLP and CNN
import torch
import torch.nn as nn
from Model.Model import MLP as Encoder
from Model.Model import Predictor

# ---------------------------------------------------------------------
# residual convolutional block 
# ---------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, use_leaky=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1) if use_leaky else nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        self.activation = nn.ReLU(inplace=True)

    # defining forward pass
    def forward(self, x):
        out = self.conv(x)
        out = out + self.skip(x)
        return self.activation(out)


# ---------------------------------------------------------------------
# baseline MLP model
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, dropout=0.2)
        self.predictor = Predictor(input_dim=32)

    # defining forward pass
    def forward(self, x):
        emb = self.encoder(x)
        return self.predictor(emb)


# ---------------------------------------------------------------------
# CNN model with residual connections
# ---------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResidualConvBlock(1, 8, stride=1, kernel_size=3)
        self.layer2 = ResidualConvBlock(8, 16, stride=2, kernel_size=5, use_leaky=True)
        self.layer3 = ResidualConvBlock(16, 24, stride=2, kernel_size=3)
        self.layer4 = ResidualConvBlock(24, 16, stride=1, kernel_size=3)
        self.layer5 = ResidualConvBlock(16, 8, stride=1, kernel_size=3)
        self.linear_head = nn.Linear(8 * 5, 1)
        self.apply(self._init_weights)

    # initialization of weights using Kaiming normal for convolutional and Xavier for linear
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # forward pass
    def forward(self, x):
        N, L = x.shape
        x = x.view(N, 1, L)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.linear_head(out.view(N, -1))
        return out.view(N, 1)


# ---------------------------------------------------------------------
# utility for counting trainable parameters
# ---------------------------------------------------------------------
def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model has {count:,} trainable parameters")
    return count
