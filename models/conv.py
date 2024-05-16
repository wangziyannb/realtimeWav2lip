import torch
from torch import nn
from torch.ao.nn.quantized import QFunctional
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.nn import functional as F
import torch.nn.quantized.functional as qF


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        out = self.conv_block(x)
        if self.residual:
            # out += x
            if x.is_quantized and out.is_quantized:
                out = QFunctional().add(out, x)
            else:
                out += x
        x = self.act(out)
        # x = self.dequant(x)
        return x


class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        out = self.conv_block(x)
        x = self.act(out)
        # x = self.dequant(x)
        return x


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()

    def forward(self, x):
        # x = self.dequant(x)
        # x = self.quant(x)
        out = self.conv_block(x)
        x = self.act(out)
        # x = self.dequant(x)
        # x = self.quant(x)
        return x
