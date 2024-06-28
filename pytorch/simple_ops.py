import torch
from torch_mlir import torchscript

import torch.nn as nn
import torch.nn.functional as F

output_type = "stablehlo"

class Model(torch.nn.Module):
    def forward(self, a, b):
        return a + b

model = Model()

a = torch.randn(128, 128)
b = torch.randn(128, 128)
params = (a, b)

print(torchscript.compile(model, params, output_type=output_type))