import torch
from torch_mlir import torchscript

import torch.nn as nn
import torch.nn.functional as F

output_type = "stablehlo"
        
model = nn.ReflectionPad1d(2)

a = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
params = a

print(torchscript.compile(model, params, output_type=output_type))