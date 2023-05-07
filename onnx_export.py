import io
import numpy as np

from torch import nn
import torch.onnx
import model
from utils import load_state_dict

def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device

def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    sr_model = model.__dict__[model_arch_name](in_channels=3,
                                               out_channels=3,
                                               channels=64,
                                               num_rcb=16)
    sr_model = load_state_dict(sr_model, "./results/pretrained_models/SRGAN_x4-ImageNet-8c4a7569.pth")
    sr_model = sr_model.to(device=device)
    sr_model.eval()

    return sr_model

x = torch.randn(1, 3, 224, 224, requires_grad=True)
model = build_model("srresnet_x4", "cpu")

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch', 2: 'width', 3: 'height'},    # variable length axes
                                'output' : {0 : 'batch', 2: 'width', 3: 'height'}})