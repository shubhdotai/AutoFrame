"""Inference-only LR-ASD wrapper with strict checkpoint loading."""
import torch
from torch import nn
from .loss import lossAV, lossV
from .model.Model import ASD_Model

class ASD(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.model = ASD_Model()
        self.lossAV = lossAV()
        self.lossV = lossV()
        self.to(device)

    def loadParameters(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.load_state_dict(state, strict=True)
