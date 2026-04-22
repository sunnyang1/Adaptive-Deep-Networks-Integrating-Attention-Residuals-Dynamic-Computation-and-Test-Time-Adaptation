import torch

TensorDict = dict[str, torch.Tensor]
MaybeTensor = torch.Tensor | None
