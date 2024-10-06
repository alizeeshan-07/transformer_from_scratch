import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Move a tensor to the MPS device
x = torch.randn(3, 3).to(device)
print(x)