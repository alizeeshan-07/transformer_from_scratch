import torch

# Check if CUDA or MPS is available
if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available


# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(f"Tensor on CPU {tensor, tensor.device}")


# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(f"Tensor on GPU {tensor_on_gpu}")

# tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu()
print(f"Tensor back on CPU {tensor_back_on_cpu, tensor_back_on_cpu.device}")

# numpy doesn't support GPU that is why we can directly convert the GPU tensor to CPU to be used with numpy
# original tensor will still be on GPU
tensor_back_on_cpu_on_numpy = tensor_on_gpu.cpu()
print(f"Tensor in Numpy {tensor_back_on_cpu_on_numpy, tensor_back_on_cpu_on_numpy.device}")

# Move a tensor to the MPS device
random_tensor_on_gpu = torch.randn(3, 3).to(device)
print(f"Random Tensor on GPU {random_tensor_on_gpu }")



# # Check for Apple Silicon GPU
# torch.backends.mps.is_available() # Note this will print false if you're not running on a Mac
# # Set device type
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# device