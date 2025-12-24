import torch
import time

x = torch.rand((5000, 5000))

# CPU
t0 = time.time()
y = x @ x
print("CPU:", time.time() - t0)

# GPU
# GPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

if device != "cpu":
    x_gpu = x.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    y_gpu = x_gpu @ x_gpu
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
        
    print("GPU:", time.time() - t0)
else:
    print("GPU acceleration not available.")
