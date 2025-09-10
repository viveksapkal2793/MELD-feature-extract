import torch

if torch.cuda.is_available():
    print("✅ CUDA is available")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ CUDA not available, running on CPU")
