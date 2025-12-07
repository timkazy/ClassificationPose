# test_cuda.py
import torch
print("PyTorch версия:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA версия:", torch.version.cuda)
    print("Имя GPU:", torch.cuda.get_device_name(0))
    print("Память GPU:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")