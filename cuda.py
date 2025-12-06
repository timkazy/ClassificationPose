import torch

print("=" * 50)
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Версия CUDA в PyTorch: {getattr(torch.version, 'cuda', 'Нет')}")
print(f"Имя GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Нет GPU'}")
print("=" * 50)

# Более правильная проверка CUDA сборки
has_cuda_in_version = any(x in torch.__version__.lower() for x in ['cuda', 'cu'])
print(f"Build with CUDA (строка версии): {has_cuda_in_version}")

# Или так
print(f"Содержит 'cu' в версии: {'cu' in torch.__version__.lower()}")

# Самый надежный способ - проверить доступность CUDA функций
if torch.cuda.is_available():
    print(f"CUDA устройств: {torch.cuda.device_count()}")
    print(f"Текущее устройство: {torch.cuda.current_device()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")