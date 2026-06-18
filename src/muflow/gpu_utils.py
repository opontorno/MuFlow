"""
GPU utilities for MuFlow.

Auto-selection of the most suitable CUDA device.
"""
import torch
import GPUtil


def select_best_gpu():
    """Automatically select the GPU with the most free memory.

    Returns:
        int  : id of the GPU with most free memory,
        0    : fallback if CUDA is available but GPUtil finds nothing,
        None : if no CUDA device is available (caller should use CPU).
    """
    if not torch.cuda.is_available():
        print("No CUDA GPUs available, using CPU")
        return None
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPUs found by GPUtil, using cuda:0")
        return 0
    best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    print(f"🎯 Auto-selected GPU {best_gpu.id}: {best_gpu.name} "
          f"(Free: {best_gpu.memoryFree}MB / {best_gpu.memoryTotal}MB)")
    return best_gpu.id


def resolve_device(gpu_id=None):
    """Resolve a torch.device from an optional manual GPU id.

    If gpu_id is None, auto-selects the GPU with most free memory.
    Falls back to CPU when no GPU is available.
    """
    if gpu_id is not None:
        print(f"📌 Using manually specified GPU {gpu_id}")
        return torch.device(f'cuda:{gpu_id}')

    selected = select_best_gpu()
    if selected is not None:
        return torch.device(f'cuda:{selected}')
    print("⚠️  No GPU available, using CPU")
    return torch.device('cpu')
