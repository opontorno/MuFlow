import torch
import GPUtil


def select_best_gpu():
    """Return the id of the GPU with the most free memory.
    Returns: int gpu id, 0 if none found by GPUtil, or None if no CUDA device.
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
    """Return a torch.device.
    gpu_id: manual GPU id, or None to auto-select the freest GPU.
    Returns: torch.device (CPU if no GPU is available).
    """
    if gpu_id is not None:
        print(f"📌 Using manually specified GPU {gpu_id}")
        return torch.device(f'cuda:{gpu_id}')

    selected = select_best_gpu()
    if selected is not None:
        return torch.device(f'cuda:{selected}')
    print("⚠️  No GPU available, using CPU")
    return torch.device('cpu')
