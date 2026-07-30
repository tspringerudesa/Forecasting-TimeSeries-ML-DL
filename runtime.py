"""Shared, conservative compute settings for the thesis experiments."""

import os


def _positive_int_from_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


# Four workers provide useful estimator-level parallelism without letting
# nested numerical libraries occupy every logical core on a laptop.
CPU_THREADS = _positive_int_from_environment(
    "THESIS_CPU_THREADS",
    min(4, os.cpu_count() or 1),
)

def neuralforecast_compute_parameters() -> dict:
    """Select CUDA explicitly when available and configure Tensor Cores."""
    try:
        import torch
    except ImportError:
        return {"accelerator": "cpu", "devices": 1}
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return {"accelerator": "gpu", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}
