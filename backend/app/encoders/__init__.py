"""검색 인코더 (MiniLM 텍스트 · Marqo-FashionSigLIP 이미지/텍스트)."""


def set_torch_threads() -> None:
    """settings.torch_threads로 torch 스레드 수를 맞춘다 (인코더를 올릴 때 한 번)."""
    from ..config import settings
    if settings.torch_threads > 0:
        import torch
        torch.set_num_threads(settings.torch_threads)
