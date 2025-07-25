import numpy as np
import torch

def mu_law_encode(audio, quantization_channels=256):
    """
    Encode waveform using mu-law companding.
    audio: np.ndarray or torch.Tensor of floats in [-1, 1]
    Returns: int tensor or array of values in [0, quantization_channels - 1]
    """
    mu = quantization_channels - 1
    safe_audio = torch.clamp(audio, -1.0, 1.0) if isinstance(audio, torch.Tensor) else np.clip(audio, -1.0, 1.0)
    encoded = torch.sign(safe_audio) * torch.log1p(mu * torch.abs(safe_audio)) / np.log1p(mu) if isinstance(audio, torch.Tensor) \
        else np.sign(safe_audio) * np.log1p(mu * np.abs(safe_audio)) / np.log1p(mu)
    scaled = ((encoded + 1) / 2 * mu + 0.5).long() if isinstance(audio, torch.Tensor) else ((encoded + 1) / 2 * mu + 0.5).astype(np.int32)
    return scaled

def mu_law_decode(encoded, quantization_channels=256):
    """
    Decode mu-law encoded signal.
    encoded: int tensor or array of values in [0, quantization_channels - 1]
    Returns: float tensor or array of values in [-1, 1]
    """
    mu = quantization_channels - 1
    x = encoded.float() if isinstance(encoded, torch.Tensor) else encoded.astype(np.float32)
    x = 2 * (x / mu) - 1
    decoded = torch.sign(x) * (1 / mu) * (torch.pow(1 + mu, torch.abs(x)) - 1) if isinstance(encoded, torch.Tensor) \
        else np.sign(x) * (1 / mu) * (np.power(1 + mu, np.abs(x)) - 1)
    return decoded
