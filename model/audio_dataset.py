import os
import re
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Tuple, List
from .mulaw import mu_law_encode


@dataclass
class AudioDatasetConfig:
    data_dir: str                                  # Path to directory with .wav files
    sequence_length: int                           # Number of samples per training sequence
    parameter_specs: Dict[str, Tuple[float, float]]  # {'param': (min, max), ...}
    add_noise: bool = False                        # Whether to add white noise
    snr_db: float = 10.0                           # Desired signal-to-noise ratio (dB)


class MuLawAudioDataset(Dataset):
    def __init__(self, config: AudioDatasetConfig):
        self.config = config
        self.sequence_length = config.sequence_length
        self.param_specs = config.parameter_specs
        self.add_noise = config.add_noise
        self.snr_db = config.snr_db

        self.data = []  # List of (mu_law_tensor, norm_params_tensor)
        self._load_files()

    def _load_files(self):
        file_list = [f for f in os.listdir(self.config.data_dir) if f.endswith('.wav')]

        for fname in sorted(file_list):
            path = os.path.join(self.config.data_dir, fname)
            waveform, sr = torchaudio.load(path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

            waveform = waveform.squeeze(0)  # shape: [samples]

            if self.add_noise:
                signal_rms = waveform.pow(2).mean().sqrt()
                noise_rms = signal_rms / (10 ** (self.snr_db / 20))
                noise = torch.randn_like(waveform) * noise_rms
                headroom = 1.0 - waveform.abs()
                noise = torch.clamp(noise, -headroom, headroom)
                waveform = waveform + noise
                waveform = waveform.clamp(-1.0, 1.0)

            encoded = mu_law_encode(waveform)  # shape: [samples]

            # Parse and normalize conditioning parameters from filename
            norm_params = self._parse_and_normalize_params(fname)

            print(f"for {fname}, normed params are: {norm_params}")
            
            if norm_params is None:
                continue  # skip file if missing required params

            total_seq = len(encoded) // self.sequence_length
            for i in range(total_seq):
                start = i * self.sequence_length
                end = start + self.sequence_length
                seq = encoded[start:end]
                if len(seq) == self.sequence_length:
                    self.data.append((seq, norm_params))

    # reads parameters from filename in order to create param vect
    def _parse_and_normalize_params(self, filename: str) -> torch.FloatTensor:
        result = []
        for key, (vmin, vmax) in self.param_specs.items():
            pattern = rf"_{key}(-?\d+\.\d+)"
            match = re.search(pattern, filename)
            if not match:
                return None
            raw_val = float(match.group(1))
            norm_val = (raw_val - vmin) / (vmax - vmin)
            result.append(norm_val)
        return torch.tensor(result, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mu_law_seq, norm_params = self.data[idx]  # [T], [p]
        input_seq = mu_law_seq[:-1].float() / 255.0  # [T-1], float in [0,1]
        target_seq = mu_law_seq[1:].long()          # [T-1]

        cond_params = norm_params.unsqueeze(0).expand(input_seq.shape[0], -1)  # [T-1, p]
        return input_seq, cond_params, target_seq
