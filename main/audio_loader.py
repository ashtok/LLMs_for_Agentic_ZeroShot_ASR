import csv
from pathlib import Path
from typing import Tuple, List, Dict

import librosa
import numpy as np
from datasets import Dataset, Audio


def load_transcriptions(txt_path: str) -> Dict[str, str]:
    """
    Reads transcriptions.txt and returns {filename: text}.
    Expected format per line: "filename<whitespace>transcription"
    """
    mapping: Dict[str, str] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)  # split on any whitespace
            if len(parts) != 2:
                continue
            fname, text = parts
            mapping[fname] = text.strip()
    return mapping


class HFAudioLoader:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.dataset: Dataset | None = None

    def from_dir(
        self,
        audio_dir: str,
        pattern: str = "hindi_*.wav",
    ) -> Dataset:
        """
        Build a Dataset with only audio paths.
        """
        audio_dir_path = Path(audio_dir)
        paths = sorted(str(p) for p in audio_dir_path.glob(pattern))
        if not paths:
            raise ValueError(f"No files matching {pattern} in {audio_dir}")

        ds = Dataset.from_dict({"audio": paths})
        # Store as Audio but DO NOT decode (avoid torchcodec); we decode via librosa
        ds = ds.cast_column("audio", Audio(decode=False))
        self.dataset = ds
        return ds

    def from_dir_with_text(
        self,
        audio_dir: str,
        transcriptions_path: str,
        pattern: str = "hindi_*.wav",
    ) -> Dataset:
        """
        Build a Dataset with audio paths and a 'text' column from transcriptions.txt.
        """
        audio_dir_path = Path(audio_dir)
        paths = sorted(list(audio_dir_path.glob(pattern)))
        if not paths:
            raise ValueError(f"No files matching {pattern} in {audio_dir}")

        trans_map = load_transcriptions(transcriptions_path)

        audio_paths_str: List[str] = []
        texts: List[str] = []
        for p in paths:
            audio_paths_str.append(str(p))
            fname = p.name  # e.g. "hindi_000.wav"
            texts.append(trans_map.get(fname, ""))

        ds = Dataset.from_dict({"audio": audio_paths_str, "text": texts})
        ds = ds.cast_column("audio", Audio(decode=False))
        self.dataset = ds
        return ds

    def get_example(self, idx: int) -> Tuple[np.ndarray, int, str]:
        """
        Returns: waveform, sample_rate, path
        """
        assert self.dataset is not None, "Call from_dir()/from_dir_with_text() first."
        audio_info = self.dataset[idx]["audio"]
        path = audio_info["path"] if isinstance(audio_info, dict) else audio_info
        waveform, sr = librosa.load(path, sr=self.target_sr)
        return waveform, sr, path

    def get_batch(
        self,
        indices: List[int],
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Returns:
            waveforms: list of 1D np.ndarrays
            sample_rates: list of int
            paths: list of str
        """
        waveforms: List[np.ndarray] = []
        srs: List[int] = []
        paths: List[str] = []
        for idx in indices:
            w, sr, p = self.get_example(idx)
            waveforms.append(w)
            srs.append(sr)
            paths.append(p)
        return waveforms, srs, paths


def main():
    base_dir = r"D:\Masters In Germany\Computer Science\Semester 5\Thesis\NLP_Thesis\data\hindi_audio"
    audio_dir = base_dir
    trans_path = str(Path(base_dir) / "transcriptions.txt")

    loader = HFAudioLoader(target_sr=16_000)

    ds = loader.from_dir_with_text(audio_dir, trans_path)
    for i in range(3):
        print(ds[i]["audio"], "|||", ds[i]["text"])

    print(f"Dataset length: {len(ds)}")
    print("Features:", ds.features)

    # Single example
    waveform, sr, path = loader.get_example(0)
    print(f"First file path: {path}")
    print(f"Sample rate: {sr}")
    print(f"Waveform dtype: {waveform.dtype}")
    print(f"Waveform shape: {waveform.shape}")
    print(f"First 10 samples: {waveform[:10]}")

    # Small batch
    waveforms, srs, paths = loader.get_batch(list(range(4)))
    print("Batch size:", len(waveforms))
    print("First path in batch:", paths[0])
    print("First waveform shape in batch:", waveforms[0].shape)


if __name__ == "__main__":
    main()

