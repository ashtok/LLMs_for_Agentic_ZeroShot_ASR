import csv
from pathlib import Path
from typing import Tuple, List, Dict
import sys

import librosa
import numpy as np
from datasets import Dataset, Audio

# Import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT,
    LANGUAGE,
    TRANSCRIPTIONS_FILE,
    ASR_SAMPLING_RATE,
    AUDIO_FILE_PATTERN,
    CLIPS_SUBDIR,
)


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
        pattern: str = "*.mp3",
        clips_subdir: str = "clips",
    ) -> Dataset:
        """
        Build a Dataset with only audio paths.
        
        Args:
            audio_dir: Base directory containing audio files
            pattern: Glob pattern for audio files (default: *.mp3)
            clips_subdir: Subdirectory containing clips (default: clips)
        """
        audio_dir_path = Path(audio_dir) / clips_subdir
        paths = sorted(str(p) for p in audio_dir_path.glob(pattern))
        if not paths:
            raise ValueError(f"No files matching {pattern} in {audio_dir_path}")

        ds = Dataset.from_dict({"audio": paths})
        # Store as Audio but DO NOT decode (avoid torchcodec); we decode via librosa
        ds = ds.cast_column("audio", Audio(decode=False))
        self.dataset = ds
        return ds

    def from_dir_with_text(
        self,
        audio_dir: str,
        transcriptions_path: str,
        pattern: str = "*.mp3",
        clips_subdir: str = "clips",
    ) -> Dataset:
        """
        Build a Dataset with audio paths and a 'text' column from transcriptions.txt.
        
        Args:
            audio_dir: Base directory containing audio files
            transcriptions_path: Path to transcriptions.txt
            pattern: Glob pattern for audio files (default: *.mp3)
            clips_subdir: Subdirectory containing clips (default: clips)
        """
        audio_dir_path = Path(audio_dir) / clips_subdir
        
        if not audio_dir_path.exists():
            raise ValueError(f"Clips directory not found: {audio_dir_path}")
        
        # Load transcriptions mapping first
        trans_map = load_transcriptions(transcriptions_path)
        
        if not trans_map:
            raise ValueError(f"No transcriptions loaded from {transcriptions_path}")
        
        # Find all audio files matching pattern
        all_paths = sorted(list(audio_dir_path.glob(pattern)))
        if not all_paths:
            raise ValueError(f"No files matching {pattern} in {audio_dir_path}")

        # Filter to only files that have transcriptions
        audio_paths_str: List[str] = []
        texts: List[str] = []
        
        for p in all_paths:
            fname = p.name  # e.g. "common_voice_hi_43600788.mp3"
            if fname in trans_map:
                audio_paths_str.append(str(p))
                texts.append(trans_map[fname])
        
        if not audio_paths_str:
            raise ValueError(
                f"No audio files found with matching transcriptions. "
                f"Found {len(all_paths)} audio files but none in transcriptions."
            )

        ds = Dataset.from_dict({"audio": audio_paths_str, "text": texts})
        ds = ds.cast_column("audio", Audio(decode=False))
        self.dataset = ds
        
        print(f"Loaded {len(ds)} audio files with transcriptions from {audio_dir_path}")
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
    # Use config paths
    data_dir = DATA_ROOT / LANGUAGE
    transcriptions_path = data_dir / TRANSCRIPTIONS_FILE
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not transcriptions_path.exists():
        raise FileNotFoundError(f"Transcriptions file not found: {transcriptions_path}")
    
    print(f"Loading data from: {data_dir}")
    print(f"Using transcriptions: {transcriptions_path}")
    print(f"Audio pattern: {AUDIO_FILE_PATTERN}")
    print(f"Clips subdirectory: {CLIPS_SUBDIR}\n")

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)

    ds = loader.from_dir_with_text(
        str(data_dir),
        str(transcriptions_path),
        pattern=AUDIO_FILE_PATTERN,
        clips_subdir=CLIPS_SUBDIR,
    )
    
    print(f"\nDataset length: {len(ds)}")
    print("Features:", ds.features)
    
    # Show first 3 examples
    print("\nFirst 3 examples:")
    for i in range(min(3, len(ds))):
        print(f"{i}: {ds[i]['audio']} ||| {ds[i]['text']}")

    # Single example
    print("\n--- Single Example Test ---")
    waveform, sr, path = loader.get_example(0)
    print(f"First file path: {path}")
    print(f"Sample rate: {sr}")
    print(f"Waveform dtype: {waveform.dtype}")
    print(f"Waveform shape: {waveform.shape}")
    print(f"First 10 samples: {waveform[:10]}")

    # Small batch
    print("\n--- Batch Test ---")
    batch_size = min(4, len(ds))
    waveforms, srs, paths = loader.get_batch(list(range(batch_size)))
    print(f"Batch size: {len(waveforms)}")
    print(f"First path in batch: {paths[0]}")
    print(f"First waveform shape in batch: {waveforms[0].shape}")


if __name__ == "__main__":
    main()
