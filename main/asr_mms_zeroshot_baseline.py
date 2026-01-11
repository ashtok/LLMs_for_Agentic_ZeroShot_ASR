from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import sys

import librosa
import numpy as np
import torch
from jiwer import wer, cer
from transformers import AutoProcessor, Wav2Vec2ForCTC

# Import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT,
    LANGUAGE,
    TRANSCRIPTIONS_FILE,
    TRANSCRIPTIONS_UROMAN_FILE,
    ASR_SAMPLING_RATE,
    AUDIO_FILE_PATTERN,
    CLIPS_SUBDIR,
)

from audio_loader import HFAudioLoader


MODEL_ID = "mms-meta/mms-zeroshot-300m"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")
    else:
        return torch.device("cpu")


def run_mms_zeroshot_baseline_basic(
    loader: HFAudioLoader,
    ds: Any,
    refs_roman: List[str],
    model_id: str = MODEL_ID,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run MMS zero-shot baseline evaluation
    """
    if verbose:
        print("[MMS-ZS-BASIC] Loading MMS zero-shot model and processor...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    device = _get_device()
    model.to(device)
    if verbose:
        print(f"[MMS-ZS-BASIC] Using device: {device}")

    hyps_roman: List[str] = []

    N = len(ds)
    if len(refs_roman) != N:
        raise ValueError(
            f"refs_roman length ({len(refs_roman)}) does not match dataset size ({N}). "
            "Make sure transcriptions_uroman.txt is aligned with transcriptions.txt."
        )

    for i in range(N):
        audio_info = ds[i]["audio"]
        path = audio_info["path"] if isinstance(audio_info, dict) else audio_info

        audio, sr = librosa.load(path, sr=ASR_SAMPLING_RATE, mono=True)
        audio = audio.astype(np.float32)

        inputs = processor(
            audio,
            sampling_rate=ASR_SAMPLING_RATE,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        hyp_roman = processor.batch_decode(pred_ids)[0].strip()
        hyps_roman.append(hyp_roman)

        ref_dev = ds[i]["text"]
        ref_roman = refs_roman[i]

        if verbose:
            filename = Path(path).name
            print(f"[MMS-ZS-BASIC] Sample {i+1}/{N}")
            print(f"FILE: {filename}")
            print(f"REF_DEV: {ref_dev}")
            print(f"REF_UROMAN: {ref_roman}")
            print(f"HYP_MMS_UROMAN: {hyp_roman}\n")

    wer_val = float(wer(refs_roman, hyps_roman))
    cer_val = float(cer(refs_roman, hyps_roman))

    if verbose:
        print(f"[MMS-ZS-BASIC] WER (uroman): {wer_val:.4f}")
        print(f"[MMS-ZS-BASIC] CER (uroman): {cer_val:.4f}")

    return {
        "model": f"{model_id} (zeroshot-greedy, uroman)",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
        "hyps": hyps_roman,
        "refs": refs_roman,
    }


def main() -> None:
    # Use config paths
    data_dir = DATA_ROOT / LANGUAGE
    transcriptions_path = data_dir / TRANSCRIPTIONS_FILE
    roman_path = data_dir / TRANSCRIPTIONS_UROMAN_FILE
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not transcriptions_path.exists():
        raise FileNotFoundError(f"Transcriptions file not found: {transcriptions_path}")
    
    if not roman_path.exists():
        raise FileNotFoundError(f"Romanized transcriptions not found: {roman_path}")
    
    print(f"Loading data from: {data_dir}")
    print(f"Using transcriptions: {transcriptions_path}")
    print(f"Using romanized transcriptions: {roman_path}\n")

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    
    # Dataset with Devanagari references
    ds = loader.from_dir_with_text(
        str(data_dir),
        str(transcriptions_path),
        pattern=AUDIO_FILE_PATTERN,
        clips_subdir=CLIPS_SUBDIR,
    )

    # Load romanized refs into a dictionary keyed by filename
    refs_roman_map = {}
    with roman_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                filename, rom_text = parts
                refs_roman_map[filename] = rom_text.strip()
    
    print(f"Loaded {len(refs_roman_map)} romanized transcriptions")
    
    # Build refs_roman list aligned with dataset order
    refs_roman = []
    missing_count = 0
    
    for i in range(len(ds)):
        audio_info = ds[i]["audio"]
        path = audio_info["path"] if isinstance(audio_info, dict) else audio_info
        filename = Path(path).name
        
        if filename in refs_roman_map:
            refs_roman.append(refs_roman_map[filename])
        else:
            refs_roman.append("")
            missing_count += 1
            if missing_count <= 5:  # Only show first 5 warnings
                print(f"Warning: No romanized transcription found for {filename}")
    
    if missing_count > 0:
        print(f"⚠ Warning: {missing_count} files missing romanized transcriptions\n")
    else:
        print(f"✔ All {len(ds)} files have romanized transcriptions\n")

    run_mms_zeroshot_baseline_basic(
        loader=loader,
        ds=ds,
        refs_roman=refs_roman,
        model_id=MODEL_ID,
        verbose=True,
    )


if __name__ == "__main__":
    main()
