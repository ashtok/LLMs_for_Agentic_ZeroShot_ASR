from pathlib import Path
from typing import Dict, List, Any
import sys

import whisper
from jiwer import wer, cer

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

from audio_loader import HFAudioLoader


def run_whisper_baseline(
    loader: HFAudioLoader,
    ds: Any,
    model_name: str = "small",
    language: str = "hi",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run OpenAI Whisper baseline on a given dataset and return metrics.
    
    Args:
        loader: Audio loader instance
        ds: Dataset loaded by audio_loader
        model_name: Whisper model size (tiny, base, small, medium, large)
        language: Language code for Whisper (e.g., "hi" for Hindi)
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with WER, CER, and other metrics
    """
    if verbose:
        print(f"[OpenAI Whisper] Loading model: whisper-{model_name}")
        print(f"[OpenAI Whisper] Language: {language}\n")
    
    model = whisper.load_model(model_name)

    refs: List[str] = []
    hyps: List[str] = []

    N = len(ds)
    for i in range(N):
        audio_info = ds[i]["audio"]
        path = audio_info["path"] if isinstance(audio_info, dict) else audio_info

        result = model.transcribe(
            path,
            language=language,
            task="transcribe",
            fp16=False,
        )

        hyp = result["text"].strip()
        ref = ds[i]["text"]

        refs.append(ref)
        hyps.append(hyp)

        if verbose:
            filename = Path(path).name
            print(f"[OpenAI Whisper] Sample {i+1}/{N}")
            print(f"FILE: {filename}")
            print(f"REF: {ref}")
            print(f"HYP: {hyp}\n")

    wer_val = float(wer(refs, hyps))
    cer_val = float(cer(refs, hyps))

    if verbose:
        print(f"[OpenAI Whisper] WER: {wer_val:.4f}")
        print(f"[OpenAI Whisper] CER: {cer_val:.4f}")

    return {
        "model": f"whisper-{model_name}",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
        "hyps": hyps,
        "refs": refs,
    }


def main():
    # Use config paths
    data_dir = DATA_ROOT / LANGUAGE
    transcriptions_path = data_dir / TRANSCRIPTIONS_FILE
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not transcriptions_path.exists():
        raise FileNotFoundError(f"Transcriptions file not found: {transcriptions_path}")
    
    print(f"Loading data from: {data_dir}")
    print(f"Using transcriptions: {transcriptions_path}\n")

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(data_dir),
        str(transcriptions_path),
        pattern=AUDIO_FILE_PATTERN,
        clips_subdir=CLIPS_SUBDIR,
    )
    
    print(f"Loaded {len(ds)} samples\n")

    run_whisper_baseline(
        loader, 
        ds, 
        model_name="small", 
        language="hi",
        verbose=True,
    )


if __name__ == "__main__":
    main()
