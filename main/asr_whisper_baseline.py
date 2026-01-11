from pathlib import Path
from typing import Dict, List, Any
import sys

import whisper
import torchaudio
import numpy as np
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
    Uses torchaudio to load files instead of ffmpeg.
    
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
        print(f"[OpenAI Whisper] Language: {language}")
        print(f"[OpenAI Whisper] Using torchaudio (no ffmpeg required)\n")
    
    model = whisper.load_model(model_name)

    refs: List[str] = []
    hyps: List[str] = []

    N = len(ds)
    for i in range(N):
        audio_info = ds[i]["audio"]
        path = audio_info["path"] if isinstance(audio_info, dict) else audio_info

        # Load audio with torchaudio instead of letting Whisper use ffmpeg
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Convert to numpy array and flatten
        audio_array = waveform.squeeze(0).numpy()

        # Pass numpy array directly to Whisper (bypasses ffmpeg)
        result = model.transcribe(
            audio_array,
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OpenAI Whisper baseline evaluation")
    parser.add_argument("--language", default=LANGUAGE, help="Language code for dataset")
    parser.add_argument("--model-name", default="small", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size")
    parser.add_argument("--whisper-lang", default="hi", help="Language code for Whisper")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--transcriptions", default=TRANSCRIPTIONS_FILE, help="Transcriptions filename")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()
    
    # Use config paths
    data_dir = DATA_ROOT / args.language
    transcriptions_path = data_dir / args.transcriptions
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not transcriptions_path.exists():
        raise FileNotFoundError(f"Transcriptions file not found: {transcriptions_path}")
    
    if not args.quiet:
        print(f"Loading data from: {data_dir}")
        print(f"Using transcriptions: {transcriptions_path}\n")

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(data_dir),
        str(transcriptions_path),
        pattern=AUDIO_FILE_PATTERN,
        clips_subdir=CLIPS_SUBDIR,
    )
    
    # Limit samples if requested
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    
    if not args.quiet:
        print(f"Loaded {len(ds)} samples\n")

    results = run_whisper_baseline(
        loader, 
        ds, 
        model_name=args.model_name, 
        language=args.whisper_lang,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
