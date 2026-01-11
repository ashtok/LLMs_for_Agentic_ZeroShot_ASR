from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys

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
    DEFAULT_MAX_SAMPLES,
    DEFAULT_START_IDX,
    DEFAULT_QUIET,
)

from audio_loader import HFAudioLoader
from asr_whisper_baseline import run_whisper_baseline
from asr_mms_1b_baseline_with_lang import run_mms_baseline
from asr_mms_zeroshot_baseline import run_mms_zeroshot_baseline_basic


def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate ASR models on a given dataset.
    
    Args:
        config: Dictionary containing evaluation configuration
            Required keys:
                - backend: str - Model backend (whisper, mms, omni, mms_zeroshot)
                - model_name: str - Model identifier
            Optional keys:
                - data_root: Path - Data directory (default: from config)
                - language: str - Language code (default: from config)
                - transcription_file: str - Transcription filename (default: from config)
                - target_lang: str - Target language for MMS
                - lang_tag: str - Language tag for OmniASR
                - max_samples: int - Limit number of samples
                - start_idx: int - Starting index (default: 0)
                - quiet: bool - Reduce verbosity (default: False)
    
    Returns:
        Dictionary with evaluation results (WER, CER, etc.)
    """
    backend = config["backend"]
    model_name = config["model_name"]
    
    # Use config defaults with overrides
    language = config.get("language", LANGUAGE)
    data_root = Path(config.get("data_root", DATA_ROOT / language))
    transcription_file = config.get("transcription_file", TRANSCRIPTIONS_FILE)
    target_lang = config.get("target_lang")
    quiet = config.get("quiet", DEFAULT_QUIET)
    
    # Sample limits
    max_samples = config.get("max_samples", DEFAULT_MAX_SAMPLES)
    start_idx = config.get("start_idx", DEFAULT_START_IDX)
    
    # Paths
    trans_path = data_root / transcription_file
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    if not trans_path.exists():
        raise FileNotFoundError(f"Transcription file not found: {trans_path}")
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Evaluating: {backend} - {model_name}")
        print(f"{'='*60}")
        print(f"Data root: {data_root}")
        print(f"Transcription file: {transcription_file}")
        if max_samples:
            print(f"Sample range: {start_idx} to {start_idx + max_samples}")
        print(f"{'='*60}\n")
    
    # Load dataset
    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(data_root),
        str(trans_path),
        pattern=AUDIO_FILE_PATTERN,
        clips_subdir=CLIPS_SUBDIR,
    )
    
    # Slice dataset if requested
    if max_samples is not None:
        end_idx = start_idx + max_samples
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
        if not quiet:
            print(f"Selected {len(ds)} samples from dataset\n")
    
    # Run evaluation based on backend
    if backend == "whisper":
        result = run_whisper_baseline(
            loader=loader,
            ds=ds,
            model_name=model_name,
            language=config.get("whisper_lang", language if language == "hi" else "en"),
            verbose=not quiet,
        )
    
    elif backend == "mms":
        result = run_mms_baseline(
            loader=loader,
            ds=ds,
            model_id=model_name,
            target_lang=target_lang or "hin",
            verbose=not quiet,
        )
    
    elif backend == "omni":
        from asr_omni_baseline import run_omni_baseline
        
        result = run_omni_baseline(
            loader=loader,
            ds=ds,
            model_card=model_name,
            lang_tag=config.get("lang_tag", "hin_Deva"),
            verbose=not quiet,
        )
    
    elif backend == "mms_zeroshot":
        # Load romanized transcriptions
        roman_path = data_root / TRANSCRIPTIONS_UROMAN_FILE
        
        if not roman_path.exists():
            raise FileNotFoundError(f"Romanized transcriptions not found: {roman_path}")
        
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
        
        # Build refs_roman list aligned with dataset order
        refs_roman = []
        for i in range(len(ds)):
            audio_info = ds[i]["audio"]
            path = audio_info["path"] if isinstance(audio_info, dict) else audio_info
            filename = Path(path).name
            
            if filename in refs_roman_map:
                refs_roman.append(refs_roman_map[filename])
            else:
                refs_roman.append("")
                if not quiet:
                    print(f"Warning: No romanized transcription for {filename}")
        
        result = run_mms_zeroshot_baseline_basic(
            loader=loader,
            ds=ds,
            refs_roman=refs_roman,
            model_id=config.get("model_id", "mms-meta/mms-zeroshot-300m"),
            verbose=not quiet,
        )
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Add metadata to results
    result["backend"] = backend
    result["data_root"] = str(data_root)
    result["transcription_file"] = transcription_file
    result["samples_evaluated"] = len(ds)
    if max_samples:
        result["sample_range"] = f"{start_idx}-{start_idx + len(ds)}"
    
    return result


def main():
    """Example usage of evaluate_model function"""
    
    # Example 1: Evaluate Whisper small model
    config_whisper = {
        "backend": "whisper",
        "model_name": "small",
        "max_samples": 10,  # Evaluate on first 10 samples
        "quiet": False,
    }
    
    # Example 2: Evaluate MMS model
    config_mms = {
        "backend": "mms",
        "model_name": "facebook/mms-1b-all",
        "target_lang": "hin",
        "max_samples": 10,
        "quiet": False,
    }
    
    # Example 3: Evaluate MMS zero-shot
    config_mms_zs = {
        "backend": "mms_zeroshot",
        "model_name": "mms-meta/mms-zeroshot-300m",
        "max_samples": 10,
        "quiet": False,
    }
    
    # Example 4: Evaluate OmniASR
    config_omni = {
        "backend": "omni",
        "model_name": "omniASR_CTC_300M",
        "lang_tag": "hin_Deva",
        "max_samples": 10,
        "quiet": False,
    }
    
    # Choose which config to run
    print("Choose evaluation to run:")
    print("1. Whisper")
    print("2. MMS")
    print("3. MMS Zero-shot")
    print("4. OmniASR")
    
    choice = input("Enter choice (1-4): ").strip()
    
    config_map = {
        "1": config_whisper,
        "2": config_mms,
        "3": config_mms_zs,
        "4": config_omni,
    }
    
    if choice in config_map:
        result = evaluate_model(config_map[choice])
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Model: {result['model']}")
        print(f"WER: {result['wer']:.4f}")
        print(f"CER: {result['cer']:.4f}")
        print(f"Samples: {result['n_samples']}")
        print(f"{'='*60}\n")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
