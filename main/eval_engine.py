from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys
import os

# Import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT,
    LANGUAGE,
    TRANSCRIPTIONS_FILE,
    TRANSCRIPTIONS_UROMAN_FILE,
    LEXICON_FILE,
    ASR_SAMPLING_RATE,
    AUDIO_FILE_PATTERN,
    CLIPS_SUBDIR,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_START_IDX,
    DEFAULT_QUIET,
    MMS_ZEROSHOT_MODEL_ID,
)

from audio_loader import HFAudioLoader
from asr_whisper_baseline import run_whisper_baseline
from asr_mms_1b_baseline_with_lang import run_mms_baseline
from asr_mms_zeroshot_baseline import run_mms_zeroshot_baseline_basic
from asr_mms_zeroshot import run_mms_zeroshot_constrained

def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate ASR models on a given dataset.
    
    Args:
        config: Dictionary containing evaluation configuration
            Required keys:
                - backend: str - Model backend (whisper, mms, omni, mms_zeroshot, mms_zeroshot_constrained)
                - model_name: str - Model identifier
            Optional keys:
                - data_root: Path - Data directory (default: from config)
                - language: str - Language code (default: from config)
                - transcription_file: str - Transcription filename (default: from config)
                - target_lang: str - Target language for MMS
                - lang_tag: str - Language tag for OmniASR
                - lexicon_file: str - Lexicon for constrained decoding
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
    
    # 🔇 OPTIMIZATION: Minimal logging for batch operations
    if not quiet and max_samples != 1:
        print(f"\n{'='*60}")
        print(f"Evaluating: {backend} - {model_name}")
        print(f"{'='*60}")
        print(f"Data root: {data_root}")
        print(f"Transcription file: {transcription_file}")
        if max_samples:
            print(f"Sample range: {start_idx} to {start_idx + max_samples}")
        print(f"{'='*60}\n")
    
    # 🔇 OPTIMIZATION: Suppress dataset loading message for single files
    original_stdout = sys.stdout
    if quiet or max_samples == 1:
        sys.stdout = open(os.devnull, 'w')
    
    # Load dataset
    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(data_root),
        str(trans_path),
        pattern=AUDIO_FILE_PATTERN,
        clips_subdir=CLIPS_SUBDIR,
    )
    
    # Restore stdout
    if quiet or max_samples == 1:
        sys.stdout.close()
        sys.stdout = original_stdout
    
    # Slice dataset if requested
    if max_samples is not None:
        end_idx = start_idx + max_samples
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
        if not quiet and max_samples != 1:
            print(f"Selected {len(ds)} samples from dataset\n")
    
    # 🔇 OPTIMIZATION: Suppress model execution for single-file batch calls
    if quiet or max_samples == 1:
        sys.stdout = open(os.devnull, 'w')
    
    # Run evaluation based on backend
    try:
        if backend == "whisper":
            result = run_whisper_baseline(
                loader=loader,
                ds=ds,
                model_name=model_name,
                language=config.get("whisper_lang", language if language == "hi" else "en"),
                verbose=False,
            )
        
        elif backend == "mms":
            result = run_mms_baseline(
                loader=loader,
                ds=ds,
                model_id=model_name,
                target_lang=target_lang or "hin",
                verbose=False,
            )
        
        elif backend == "omni":
            from asr_omni_baseline import run_omni_baseline
            
            result = run_omni_baseline(
                loader=loader,
                ds=ds,
                model_card=model_name,
                lang_tag=config.get("lang_tag", "hin_Deva"),
                verbose=False,
            )
        
        elif backend == "mms_zeroshot":
            roman_path = data_root / TRANSCRIPTIONS_UROMAN_FILE
            
            if not roman_path.exists():
                raise FileNotFoundError(f"Romanized transcriptions not found: {roman_path}")
            
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
            
            refs_roman = []
            for i in range(len(ds)):
                audio_info = ds[i]["audio"]
                path = audio_info["path"] if isinstance(audio_info, dict) else audio_info
                filename = Path(path).name
                
                if filename in refs_roman_map:
                    refs_roman.append(refs_roman_map[filename])
                else:
                    refs_roman.append("")
            
            result = run_mms_zeroshot_baseline_basic(
                loader=loader,
                ds=ds,
                refs_roman=refs_roman,
                model_id=config.get("model_id", MMS_ZEROSHOT_MODEL_ID),
                verbose=False,
            )
        
        elif backend == "mms_zeroshot_constrained":  # ✅ CLEAN VERSION
            result = run_mms_zeroshot_constrained(
                loader=loader,
                ds=ds,
                lexicon_path=data_root / config.get("lexicon_file", LEXICON_FILE),
                model_id=config.get("model_id", MMS_ZEROSHOT_MODEL_ID),
                verbose=False,
            )
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    finally:
        if quiet or max_samples == 1:
            sys.stdout.close()
            sys.stdout = original_stdout
    
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
    
    config_whisper = {
        "backend": "whisper",
        "model_name": "small",
        "max_samples": 10,
        "quiet": False,
    }
    
    config_mms = {
        "backend": "mms",
        "model_name": "facebook/mms-1b-all",
        "target_lang": "hin",
        "max_samples": 10,
        "quiet": False,
    }
    
    config_mms_zs = {
        "backend": "mms_zeroshot",
        "model_name": "mms-meta/mms-zeroshot-300m",
        "max_samples": 10,
        "quiet": False,
    }
    
    config_mms_zs_constrained = {
        "backend": "mms_zeroshot_constrained",
        "model_name": "mms-meta/mms-zeroshot-300m",
        "lexicon_file": "lexicon.txt",
        "max_samples": 10,
        "quiet": False,
    }
    
    config_omni = {
        "backend": "omni",
        "model_name": "omniASR_CTC_300M",
        "lang_tag": "hin_Deva",
        "max_samples": 10,
        "quiet": False,
    }
    
    print("Choose evaluation to run:")
    print("1. Whisper")
    print("2. MMS")
    print("3. MMS Zero-shot (greedy)")
    print("4. MMS Zero-shot Constrained (lexicon)")
    print("5. OmniASR")
    
    choice = input("Enter choice (1-5): ").strip()
    
    config_map = {
        "1": config_whisper,
        "2": config_mms,
        "3": config_mms_zs,
        "4": config_mms_zs_constrained,
        "5": config_omni,
    }
    
    if choice in config_map:
        result = evaluate_model(config_map[choice])
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Model: {result['model']}")
        print(f"WER: {result.get('wer_native', result.get('wer')):.4f}")
        print(f"CER: {result.get('cer_native', result.get('cer')):.4f}")
        print(f"Samples: {result['n_samples']}")
        print(f"{'='*60}\n")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
