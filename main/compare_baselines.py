from __future__ import annotations

from pathlib import Path
import json
import sys
from datetime import datetime

# Import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT,
    LANGUAGE,
    TRANSCRIPTIONS_FILE,
    TRANSCRIPTIONS_UROMAN_FILE,
    BASELINES_RESULTS_DIR,
    MMS_MODEL_ID,
    MMS_TARGET_LANG,
    MMS_ZEROSHOT_MODEL_ID,
    OMNI_MODEL_CARD,
    OMNI_LANG_TAG,
    WHISPER_MODEL_NAME,
    WHISPER_LANG_CODE,
)

from eval_engine import evaluate_model


def main():
    """Run all baseline evaluations and save results"""
    
    data_root = DATA_ROOT / LANGUAGE
    
    # Check if data exists
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    print(f"\n{'='*80}")
    print("RUNNING ALL BASELINE EVALUATIONS")
    print(f"{'='*80}")
    print(f"Data root: {data_root}")
    print(f"Language: {LANGUAGE}")
    print(f"Results will be saved to: {BASELINES_RESULTS_DIR}")
    print(f"{'='*80}\n")
    
    # Configuration for each model
    whisper_cfg = {
        "backend": "whisper",
        "model_name": WHISPER_MODEL_NAME,
        "whisper_lang": WHISPER_LANG_CODE,
        "language": LANGUAGE,
        "data_root": str(data_root),
        "transcription_file": TRANSCRIPTIONS_FILE,
        "quiet": False,
    }
    
    mms_cfg = {
        "backend": "mms",
        "model_name": MMS_MODEL_ID,
        "target_lang": MMS_TARGET_LANG,
        "language": LANGUAGE,
        "data_root": str(data_root),
        "transcription_file": TRANSCRIPTIONS_FILE,
        "quiet": False,
    }
    
    omni_cfg = {
        "backend": "omni",
        "model_name": OMNI_MODEL_CARD,
        "lang_tag": OMNI_LANG_TAG,
        "language": LANGUAGE,
        "data_root": str(data_root),
        "transcription_file": TRANSCRIPTIONS_FILE,
        "quiet": False,
    }
    
    # MMS zero-shot baseline (uroman WER/CER)
    mms_zs_cfg = {
        "backend": "mms_zeroshot",
        "model_name": MMS_ZEROSHOT_MODEL_ID,
        "language": LANGUAGE,
        "data_root": str(data_root),
        "transcription_file": TRANSCRIPTIONS_FILE,  # Uses both regular and uroman
        "quiet": False,
    }
    
    # Evaluate all models
    results = {}
    evaluations = [
        ("whisper", whisper_cfg),
        ("mms", mms_cfg),
        ("omni", omni_cfg),
        ("mms_zeroshot", mms_zs_cfg),
    ]
    
    for model_key, config in evaluations:
        print(f"\n{'#'*80}")
        print(f"# Evaluating: {model_key.upper()}")
        print(f"{'#'*80}\n")
        
        try:
            result = evaluate_model(config)
            results[model_key] = result
            
            print(f"\n{'='*60}")
            print(f"✔ {model_key.upper()} COMPLETED")
            print(f"{'='*60}")
            print(f"Model: {result['model']}")
            print(f"WER: {result['wer']:.4f}")
            print(f"CER: {result['cer']:.4f}")
            print(f"Samples: {result['n_samples']}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"✗ {model_key.upper()} FAILED")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"{'='*60}\n")
            results[model_key] = {
                "error": str(e),
                "status": "failed"
            }
    
    # Add metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "language": LANGUAGE,
        "data_root": str(data_root),
        "transcription_file": TRANSCRIPTIONS_FILE,
    }
    
    output = {
        "metadata": metadata,
        "results": results,
    }
    
    # Save results
    BASELINES_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASELINES_RESULTS_DIR / f"{LANGUAGE}_baselines.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("ALL EVALUATIONS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {out_path}")
    print(f"\nSUMMARY:")
    print(f"{'-'*80}")
    print(f"{'Model':<30} {'Status':<15} {'WER':<10} {'CER':<10}")
    print(f"{'-'*80}")
    
    for model_key, result in results.items():
        if "error" in result:
            print(f"{model_key:<30} {'FAILED':<15} {'-':<10} {'-':<10}")
        else:
            status = "SUCCESS"
            wer = f"{result['wer']:.4f}"
            cer = f"{result['cer']:.4f}"
            print(f"{model_key:<30} {status:<15} {wer:<10} {cer:<10}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
