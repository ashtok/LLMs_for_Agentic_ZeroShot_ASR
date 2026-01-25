"""
test_all_models.py - Quick test script to verify all ASR models work correctly
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT,
    LANGUAGE,
    TRANSCRIPTIONS_FILE,
    LEXICON_FILE,
    ASR_SAMPLING_RATE,
    AUDIO_FILE_PATTERN,
    CLIPS_SUBDIR,
    WHISPER_MODEL_NAME,
    WHISPER_LANG_CODE,
    MMS_MODEL_ID,
    MMS_TARGET_LANG,
    MMS_ZEROSHOT_MODEL_ID,
    OMNI_MODEL_CARD,
    OMNI_LANG_TAG,
)

from audio_loader import HFAudioLoader


def test_whisper(loader, ds, verbose=True):
    """Test Whisper model"""
    print(f"\n{'='*80}")
    print("TESTING: Whisper")
    print(f"{'='*80}")
    
    from asr_whisper_baseline import run_whisper_baseline
    
    try:
        result = run_whisper_baseline(
            loader=loader,
            ds=ds,
            model_name=WHISPER_MODEL_NAME,
            language=WHISPER_LANG_CODE,
            verbose=verbose,
        )
        print(f"✅ SUCCESS - WER: {result['wer']:.4f}, CER: {result['cer']:.4f}")
        return result
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mms(loader, ds, verbose=True):
    """Test MMS 1B model"""
    print(f"\n{'='*80}")
    print("TESTING: MMS 1B")
    print(f"{'='*80}")
    
    from asr_mms_1b_baseline_with_lang import run_mms_baseline
    
    try:
        result = run_mms_baseline(
            loader=loader,
            ds=ds,
            model_id=MMS_MODEL_ID,
            target_lang=MMS_TARGET_LANG,
            verbose=verbose,
        )
        print(f"✅ SUCCESS - WER: {result['wer']:.4f}, CER: {result['cer']:.4f}")
        return result
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mms_zeroshot_constrained(loader, ds, data_dir, verbose=True):
    """Test MMS Zero-shot Constrained"""
    print(f"\n{'='*80}")
    print("TESTING: MMS Zero-shot Constrained")
    print(f"{'='*80}")
    
    from asr_mms_zeroshot import run_mms_zeroshot_constrained
    
    try:
        lexicon_path = data_dir / LEXICON_FILE
        if not lexicon_path.exists():
            print(f"⚠️  Lexicon not found: {lexicon_path}")
            print("Run: python main/collect_words.py")
            return None
        
        result = run_mms_zeroshot_constrained(
            loader=loader,
            ds=ds,
            lexicon_path=lexicon_path,
            model_id=MMS_ZEROSHOT_MODEL_ID,
            verbose=verbose,
        )
        print(f"✅ SUCCESS - WER: {result['wer_native']:.4f}, CER: {result['cer_native']:.4f}")
        return result
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_omni(loader, ds, verbose=True):
    """Test OmniASR"""
    print(f"\n{'='*80}")
    print("TESTING: OmniASR")
    print(f"{'='*80}")
    
    try:
        from asr_omni_baseline import run_omni_baseline
        
        result = run_omni_baseline(
            loader=loader,
            ds=ds,
            model_card=OMNI_MODEL_CARD,
            lang_tag=OMNI_LANG_TAG,
            verbose=verbose,
        )
        print(f"✅ SUCCESS - WER: {result.get('wer', result.get('wer_native', 'N/A')):.4f}")
        return result
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test all ASR models on sample data")
    parser.add_argument("--max-samples", type=int, default=25, help="Number of samples to test")
    parser.add_argument("--language", default=LANGUAGE, help="Language code")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print(f"# ASR MODELS TEST SUITE")
    print(f"#")
    print(f"# Testing all models on {args.max_samples} samples")
    print(f"# Language: {args.language}")
    print(f"#")
    print(f"{'#'*80}\n")
    
    # Setup paths
    data_dir = DATA_ROOT / args.language
    transcriptions_path = data_dir / TRANSCRIPTIONS_FILE
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not transcriptions_path.exists():
        raise FileNotFoundError(f"Transcriptions file not found: {transcriptions_path}")
    
    print(f"Data directory: {data_dir}")
    print(f"Transcriptions: {transcriptions_path}\n")
    
    # Load dataset
    print("Loading dataset...")
    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(data_dir),
        str(transcriptions_path),
        pattern=AUDIO_FILE_PATTERN,
        clips_subdir=CLIPS_SUBDIR,
    )
    
    # Limit samples
    ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"✅ Loaded {len(ds)} samples\n")
    
    # Test all models
    results = {}
    verbose = not args.quiet
    
    results['whisper'] = test_whisper(loader, ds, verbose)
    results['mms'] = test_mms(loader, ds, verbose)
    results['mms_zs_constrained'] = test_mms_zeroshot_constrained(loader, ds, data_dir, verbose)
    results['omni'] = test_omni(loader, ds, verbose)
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"# TEST SUMMARY")
    print(f"#")
    print(f"# Tested {args.max_samples} samples on {args.language}")
    print(f"#")
    print(f"{'#'*80}\n")
    
    for model_name, result in results.items():
        if result is None:
            print(f"❌ {model_name.upper()}: FAILED")
        else:
            wer_key = 'wer_native' if 'wer_native' in result else 'wer'
            cer_key = 'cer_native' if 'cer_native' in result else 'cer'
            wer_val = result.get(wer_key, 'N/A')
            cer_val = result.get(cer_key, 'N/A')
            
            if isinstance(wer_val, float):
                print(f"✅ {model_name.upper()}: WER={wer_val:.4f}, CER={cer_val:.4f}")
            else:
                print(f"✅ {model_name.upper()}: WER={wer_val}, CER={cer_val}")
    
    print(f"\n{'#'*80}\n")
    
    # Check if all passed
    failed = [k for k, v in results.items() if v is None]
    if failed:
        print(f"⚠️  {len(failed)} model(s) failed: {', '.join(failed)}")
        print("\nPlease fix the errors before running the full agent.\n")
        sys.exit(1)
    else:
        print("✅ All models passed! Ready to run the full agent.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
