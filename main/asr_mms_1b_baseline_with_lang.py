from pathlib import Path
from typing import List, Dict
import sys

import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
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
    MMS_MODEL_ID,
    MMS_TARGET_LANG,
)

from audio_loader import HFAudioLoader


def run_mms_baseline(
    loader: HFAudioLoader,
    ds,
    model_id: str = "facebook/mms-1b-all",
    target_lang: str = "bul",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run MMS baseline evaluation on audio dataset
    
    Args:
        loader: Audio loader instance
        ds: Dataset loaded by audio_loader
        model_id: HuggingFace model ID
        target_lang: Target language code for MMS
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with WER, CER, and other metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Using device:", device)

    processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id,
        target_lang=target_lang,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    blank_id = processor.tokenizer.pad_token_id
    if verbose:
        print("Blank / pad token id:", blank_id)

    refs: List[str] = []
    hyps: List[str] = []

    N = len(ds)
    for i in range(N):
        waveform, sr, path = loader.get_example(i)

        inputs = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        pred_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(
            pred_ids.cpu().numpy(),
            skip_special_tokens=True,
        )[0]

        ref = ds[i]["text"]

        refs.append(ref)
        hyps.append(hyp)

        if verbose:
            print(f"[MMS-Lang-Adapter] Sample {i}/{N}")
            print(str(path))
            print(f"REF: {ref}")
            print(f"HYP_RAW: {hyp}\n")

    wer_val = float(wer(refs, hyps))
    cer_val = float(cer(refs, hyps))

    if verbose:
        print("WER:", wer_val)
        print("CER:", cer_val)

    return {
        "model": f"{model_id} ({target_lang})",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
        "hyps": hyps,
        "refs": refs,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MMS baseline evaluation")
    parser.add_argument("--language", default=LANGUAGE, help="Language code for dataset")
    parser.add_argument("--model-id", default=MMS_MODEL_ID, help="MMS model ID")
    parser.add_argument("--target-lang", default=MMS_TARGET_LANG, help="Target language code for MMS")
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
    
    results = run_mms_baseline(
        loader,
        ds,
        model_id=args.model_id,
        target_lang=args.target_lang,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
