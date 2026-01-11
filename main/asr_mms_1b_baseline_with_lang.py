from pathlib import Path
from typing import List, Dict
import sys

import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
from jiwer import wer, cer

# Import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT, LANGUAGE, TRANSCRIPTIONS_FILE, ASR_SAMPLING_RATE

from audio_loader import HFAudioLoader


def run_mms_baseline(
    loader: HFAudioLoader,
    ds,
    model_id: str = "facebook/mms-1b-all",
    target_lang: str = "hin",
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
    # Use config paths
    data_dir = DATA_ROOT / LANGUAGE
    transcriptions_path = data_dir / TRANSCRIPTIONS_FILE
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not transcriptions_path.exists():
        raise FileNotFoundError(f"Transcriptions file not found: {transcriptions_path}")
    
    print(f"Loading data from: {data_dir}")
    print(f"Using transcriptions: {transcriptions_path}")
    
    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(data_dir),
        str(transcriptions_path),
    )
    
    print(f"Loaded {len(ds)} samples\n")
    
    run_mms_baseline(loader, ds, target_lang="hin", verbose=True)


if __name__ == "__main__":
    main()
