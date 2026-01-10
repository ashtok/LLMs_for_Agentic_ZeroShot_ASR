from pathlib import Path
from typing import List, Dict

import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
from jiwer import wer, cer

from audio_loader import HFAudioLoader


def run_mms_baseline(
    loader: HFAudioLoader,
    ds,
    model_id: str = "facebook/mms-1b-all",
    target_lang: str = "hin",
    verbose: bool = True,   # NEW
) -> Dict[str, float]:
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
    base_dir = Path(__file__).resolve().parent.parent / "data" / "hindi_audio"
    base_dir_str = str(base_dir)

    loader = HFAudioLoader(target_sr=16_000)
    ds = loader.from_dir_with_text(
        base_dir_str,
        str(base_dir / "transcriptions.txt"),
    )
    run_mms_baseline(loader, ds)


if __name__ == "__main__":
    main()
