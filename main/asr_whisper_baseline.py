from pathlib import Path
from typing import Dict, List

import whisper
from jiwer import wer, cer

from audio_loader import HFAudioLoader


def run_whisper_baseline(
    loader: HFAudioLoader,
    ds,
    model_name: str = "small",
    language: str = "hi",
    verbose: bool = True,   # NEW
) -> Dict[str, float]:
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

        hyp = result["text"]
        ref = ds[i]["text"]

        refs.append(ref)
        hyps.append(hyp)

        if verbose:  # NEW
            print(f"[OpenAI Whisper] Sample {i}/{N}")
            print(f"{path}")
            print(f"REF: {ref}")
            print(f"HYP: {hyp}\n")

    wer_val = float(wer(refs, hyps))
    cer_val = float(cer(refs, hyps))

    if verbose:  # NEW
        print("WER:", wer_val)
        print("CER:", cer_val)

    return {
        "model": f"whisper-{model_name}",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
        "hyps": hyps,
        "refs": refs,
    }


def main():
    # Keep a script entry point for quick manual runs
    base_dir = Path(__file__).resolve().parent.parent / "data" / "hindi_audio"
    base_dir_abs = str(base_dir)

    loader = HFAudioLoader(target_sr=16_000)
    ds = loader.from_dir_with_text(
        base_dir_abs,
        str(base_dir / "transcriptions.txt"),
    )
    run_whisper_baseline(loader, ds, model_name="small", language="hi")


if __name__ == "__main__":
    main()
