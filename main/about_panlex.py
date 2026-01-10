from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import pandas as pd
import torch
from jiwer import wer, cer
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audio_loader import HFAudioLoader

ASR_SAMPLING_RATE = 16_000
MODEL_ID = "mms-meta/mms-zeroshot-300m"  # basic zero-shot MMS model


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
    ):
        return torch.device("mps")
    else:
        return torch.device("cpu")


def oov_rate(hyp: str, lexicon: set) -> float:
    """Calculate out-of-vocabulary rate for a hypothesis string."""
    toks = hyp.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if t not in lexicon) / len(toks)


def run_mms_zeroshot_baseline_basic(
        loader: HFAudioLoader,
        ds: Any,
        refs_roman: List[str],
        verbose: bool = True,
) -> Dict[str, float]:
    if verbose:
        print("[MMS-ZS-BASIC] Loading MMS zero-shot model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    device = _get_device()
    model.to(device)
    if verbose:
        print(f"[MMS-ZS-BASIC] Using device: {device}")

    hyps_roman: List[str] = []

    N = len(ds)
    if len(refs_roman) != N:
        raise ValueError(
            f"refs_roman length ({len(refs_roman)}) does not match dataset size ({N}). "
            "Make sure transcriptions_uroman.txt is aligned with transcriptions.txt."
        )

    for i in range(N):
        audio_info = ds[i]["audio"]
        path = audio_info["path"] if isinstance(audio_info, dict) else audio_info

        audio, sr = librosa.load(path, sr=ASR_SAMPLING_RATE, mono=True)
        audio = audio.astype(np.float32)

        inputs = processor(
            audio,
            sampling_rate=ASR_SAMPLING_RATE,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        hyp_roman = processor.batch_decode(pred_ids)[0].strip()
        hyps_roman.append(hyp_roman)

        ref_dev = ds[i]["text"]
        ref_roman = refs_roman[i]

        if verbose:
            print(f"[MMS-ZS-BASIC] Sample {i}/{N}")
            print(f"PATH: {path}")
            print(f"REF_DEV: {ref_dev}")
            print(f"REF_UROMAN: {ref_roman}")
            print(f"HYP_MMS_UROMAN: {hyp_roman}\n")

    wer_val = float(wer(refs_roman, hyps_roman))
    cer_val = float(cer(refs_roman, hyps_roman))

    if verbose:
        print(f"[MMS-ZS-BASIC] WER (uroman): {wer_val}")
        print(f"[MMS-ZS-BASIC] CER (uroman): {cer_val}")

    return {
        "model": f"{MODEL_ID} (zeroshot-greedy, uroman)",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
        "hyps": hyps_roman,
        "refs": refs_roman,
    }


def load_panlex_hindi_lexicon(panlex_path: Path, verbose: bool = True) -> set:
    """Load Hindi vocabulary from PanLex CSV."""
    if verbose:
        print(f"[PANLEX] Loading PanLex from: {panlex_path}")

    df = pd.read_csv(panlex_path, sep=";", engine="python")

    # Filter for Hindi (ISO 639-3 code: hin)
    df_hin = df[df["639-3"] == "hin"]

    if verbose:
        print(f"[PANLEX] Found {len(df_hin)} Hindi entries")
        print(f"[PANLEX] Variant codes: {df_hin['var_code'].unique()}")
        print(f"[PANLEX] Sample entries: {df_hin['vocab'].head(10).tolist()}")

    return set(df_hin["vocab"].tolist())


def analyze_lexicon_coverage(
        hyps: List[str],
        refs: List[str],
        lexicon: set,
        verbose: bool = True,
) -> Dict[str, float]:
    """Analyze OOV rates for hypotheses and references against lexicon."""
    hyp_oov_rates = [oov_rate(h, lexicon) for h in hyps]
    ref_oov_rates = [oov_rate(r, lexicon) for r in refs]

    avg_hyp_oov = sum(hyp_oov_rates) / len(hyp_oov_rates) if hyp_oov_rates else 0.0
    avg_ref_oov = sum(ref_oov_rates) / len(ref_oov_rates) if ref_oov_rates else 0.0

    if verbose:
        print(f"\n[LEXICON] Average OOV rate (hypotheses): {avg_hyp_oov:.2%}")
        print(f"[LEXICON] Average OOV rate (references): {avg_ref_oov:.2%}")

        # Show some examples
        print("\n[LEXICON] Sample OOV analysis:")
        for i in range(min(3, len(hyps))):
            print(f"  Sample {i}:")
            print(f"    HYP: {hyps[i]} (OOV: {hyp_oov_rates[i]:.2%})")
            print(f"    REF: {refs[i]} (OOV: {ref_oov_rates[i]:.2%})")

    return {
        "avg_hyp_oov": avg_hyp_oov,
        "avg_ref_oov": avg_ref_oov,
        "hyp_oov_rates": hyp_oov_rates,
        "ref_oov_rates": ref_oov_rates,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "data" / "hindi_audio"

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)

    # Dataset with Devanagari references
    ds = loader.from_dir_with_text(
        str(base_dir),
        str(base_dir / "transcriptions.txt"),
    )

    # Romanized refs (uroman), aligned line-by-line
    roman_path = base_dir / "transcriptions_uroman.txt"
    with roman_path.open("r", encoding="utf-8") as f:
        refs_roman = [ln.rstrip("\n") for ln in f]

    # Load PanLex Hindi lexicon
    panlex_path = repo_root / "data" / "panlex.csv"
    hindi_lexicon = load_panlex_hindi_lexicon(panlex_path)

    # Run ASR evaluation
    results = run_mms_zeroshot_baseline_basic(
        loader=loader,
        ds=ds,
        refs_roman=refs_roman
    )

    # Analyze lexicon coverage
    lexicon_analysis = analyze_lexicon_coverage(
        hyps=results["hyps"],
        refs=results["refs"],
        lexicon=hindi_lexicon,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Samples: {results['n_samples']}")
    print(f"WER: {results['wer']:.4f}")
    print(f"CER: {results['cer']:.4f}")
    print(f"Avg OOV (hypotheses): {lexicon_analysis['avg_hyp_oov']:.2%}")
    print(f"Avg OOV (references): {lexicon_analysis['avg_ref_oov']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
