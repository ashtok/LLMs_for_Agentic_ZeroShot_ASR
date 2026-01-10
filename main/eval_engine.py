from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from audio_loader import HFAudioLoader
from asr_whisper_baseline import run_whisper_baseline
from asr_mms_1b_baseline_with_lang import run_mms_baseline
from asr_mms_zeroshot_baseline import (
    run_mms_zeroshot_baseline_basic,
    ASR_SAMPLING_RATE,
)


def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    backend = config["backend"]
    model_name = config["model_name"]
    data_root = Path(config["data_root"])
    transcription_file = config["transcription_file"]
    target_lang = config.get("target_lang")
    language = config.get("language", "hi")
    quiet = config.get("quiet", False)

    # NEW: optional limits
    max_samples = config.get("max_samples")     # e.g. 1 or 5
    start_idx = config.get("start_idx", 0)      # default 0

    base_dir_abs = str(data_root)
    trans_path = str(data_root / transcription_file)

    loader = HFAudioLoader(target_sr=16_000)
    ds = loader.from_dir_with_text(base_dir_abs, trans_path)

    # Slice dataset if requested
    if max_samples is not None:
        end_idx = start_idx + max_samples
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))

    if backend == "whisper":
        result = run_whisper_baseline(
            loader=loader,
            ds=ds,
            model_name=model_name,
            language=language,
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
        )

    elif backend == "mms_zeroshot":
        # Romanized refs for full dataset
        roman_path = data_root / "transcriptions_uroman.txt"
        with roman_path.open("r", encoding="utf-8") as f:
            refs_roman_all = [ln.rstrip("\n") for ln in f]

        # Slice refs_roman to match ds
        if max_samples is not None:
            end_idx = start_idx + max_samples
            refs_roman = refs_roman_all[start_idx:end_idx]
        else:
            refs_roman = refs_roman_all

        loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
        ds_full = loader.from_dir_with_text(
            str(data_root),
            str(data_root / "transcriptions.txt"),
        )

        # Slice ds_full the same way
        if max_samples is not None:
            ds_full = ds_full.select(range(start_idx, min(end_idx, len(ds_full))))

        result = run_mms_zeroshot_baseline_basic(
            loader=loader,
            ds=ds_full,
            refs_roman=refs_roman,
            verbose=not quiet,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    result["data_root"] = str(data_root)
    result["transcription_file"] = transcription_file
    return result
