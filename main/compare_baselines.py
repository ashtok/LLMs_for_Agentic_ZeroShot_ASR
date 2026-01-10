from __future__ import annotations

from pathlib import Path
import json

from eval_engine import evaluate_model


def main():
    data_root = Path(__file__).resolve().parent.parent / "data" / "hindi_audio"

    whisper_cfg = {
        "backend": "whisper",
        "model_name": "small",
        "language": "hi",
        "data_root": str(data_root),
        "transcription_file": "transcriptions.txt",
    }

    mms_cfg = {
        "backend": "mms",
        "model_name": "facebook/mms-1b-all",
        "target_lang": "hin",
        "data_root": str(data_root),
        "transcription_file": "transcriptions.txt",
    }

    omni_cfg = {
        "backend": "omni",
        "model_name": "omniASR-CTC-300M",
        "lang_tag": "hin_Deva",
        "data_root": str(data_root),
        "transcription_file": "transcriptions.txt",
    }

    # NEW: MMS zero-shot baseline (uroman WER/CER)
    mms_zs_cfg = {
        "backend": "mms_zeroshot",
        "model_name": "mms-meta/mms-zeroshot-300m",
        # uroman references precomputed from transcriptions.txt
        "data_root": str(data_root),
        "transcription_file": "transcriptions_uroman.txt",
    }

    # Evaluate all four
    whisper_res = evaluate_model(whisper_cfg)
    mms_res = evaluate_model(mms_cfg)
    #omni_res = evaluate_model(omni_cfg)
    mms_zs_res = evaluate_model(mms_zs_cfg)

    # Collect results
    results = {
        "whisper_small": whisper_res,
        "mms_1b_all_hin": mms_res,
        #"omniASR-CTC-300M_hin_Deva": omni_res,
        "mms_zeroshot_300m_uroman": mms_zs_res,
    }

    results_dir = Path(__file__).resolve().parent.parent / "results" / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "hindi_baselines.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
