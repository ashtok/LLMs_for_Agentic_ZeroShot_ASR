from pathlib import Path
from typing import Dict, List, Any

from jiwer import wer, cer
from audio_loader import HFAudioLoader

# NOTE: do NOT import ASRInferencePipeline at module level
# from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # remove this


def _get_omni_pipeline(model_card: str):
    """
    Lazy-import the Omni pipeline so that importing this module
    does not immediately trigger fairseq2/fairseq2n/TBB loading.
    """
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # type: ignore
    return ASRInferencePipeline(model_card=model_card)


def run_omni_baseline(
    loader: HFAudioLoader,
    ds: Any,
    model_card: str = "omniASR_CTC_300M",
    lang_tag: str = "hin_Deva",
) -> Dict[str, float]:
    """
    Run Omnilingual ASR baseline on a given dataset (loader + ds) and return metrics.
    """
    pipeline = _get_omni_pipeline(model_card=model_card)

    refs: List[str] = []
    hyps: List[str] = []

    N = len(ds)
    for i in range(N):
        audio_info = ds[i]["audio"]
        path = audio_info["path"] if isinstance(audio_info, dict) else audio_info

        transcripts = pipeline.transcribe(
            [path],
            lang=[lang_tag],
            batch_size=1,
        )

        hyp = transcripts[0]
        ref = ds[i]["text"]

        refs.append(ref)
        hyps.append(hyp)

        print(str(path))
        print(f"REF: {ref}")
        print(f"HYP_OMNI: {hyp}\n")

    wer_val = float(wer(refs, hyps))
    cer_val = float(cer(refs, hyps))

    print(f"[OmniASR] WER: {wer_val}")
    print(f"[OmniASR] CER: {cer_val}")

    return {
        "model": f"{model_card} ({lang_tag})",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "data" / "hindi_audio"

    loader = HFAudioLoader(target_sr=16_000)
    ds = loader.from_dir_with_text(
        str(base_dir),
        str(base_dir / "transcriptions.txt"),
    )

    run_omni_baseline(
        loader,
        ds,
        model_card="omniASR_CTC_300M", #todo encoder-decoder model
        lang_tag="hin_Deva",
    )


if __name__ == "__main__":
    main()
