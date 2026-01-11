from pathlib import Path
from typing import Dict, List, Any
import sys
import tempfile
import shutil

import torchaudio
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
)

from audio_loader import HFAudioLoader
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def convert_to_wav(mp3_path: str, wav_path: str):
    """Convert MP3 to WAV using torchaudio (no ffmpeg required)"""
    # Load MP3
    waveform, sr = torchaudio.load(mp3_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # Save as WAV
    torchaudio.save(wav_path, waveform, 16000)


def run_omni_baseline(
    loader: HFAudioLoader,
    ds: Any,
    model_card: str = "omniASR_CTC_300M",
    lang_tag: str = "hin_Deva",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run Omnilingual ASR baseline on a given dataset and return metrics.
    Converts MP3 to WAV on-the-fly using torchaudio (no ffmpeg required).
    
    Args:
        loader: Audio loader instance
        ds: Dataset loaded by audio_loader
        model_card: Model card name (e.g., "omniASR_CTC_300M")
        lang_tag: Language tag (e.g., "hin_Deva" for Hindi Devanagari)
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with WER, CER, and other metrics
    """
    if verbose:
        print(f"[OmniASR] Loading model: {model_card}")
        print(f"[OmniASR] Language tag: {lang_tag}")
        print(f"[OmniASR] Note: Converting MP3 to WAV on-the-fly (torchaudio)\n")
    
    pipeline = ASRInferencePipeline(model_card=model_card)

    refs: List[str] = []
    hyps: List[str] = []

    N = len(ds)
    
    # Create temp directory for WAV files
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i in range(N):
            audio_info = ds[i]["audio"]
            path = audio_info["path"] if isinstance(audio_info, dict) else audio_info

            # Convert MP3 to WAV using torchaudio
            wav_path = f"{temp_dir}/audio_{i}.wav"
            convert_to_wav(path, wav_path)

            # Transcribe WAV file
            transcripts = pipeline.transcribe(
                [wav_path],
                lang=[lang_tag],
                batch_size=1,
            )

            hyp = transcripts[0]
            ref = ds[i]["text"]

            refs.append(ref)
            hyps.append(hyp)

            if verbose:
                filename = Path(path).name
                print(f"[OmniASR] Sample {i+1}/{N}")
                print(f"FILE: {filename}")
                print(f"REF: {ref}")
                print(f"HYP_OMNI: {hyp}\n")
            
            # Clean up WAV file immediately to save space
            Path(wav_path).unlink()

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    wer_val = float(wer(refs, hyps))
    cer_val = float(cer(refs, hyps))

    if verbose:
        print(f"[OmniASR] WER: {wer_val:.4f}")
        print(f"[OmniASR] CER: {cer_val:.4f}")

    return {
        "model": f"{model_card} ({lang_tag})",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
        "hyps": hyps,
        "refs": refs,
    }


def main() -> None:
    # Use config paths
    data_dir = DATA_ROOT / LANGUAGE
    transcriptions_path = data_dir / TRANSCRIPTIONS_FILE
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not transcriptions_path.exists():
        raise FileNotFoundError(f"Transcriptions file not found: {transcriptions_path}")
    
    print(f"Loading data from: {data_dir}")
    print(f"Using transcriptions: {transcriptions_path}\n")

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(data_dir),
        str(transcriptions_path),
        pattern=AUDIO_FILE_PATTERN,
        clips_subdir=CLIPS_SUBDIR,
    )
    
    print(f"Loaded {len(ds)} samples\n")

    run_omni_baseline(
        loader,
        ds,
        model_card="omniASR_CTC_300M",
        lang_tag="hin_Deva",
        verbose=True,
    )


if __name__ == "__main__":
    main()
