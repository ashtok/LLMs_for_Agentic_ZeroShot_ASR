from __future__ import annotations

from pathlib import Path
import librosa 
from typing import Any, Dict, List
import sys
import numpy as np
import torch
from jiwer import wer, cer
from transformers import AutoProcessor, Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
from torchaudio.models.decoder import ctc_decoder

from config import (
    DATA_ROOT, LANGUAGE, TRANSCRIPTIONS_FILE,
    WORDS_FILE, LEXICON_FILE,
    ASR_SAMPLING_RATE, AUDIO_FILE_PATTERN, CLIPS_SUBDIR,
)

from audio_loader import HFAudioLoader

MODEL_ID = "mms-meta/mms-zeroshot-300m"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def validate_lexicon(lexicon_path: Path, token_file: Path, verbose: bool = True) -> Path:
    """Filter lexicon to only valid MMS character-level tokens."""
    with open(token_file, 'r', encoding='utf-8') as f_tok:
        tokens_set = set(line.strip() for line in f_tok)
    
    if verbose:
        print(f"[LEXICON VALIDATION] MMS vocabulary size: {len(tokens_set)}")
        print(f"[LEXICON VALIDATION] Sample tokens: {list(tokens_set)[:20]}")
    
    valid_entries = []
    bad_entries = []
    total_entries = 0
    
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_entries += 1
            parts = line.strip().split(maxsplit=1)
            
            if len(parts) != 2:
                bad_entries.append((line.strip(), "malformed line"))
                continue
            
            dev_word, uroman_spell = parts
            uroman_clean = uroman_spell.rstrip('|').strip()
            spell_tokens = uroman_clean.split()
            
            invalid_tokens = [tok for tok in spell_tokens if tok not in tokens_set]
            
            if not invalid_tokens:
                valid_entries.append(line)
            else:
                if len(bad_entries) < 10:
                    bad_entries.append((dev_word, f"invalid: {', '.join(invalid_tokens)}"))
    
    if verbose:
        print(f"[LEXICON VALIDATION] Total entries: {total_entries}")
        print(f"[LEXICON VALIDATION] Valid entries: {len(valid_entries)}")
        print(f"[LEXICON VALIDATION] Invalid entries: {total_entries - len(valid_entries)}")
        
        if bad_entries:
            print(f"\n❌ Sample of bad entries (showing first {len(bad_entries)}):")
            for item, reason in bad_entries:
                print(f"  {item} → {reason}")
            print()
    
    if len(valid_entries) == 0:
        raise ValueError(
            "No valid lexicon entries! Lexicon format must be: word c h a r s |\n"
            "Example: мудде m u m d d e |\n"
            "Run collect_words.py with updated character-level tokenization."
        )
    
    validated_path = lexicon_path.with_stem(lexicon_path.stem + '.validated')
    with open(validated_path, 'w', encoding='utf-8') as f:
        f.writelines(valid_entries)
    
    if verbose:
        print(f"✅ Validated lexicon saved to: {validated_path}")
        print(f"✅ Filtered: {len(valid_entries)}/{total_entries} valid entries\n")
    
    return validated_path


def run_mms_zeroshot_constrained(
    loader: HFAudioLoader,
    ds: Any,
    lexicon_path: Path,
    model_id: str = MODEL_ID,
    verbose: bool = True,
) -> Dict:
    """MMS Zero-shot with lexicon-constrained CTC decoding (Meta method)."""
    if verbose:
        print("[MMS-ZS-CONSTRAINED] Loading model, processor, tokens...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    token_file = hf_hub_download(repo_id=model_id, filename="tokens.txt")
    
    device = _get_device()
    model.to(device)
    
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon missing: {lexicon_path}\nRun: python main/collect_words.py")
    
    validated_lexicon = validate_lexicon(lexicon_path, Path(token_file), verbose=verbose)
    
    if verbose:
        print(f"[MMS-ZS-CONSTRAINED] Using validated lexicon: {validated_lexicon}")
        print(f"Using device: {device}\n")
    
    # Meta decoder parameters
    WORD_SCORE = -0.18
    LM_WEIGHT = 0
    
    # Create decoder ONCE
    decoder = ctc_decoder(
        lexicon=str(validated_lexicon),
        tokens=token_file,
        nbest=1,
        beam_size=500,
        beam_size_token=50,
        word_score=WORD_SCORE,
        lm_weight=LM_WEIGHT,
        sil_score=0,
        blank_token="<s>",
    )
    
    hyps_native: List[str] = []
    refs_dev: List[str] = []
    N = len(ds)
    
    if verbose:
        print(f"[MMS-ZS-CONSTRAINED] Processing {N} audio files...\n")
    
    for i in range(N):
        audio_info = ds[i]["audio"]
        path = audio_info["path"] if isinstance(audio_info, dict) else audio_info
        audio, _ = librosa.load(path, sr=ASR_SAMPLING_RATE, mono=True)
        audio = audio.astype(np.float32)
        
        inputs = processor(audio, sampling_rate=ASR_SAMPLING_RATE, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits.cpu()
        
        result = decoder(logits)
        hyp_native = " ".join(result[0][0].words).strip()
        
        # ✅ Get ref from dataset like Whisper does
        ref = ds[i]["text"]
        
        hyps_native.append(hyp_native)
        refs_dev.append(ref)
        
        if verbose and (i % 100 == 0 or i < 5):
            filename = Path(path).name
            print(f"[MMS-ZS] {i+1}/{N} {filename}")
            print(f"  REF: {ref}")
            print(f"  HYP: {hyp_native}\n")
    
    wer_native = float(wer(refs_dev, hyps_native))
    cer_native = float(cer(refs_dev, hyps_native))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[MMS-ZS-CONSTRAINED] Final Metrics:")
        print(f"  Native WER: {wer_native:.4f}")
        print(f"  Native CER: {cer_native:.4f}")
        print(f"  Samples: {N}")
        print(f"{'='*60}\n")
    
    return {
        "model": f"{model_id} (constrained)",
        "wer_native": wer_native,
        "cer_native": cer_native,
        "n_samples": N,
        "hyps": hyps_native,
        "refs": refs_dev,
    }


def main() -> None:
    import argparse
    
    # ✅ ADD ARGUMENT PARSER
    parser = argparse.ArgumentParser(description="Run MMS Zero-shot Constrained ASR")
    parser.add_argument("--language", default=LANGUAGE, help="Language code")
    parser.add_argument("--model-id", default=MODEL_ID, help="Model ID")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()
    
    data_dir = DATA_ROOT / args.language
    transcriptions_path = data_dir / TRANSCRIPTIONS_FILE
    lexicon_path = data_dir / LEXICON_FILE
    
    for path in [transcriptions_path, lexicon_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}\nRun: python main/collect_words.py")
    
    if not args.quiet:
        print(f"{'='*60}")
        print(f"MMS Zero-Shot Constrained ASR Evaluation")
        print(f"{'='*60}")
        print(f"Transcriptions: {transcriptions_path}")
        print(f"Lexicon: {lexicon_path}")
        print(f"Language: {args.language}\n")
    
    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(data_dir), str(transcriptions_path),
        pattern=AUDIO_FILE_PATTERN, clips_subdir=CLIPS_SUBDIR,
    )
    
    # ✅ LIMIT SAMPLES IF REQUESTED
    if args.max_samples:
        end_idx = args.start_idx + args.max_samples
        ds = ds.select(range(args.start_idx, min(end_idx, len(ds))))
    
    if not args.quiet:
        print(f"Loaded {len(ds)} audio files\n")
    
    result = run_mms_zeroshot_constrained(
        loader, ds, 
        lexicon_path=lexicon_path, 
        model_id=args.model_id,
        verbose=not args.quiet
    )
    
    if not args.quiet:
        print("\n✅ Evaluation Complete!")
        print(f"Final Results: {result}")


if __name__ == "__main__":
    main()
