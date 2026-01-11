from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import sys

import torch
from jiwer import wer, cer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT,
    LANGUAGE,
    TRANSCRIPTIONS_FILE,
    TRANSCRIPTIONS_UROMAN_FILE,
    QWEN_AGENT_RESULTS_DIR,
    QWEN_MODEL_NAME,
    QWEN_LOAD_8BIT,
    QWEN_MAX_NEW_TOKENS,
    AGENT_MAX_FILES,
    AGENT_START_IDX,
    WHISPER_MODEL_NAME,
    MMS_MODEL_ID,
    MMS_TARGET_LANG,
    MMS_ZEROSHOT_MODEL_ID,
)

from eval_engine import evaluate_model


class QwenASRAgent:
    """
    Agent that:
    - Runs Whisper, MMS-1B, MMS-ZS on the same Hindi audio.
    - Sends all hypotheses + metadata to Qwen2.5.
    - Qwen chooses best transcription and explains.
    - We evaluate that transcription vs ground truth (WER/CER).
    """

    def __init__(
        self,
        model_name: str = QWEN_MODEL_NAME,
        device: str | None = None,
        load_in_8bit: bool = QWEN_LOAD_8BIT,
    ) -> None:
        print(f"Initializing Qwen Hindi ASR Agent with {model_name}...")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Qwen model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if load_in_8bit and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )

        self.model.eval()
        print(f"✓ Qwen model loaded on {self.device}")

    def _run_baselines_for_file(
        self,
        audio_idx: int,
        data_root: Path,
        language: str = LANGUAGE,
    ) -> Dict[str, Any]:
        """
        Run Whisper, MMS-1B, MMS-ZS on a *single* audio index.

        Uses eval_engine with max_samples=1 and start_idx=audio_idx.
        """

        whisper_cfg = {
            "backend": "whisper",
            "model_name": WHISPER_MODEL_NAME,
            "whisper_lang": "hi",
            "language": language,
            "data_root": str(data_root),
            "transcription_file": TRANSCRIPTIONS_FILE,
            "quiet": True,
            "max_samples": 1,       # exactly one file
            "start_idx": audio_idx,
        }

        mms_cfg = {
            "backend": "mms",
            "model_name": MMS_MODEL_ID,
            "target_lang": MMS_TARGET_LANG,
            "language": language,
            "data_root": str(data_root),
            "transcription_file": TRANSCRIPTIONS_FILE,
            "quiet": True,
            "max_samples": 1,
            "start_idx": audio_idx,
        }

        mms_zs_cfg = {
            "backend": "mms_zeroshot",
            "model_name": MMS_ZEROSHOT_MODEL_ID,
            "language": language,
            "data_root": str(data_root),
            "transcription_file": TRANSCRIPTIONS_FILE,  # Uses both regular and uroman internally
            "quiet": True,
            "max_samples": 1,
            "start_idx": audio_idx,
        }

        print(f"  Running Whisper on audio index {audio_idx}...")
        whisper_res = evaluate_model(whisper_cfg)
        print(f"  Running MMS-1B on audio index {audio_idx}...")
        mms_res = evaluate_model(mms_cfg)
        print(f"  Running MMS-ZeroShot on audio index {audio_idx}...")
        mms_zs_res = evaluate_model(mms_zs_cfg)

        # Because max_samples=1, each result has exactly one hyp/ref at index 0
        whisper_hyp = whisper_res["hyps"][0]
        mms_hyp = mms_res["hyps"][0]
        mms_zs_hyp = mms_zs_res["hyps"][0]
        ref_dev = whisper_res["refs"][0]

        return {
            "whisper": whisper_hyp,
            "mms_1b": mms_hyp,
            "mms_zs_uroman": mms_zs_hyp,
            "ref_dev": ref_dev,
        }

    def _build_system_prompt(self) -> str:
        """System prompt for Qwen."""
        return """You are an expert Hindi ASR judge.

You will receive:
- Multiple ASR hypotheses for the SAME Hindi utterance:
  - Whisper (Devanagari script)
  - MMS-1B (Devanagari script)
  - MMS-zeroshot (romanized/uroman)
- The target language is Hindi.

Your task:
1. Carefully read all candidate transcriptions.
2. Decide which hypothesis is the most accurate and fluent Hindi sentence.
3. Optionally combine or lightly correct them if that clearly improves accuracy.
4. Output your final answer in a STRICT format.

Output format (MANDATORY):
LANGUAGE: Hindi
TRANSCRIPTION: <your best transcription in Devanagari script>
CONFIDENCE: <high|medium|low>
REASONING: <1-3 sentences in English explaining which system(s) you trusted and why>
"""

    def _build_user_prompt(
        self,
        audio_name: str,
        hypotheses: Dict[str, str],
    ) -> str:
        lines = [
            f"Audio file: {audio_name}",
            "",
            "Here are the ASR hypotheses:",
        ]

        for src, hyp in hypotheses.items():
            if src == "ref_dev":
                continue
            lines.append(f"- {src}: {hyp}")

        lines.append("")
        lines.append("Please select the best overall transcription and respond in the required OUTPUT format.")
        return "\n".join(lines)

    def _call_qwen(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return response

    def _parse_final_answer(self, response: str) -> Dict[str, Any]:
        result = {
            "raw": response,
            "language": "",
            "transcription": "",
            "confidence": "",
            "reasoning": "",
        }

        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("LANGUAGE:"):
                result["language"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("TRANSCRIPTION:"):
                result["transcription"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                result["confidence"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        return result

    def run_on_file(
        self,
        audio_idx: int,
        data_root: Path,
        language: str = LANGUAGE,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run Qwen agent on a single audio file"""
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing audio index: {audio_idx}")
            print(f"{'='*80}")
        
        baseline_out = self._run_baselines_for_file(audio_idx, data_root, language)
        whisper_hyp = baseline_out["whisper"]
        mms_hyp = baseline_out["mms_1b"]
        mms_zs_hyp = baseline_out["mms_zs_uroman"]
        ref_dev = baseline_out["ref_dev"]

        audio_name = f"{language}_{audio_idx:05d}"

        if verbose:
            print(f"\nBaseline Hypotheses:")
            print(f"  Whisper: {whisper_hyp}")
            print(f"  MMS-1B: {mms_hyp}")
            print(f"  MMS-ZS (uroman): {mms_zs_hyp}")
            print(f"  REF (Ground Truth): {ref_dev}")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            audio_name,
            {
                "whisper": whisper_hyp,
                "mms_1b": mms_hyp,
                "mms_zeroshot": mms_zs_hyp,
            },
        )

        if verbose:
            print(f"\n  Calling Qwen model...")
        
        response = self._call_qwen(system_prompt, user_prompt)

        if verbose:
            print(f"\nQwen Response:")
            print(f"{'-'*60}")
            print(response)
            print(f"{'-'*60}")

        parsed = self._parse_final_answer(response)
        qwen_transcription = parsed["transcription"]

        # Calculate metrics
        wer_val = float(wer([ref_dev], [qwen_transcription]))
        cer_val = float(cer([ref_dev], [qwen_transcription]))

        if verbose:
            print(f"\nEvaluation vs Ground Truth:")
            print(f"  Qwen Transcription: {qwen_transcription}")
            print(f"  Ground Truth: {ref_dev}")
            print(f"  WER: {wer_val:.4f}")
            print(f"  CER: {cer_val:.4f}")
            print(f"  Confidence: {parsed['confidence']}")
            print(f"  Reasoning: {parsed['reasoning']}")

        return {
            "audio_idx": audio_idx,
            "audio_name": audio_name,
            "whisper_hyp": whisper_hyp,
            "mms_1b_hyp": mms_hyp,
            "mms_zs_hyp": mms_zs_hyp,
            "ref_dev": ref_dev,
            "qwen_decision": parsed,
            "wer": wer_val,
            "cer": cer_val,
        }

    def run_on_dataset(
        self,
        data_root: Path,
        max_files: int = AGENT_MAX_FILES,
        start_idx: int = AGENT_START_IDX,
        language: str = LANGUAGE,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run Qwen agent on multiple audio files"""
        
        print(f"\n{'='*80}")
        print(f"QWEN ASR AGENT - DATASET EVALUATION")
        print(f"{'='*80}")
        print(f"Processing {max_files} files starting from index {start_idx}")
        print(f"Language: {language}")
        print(f"Data root: {data_root}")
        print(f"{'='*80}\n")
        
        results: List[Dict[str, Any]] = []
        for idx in range(start_idx, start_idx + max_files):
            res = self.run_on_file(
                audio_idx=idx,
                data_root=data_root,
                language=language,
                verbose=verbose
            )
            results.append(res)
        
        # Calculate average metrics
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_cer = sum(r["cer"] for r in results) / len(results)
        
        print(f"\n{'='*80}")
        print(f"QWEN AGENT - SUMMARY")
        print(f"{'='*80}")
        print(f"Files processed: {len(results)}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"{'='*80}\n")
        
        return results


def main():
    """Main entry point for Qwen ASR agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Qwen ASR Agent")
    parser.add_argument("--language", default=LANGUAGE, help="Language code")
    parser.add_argument("--model-name", default=QWEN_MODEL_NAME, help="Qwen model name")
    parser.add_argument("--max-files", type=int, default=AGENT_MAX_FILES, help="Number of files to process")
    parser.add_argument("--start-idx", type=int, default=AGENT_START_IDX, help="Starting index")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()
    
    # Setup paths
    data_root = DATA_ROOT / args.language
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Initialize agent
    agent = QwenASRAgent(
        model_name=args.model_name,
        load_in_8bit=args.load_8bit,
    )

    # Run on dataset
    results = agent.run_on_dataset(
        data_root=data_root,
        max_files=args.max_files,
        start_idx=args.start_idx,
        language=args.language,
        verbose=not args.quiet,
    )

    # Save results
    QWEN_AGENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    for res in results:
        out_path = QWEN_AGENT_RESULTS_DIR / f"{args.language}_{res['audio_idx']:05d}_qwen_agent.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"Saved: {out_path}")
    
    # Save summary
    summary_path = QWEN_AGENT_RESULTS_DIR / f"{args.language}_qwen_agent_summary.json"
    summary = {
        "model": args.model_name,
        "language": args.language,
        "num_files": len(results),
        "start_idx": args.start_idx,
        "avg_wer": sum(r["wer"] for r in results) / len(results),
        "avg_cer": sum(r["cer"] for r in results) / len(results),
        "results": results,
    }
    
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
