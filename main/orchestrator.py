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
    QWEN_AGENT_RESULTS_DIR,
    QWEN_MODEL_NAME,
    QWEN_LOAD_8BIT,
    QWEN_MAX_NEW_TOKENS,
    QWEN_BATCH_SIZE,
    QWEN_USE_FLASH_ATTENTION,
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
    Optimized agent for multi-GPU inference.
    Uses device_map="auto" for model parallelism and batch processing.
    """

    def __init__(
        self,
        model_name: str = QWEN_MODEL_NAME,
        device: str | None = None,
        load_in_8bit: bool = QWEN_LOAD_8BIT,
        use_flash_attention: bool = QWEN_USE_FLASH_ATTENTION,
    ) -> None:
        print(f"\n{'='*80}")
        print(f"Initializing Qwen ASR Agent")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"8-bit quantization: {load_in_8bit}")
        print(f"Flash attention: {use_flash_attention}")
        print(f"Multi-GPU: Using device_map='auto' for tensor parallelism")
        print(f"{'='*80}\n")

        self.model_name = model_name

        print("Loading Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print("Loading Qwen model (this may take a few minutes)...")
        
        if load_in_8bit and torch.cuda.is_available():
            # 8-bit quantization for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  # Automatic multi-GPU distribution
                load_in_8bit=True,
                trust_remote_code=True,
            )
        else:
            # Use device_map="auto" for multi-GPU tensor parallelism
            model_kwargs = {
                "dtype": torch.bfloat16,  # Fixed: was torch_dtype (deprecated)
                "device_map": "auto",  # Spreads across all available GPUs
                "trust_remote_code": True,
            }
            
            # Only add flash attention if explicitly enabled
            if use_flash_attention:
                print("Attempting to use Flash Attention 2...")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                print("Using default attention implementation")
                model_kwargs["attn_implementation"] = "eager"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )

        self.model.eval()
    
        # Report GPU distribution
        if hasattr(self.model, 'hf_device_map'):
            print(f"\n✓ Model loaded across GPUs:")
            devices_used = set()
            for layer, device in self.model.hf_device_map.items():
                devices_used.add(str(device))
            print(f"  Devices: {', '.join(sorted(devices_used))}")
        else:
            print(f"\n✓ Model loaded on single device")
        
        print(f"✓ Qwen model ready for inference\n")

    def _run_baselines_batch(
        self,
        audio_indices: List[int],
        data_root: Path,
        language: str = LANGUAGE,
    ) -> List[Dict[str, Any]]:
        """
        Run baselines for a BATCH of audio files.
        This is more efficient than processing one at a time.
        """
        results = []
        
        # Run all files through Whisper
        print(f"  Running Whisper on {len(audio_indices)} files...")
        whisper_results = []
        for idx in audio_indices:
            cfg = {
                "backend": "whisper",
                "model_name": WHISPER_MODEL_NAME,
                "whisper_lang": "hi",
                "language": language,
                "data_root": str(data_root),
                "transcription_file": TRANSCRIPTIONS_FILE,
                "quiet": True,
                "max_samples": 1,
                "start_idx": idx,
            }
            whisper_results.append(evaluate_model(cfg))
        
        # Run all files through MMS
        print(f"  Running MMS-1B on {len(audio_indices)} files...")
        mms_results = []
        for idx in audio_indices:
            cfg = {
                "backend": "mms",
                "model_name": MMS_MODEL_ID,
                "target_lang": MMS_TARGET_LANG,
                "language": language,
                "data_root": str(data_root),
                "transcription_file": TRANSCRIPTIONS_FILE,
                "quiet": True,
                "max_samples": 1,
                "start_idx": idx,
            }
            mms_results.append(evaluate_model(cfg))
        
        # Run all files through MMS-ZS
        print(f"  Running MMS-ZeroShot on {len(audio_indices)} files...")
        mms_zs_results = []
        for idx in audio_indices:
            cfg = {
                "backend": "mms_zeroshot",
                "model_name": MMS_ZEROSHOT_MODEL_ID,
                "language": language,
                "data_root": str(data_root),
                "transcription_file": TRANSCRIPTIONS_FILE,
                "quiet": True,
                "max_samples": 1,
                "start_idx": idx,
            }
            mms_zs_results.append(evaluate_model(cfg))
        
        # Combine results
        for i, idx in enumerate(audio_indices):
            results.append({
                "audio_idx": idx,
                "whisper": whisper_results[i]["hyps"][0],
                "mms_1b": mms_results[i]["hyps"][0],
                "mms_zs_uroman": mms_zs_results[i]["hyps"][0],
                "ref_dev": whisper_results[i]["refs"][0],
            })
        
        return results

    def _build_system_prompt(self) -> str:
        """System prompt for Qwen."""
        return """You are an expert Hindi ASR judge.

You will receive multiple ASR hypotheses for the SAME Hindi utterance from different systems.

Your task:
1. Carefully analyze all candidate transcriptions
2. Select or intelligently combine the best elements
3. Output the most accurate transcription

Output format (MANDATORY):
LANGUAGE: Hindi
TRANSCRIPTION: <best transcription in Devanagari>
CONFIDENCE: <high|medium|low>
REASONING: <brief explanation in English>
"""

    def _build_user_prompt(
        self,
        audio_name: str,
        hypotheses: Dict[str, str],
    ) -> str:
        lines = [
            f"Audio: {audio_name}",
            "",
            "ASR Hypotheses:",
        ]

        for src, hyp in hypotheses.items():
            if src == "ref_dev":
                continue
            lines.append(f"- {src}: {hyp}")

        lines.append("")
        lines.append("Select or combine the best transcription.")
        return "\n".join(lines)

    def _call_qwen_batch(
        self,
        prompts: List[tuple],  # List of (system, user) prompt pairs
        batch_size: int = QWEN_BATCH_SIZE,
    ) -> List[str]:
        """
        Process multiple prompts in batches for efficiency.
        """
        responses = []
        
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(prompts), batch_size):
            batch = prompts[batch_idx:batch_idx+batch_size]
            batch_num = batch_idx // batch_size + 1
            
            print(f"    Processing Qwen batch {batch_num}/{num_batches} ({len(batch)} prompts)...")
            
            # Prepare messages for batch
            batch_texts = []
            for system_prompt, user_prompt in batch:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                batch_texts.append(text)
            
            # Tokenize batch
            model_inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            
            # Move to appropriate device
            if hasattr(self.model, 'device'):
                model_inputs = model_inputs.to(self.model.device)
            else:
                # For device_map="auto", use the first device
                first_device = next(self.model.parameters()).device
                model_inputs = model_inputs.to(first_device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=QWEN_MAX_NEW_TOKENS,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,  # Deterministic for consistency
                )
            
            # Decode
            for j, output_ids in enumerate(generated_ids):
                input_len = model_inputs.input_ids[j].shape[0]
                response = self.tokenizer.decode(
                    output_ids[input_len:],
                    skip_special_tokens=True
                )
                responses.append(response)
        
        return responses

    def _parse_final_answer(self, response: str) -> Dict[str, Any]:
        """Parse Qwen's structured response."""
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

    def run_on_dataset(
        self,
        data_root: Path,
        max_files: int = AGENT_MAX_FILES,
        start_idx: int = AGENT_START_IDX,
        language: str = LANGUAGE,
        batch_size: int = QWEN_BATCH_SIZE,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run Qwen agent on multiple files with batching for efficiency.
        """
        
        print(f"\n{'='*80}")
        print(f"QWEN ASR AGENT - MULTI-GPU OPTIMIZED EVALUATION")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Files to process: {max_files} (starting from index {start_idx})")
        print(f"Qwen batch size: {batch_size}")
        print(f"Language: {language}")
        print(f"Data root: {data_root}")
        print(f"{'='*80}\n")
        
        audio_indices = list(range(start_idx, start_idx + max_files))
        
        # Step 1: Run all baselines
        print(f"Step 1/4: Running baseline ASR models...")
        print(f"{'-'*80}")
        baseline_results = self._run_baselines_batch(audio_indices, data_root, language)
        print(f"✓ Baseline models completed\n")
        
        # Step 2: Prepare Qwen prompts
        print(f"Step 2/4: Preparing Qwen prompts...")
        print(f"{'-'*80}")
        prompts = []
        for result in baseline_results:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                f"{language}_{result['audio_idx']:05d}",
                {
                    "whisper": result["whisper"],
                    "mms_1b": result["mms_1b"],
                    "mms_zeroshot": result["mms_zs_uroman"],
                }
            )
            prompts.append((system_prompt, user_prompt))
        print(f"✓ Prepared {len(prompts)} prompts\n")
        
        # Step 3: Run Qwen in batches
        print(f"Step 3/4: Running Qwen inference...")
        print(f"{'-'*80}")
        qwen_responses = self._call_qwen_batch(prompts, batch_size=batch_size)
        print(f"✓ Qwen inference completed\n")
        
        # Step 4: Parse and evaluate
        print(f"Step 4/4: Parsing results and calculating metrics...")
        print(f"{'-'*80}")
        final_results = []
        
        for i, result in enumerate(baseline_results):
            parsed = self._parse_final_answer(qwen_responses[i])
            qwen_trans = parsed["transcription"]
            ref = result["ref_dev"]
            
            # Calculate metrics
            wer_val = float(wer([ref], [qwen_trans])) if qwen_trans else 1.0
            cer_val = float(cer([ref], [qwen_trans])) if qwen_trans else 1.0
            
            final_results.append({
                "audio_idx": result["audio_idx"],
                "audio_name": f"{language}_{result['audio_idx']:05d}",
                "whisper_hyp": result["whisper"],
                "mms_1b_hyp": result["mms_1b"],
                "mms_zs_hyp": result["mms_zs_uroman"],
                "ref_dev": ref,
                "qwen_decision": parsed,
                "wer": wer_val,
                "cer": cer_val,
            })
            
            if verbose:
                print(f"  [{i+1}/{len(baseline_results)}] {result['audio_idx']:05d}: WER={wer_val:.4f}, CER={cer_val:.4f}, Conf={parsed['confidence']}")
        
        # Summary
        avg_wer = sum(r["wer"] for r in final_results) / len(final_results)
        avg_cer = sum(r["cer"] for r in final_results) / len(final_results)
        
        print(f"\n{'='*80}")
        print(f"QWEN AGENT - EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Files processed: {len(final_results)}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"{'='*80}\n")
        
        return final_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Qwen ASR Agent (Multi-GPU Optimized)")
    parser.add_argument("--language", default=LANGUAGE, help="Language code")
    parser.add_argument("--model-name", default=QWEN_MODEL_NAME, help="Qwen model name")
    parser.add_argument("--max-files", type=int, default=AGENT_MAX_FILES, help="Number of files to process")
    parser.add_argument("--start-idx", type=int, default=AGENT_START_IDX, help="Starting audio index")
    parser.add_argument("--batch-size", type=int, default=QWEN_BATCH_SIZE, help="Qwen inference batch size")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit mode")
    parser.add_argument("--no-flash-attention", action="store_true", help="Disable flash attention")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()
    
    # Setup paths
    data_root = DATA_ROOT / args.language
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Initialize agent (uses all available GPUs automatically)
    agent = QwenASRAgent(
        model_name=args.model_name,
        load_in_8bit=args.load_8bit,
        use_flash_attention=not args.no_flash_attention,
    )

    # Run on dataset
    results = agent.run_on_dataset(
        data_root=data_root,
        max_files=args.max_files,
        start_idx=args.start_idx,
        language=args.language,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )

    # Save results
    QWEN_AGENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    print(f"Saving results to: {QWEN_AGENT_RESULTS_DIR}")
    for res in results:
        out_path = QWEN_AGENT_RESULTS_DIR / f"{args.language}_{res['audio_idx']:05d}_qwen_agent.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "model": args.model_name,
        "language": args.language,
        "num_files": len(results),
        "start_idx": args.start_idx,
        "avg_wer": sum(r["wer"] for r in results) / len(results),
        "avg_cer": sum(r["cer"] for r in results) / len(results),
        "results": results,
    }
    
    summary_path = QWEN_AGENT_RESULTS_DIR / f"{args.language}_qwen_agent_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✓ All results saved")
    print(f"  Individual results: {QWEN_AGENT_RESULTS_DIR}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
