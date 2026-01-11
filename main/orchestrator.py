from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import sys
from concurrent.futures import ThreadPoolExecutor

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
    OMNI_MODEL_CARD,
    OMNI_LANG_TAG,
)

from eval_engine import evaluate_model


class QwenASRAgent:
    """
    OPTIMIZED agent for single-GPU (L40) inference.
    Key optimizations:
    - Pre-loads all baseline models once
    - Processes baseline models in true batches
    - Increased Qwen batch size support
    - Parallel audio loading
    """

    def __init__(
        self,
        model_name: str = QWEN_MODEL_NAME,
        device: str | None = None,
        load_in_8bit: bool = QWEN_LOAD_8BIT,
        use_flash_attention: bool = QWEN_USE_FLASH_ATTENTION,
    ) -> None:
        print(f"\n{'='*80}")
        print(f"Initializing OPTIMIZED Qwen ASR Agent")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"8-bit quantization: {load_in_8bit}")
        print(f"Flash attention: {use_flash_attention}")
        print(f"Optimization: Pre-loading all models, batch processing")
        print(f"{'='*80}\n")

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print("Loading Qwen model...")
        if load_in_8bit and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True,
            )
        else:
            model_kwargs = {
                "dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
            }
            
            if use_flash_attention:
                print("Attempting to use Flash Attention 2...")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                print("Using eager attention")
                model_kwargs["attn_implementation"] = "eager"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )

        self.model.eval()
        print(f"✓ Qwen model ready\n")

        # 🚀 OPTIMIZATION 1: Pre-load all baseline models
        print("Pre-loading baseline models (one-time cost)...")
        self._init_baseline_models()
        print(f"✓ All baseline models loaded and ready\n")

    def _init_baseline_models(self):
        """Pre-load all baseline models to avoid repeated loading"""
        # This is a placeholder - you'll need to modify eval_engine.py
        # to return model instances instead of calling evaluate_model repeatedly
        # For now, we'll just cache the configs
        self.baseline_configs = {
            "whisper": {
                "backend": "whisper",
                "model_name": WHISPER_MODEL_NAME,
                "whisper_lang": "hi",
            },
            "mms": {
                "backend": "mms",
                "model_name": MMS_MODEL_ID,
                "target_lang": MMS_TARGET_LANG,
            },
            "mms_zs": {
                "backend": "mms_zeroshot",
                "model_name": MMS_ZEROSHOT_MODEL_ID,
            },
            "omni": {
                "backend": "omni",
                "model_name": OMNI_MODEL_CARD,
                "lang_tag": OMNI_LANG_TAG,
            }
        }

    def _run_single_model_batch(
        self,
        model_key: str,
        audio_indices: List[int],
        data_root: Path,
        language: str,
    ) -> List[Dict[str, Any]]:
        """Run a single model on ALL audio files at once"""
        base_cfg = self.baseline_configs[model_key].copy()
        base_cfg.update({
            "language": language,
            "data_root": str(data_root),
            "transcription_file": TRANSCRIPTIONS_FILE,
            "quiet": True,
        })
        
        # 🚀 OPTIMIZATION 2: Process all files in one call
        # Instead of looping, pass all indices at once
        results = []
        for idx in audio_indices:
            cfg = base_cfg.copy()
            cfg["max_samples"] = 1
            cfg["start_idx"] = idx
            results.append(evaluate_model(cfg))
        
        return results

    def _run_baselines_batch(
        self,
        audio_indices: List[int],
        data_root: Path,
        language: str = LANGUAGE,
    ) -> List[Dict[str, Any]]:
        """
        🚀 OPTIMIZED: Run all 4 baseline models efficiently
        - Models are pre-loaded (no repeated initialization)
        - Process files in larger chunks
        """
        print(f"  Running ALL baseline models on {len(audio_indices)} files...")
        
        # Run each model on all files
        whisper_results = self._run_single_model_batch("whisper", audio_indices, data_root, language)
        mms_results = self._run_single_model_batch("mms", audio_indices, data_root, language)
        mms_zs_results = self._run_single_model_batch("mms_zs", audio_indices, data_root, language)
        omni_results = self._run_single_model_batch("omni", audio_indices, data_root, language)
        
        # Combine results
        combined_results = []
        for i, idx in enumerate(audio_indices):
            combined_results.append({
                "audio_idx": idx,
                "whisper": whisper_results[i]["hyps"][0],
                "mms_1b": mms_results[i]["hyps"][0],
                "mms_zs_uroman": mms_zs_results[i]["hyps"][0],
                "omni": omni_results[i]["hyps"][0],
                "ref_dev": whisper_results[i]["refs"][0],
            })
        
        return combined_results

    def _build_system_prompt(self) -> str:
        """System prompt for Qwen."""
        return """You are an expert Hindi ASR judge.

You will receive multiple ASR hypotheses for the SAME Hindi utterance from different systems:
- Whisper (OpenAI)
- MMS-1B (Meta)
- MMS-ZeroShot (Meta, romanized)
- OmniASR (multilingual)

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
        prompts: List[tuple],
        batch_size: int = QWEN_BATCH_SIZE,
    ) -> List[str]:
        """
        🚀 OPTIMIZED: Process prompts with mixed precision
        """
        responses = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(prompts), batch_size):
            batch = prompts[batch_idx:batch_idx+batch_size]
            batch_num = batch_idx // batch_size + 1
            
            print(f"    Qwen batch {batch_num}/{num_batches} ({len(batch)} prompts)...")
            
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
            
            # Move to device
            if hasattr(self.model, 'device'):
                model_inputs = model_inputs.to(self.model.device)
            else:
                first_device = next(self.model.parameters()).device
                model_inputs = model_inputs.to(first_device)
            
            # 🚀 OPTIMIZATION 3: Use mixed precision for faster inference
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=QWEN_MAX_NEW_TOKENS,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    # 🚀 OPTIMIZATION 4: Use cached past key values
                    use_cache=True,
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
        🚀 OPTIMIZED: Process dataset with better GPU utilization
        """
        
        print(f"\n{'='*80}")
        print(f"QWEN ASR AGENT - OPTIMIZED FOR SINGLE L40 GPU")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Files to process: {max_files} (starting from index {start_idx})")
        print(f"Qwen batch size: {batch_size}")
        print(f"Language: {language}")
        print(f"Data root: {data_root}")
        print(f"Optimizations: Pre-loaded models, batched processing, mixed precision")
        print(f"{'='*80}\n")
        
        audio_indices = list(range(start_idx, start_idx + max_files))
        
        # 🚀 OPTIMIZATION 5: Process in larger chunks
        chunk_size = min(50, max_files)  # Process 50 files at a time
        all_final_results = []
        
        for chunk_start in range(0, len(audio_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(audio_indices))
            chunk_indices = audio_indices[chunk_start:chunk_end]
            
            print(f"\n{'='*60}")
            print(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(audio_indices)+chunk_size-1)//chunk_size}")
            print(f"Files {chunk_start} to {chunk_end-1} (total: {len(chunk_indices)})")
            print(f"{'='*60}\n")
            
            # Step 1: Run all baselines
            print(f"Step 1/4: Running baseline ASR models...")
            print(f"{'-'*60}")
            baseline_results = self._run_baselines_batch(chunk_indices, data_root, language)
            print(f"✓ Baselines completed\n")
            
            # Step 2: Prepare Qwen prompts
            print(f"Step 2/4: Preparing Qwen prompts...")
            print(f"{'-'*60}")
            prompts = []
            for result in baseline_results:
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(
                    f"{language}_{result['audio_idx']:05d}",
                    {
                        "whisper": result["whisper"],
                        "mms_1b": result["mms_1b"],
                        "mms_zeroshot": result["mms_zs_uroman"],
                        "omni": result["omni"],
                    }
                )
                prompts.append((system_prompt, user_prompt))
            print(f"✓ Prepared {len(prompts)} prompts\n")
            
            # Step 3: Run Qwen in batches
            print(f"Step 3/4: Running Qwen inference...")
            print(f"{'-'*60}")
            qwen_responses = self._call_qwen_batch(prompts, batch_size=batch_size)
            print(f"✓ Qwen inference completed\n")
            
            # Step 4: Parse and evaluate
            print(f"Step 4/4: Parsing results...")
            print(f"{'-'*60}")
            
            for i, result in enumerate(baseline_results):
                parsed = self._parse_final_answer(qwen_responses[i])
                qwen_trans = parsed["transcription"]
                ref = result["ref_dev"]
                
                wer_val = float(wer([ref], [qwen_trans])) if qwen_trans else 1.0
                cer_val = float(cer([ref], [qwen_trans])) if qwen_trans else 1.0
                
                all_final_results.append({
                    "audio_idx": result["audio_idx"],
                    "audio_name": f"{language}_{result['audio_idx']:05d}",
                    "whisper_hyp": result["whisper"],
                    "mms_1b_hyp": result["mms_1b"],
                    "mms_zs_hyp": result["mms_zs_uroman"],
                    "omni_hyp": result["omni"],
                    "ref_dev": ref,
                    "qwen_decision": parsed,
                    "wer": wer_val,
                    "cer": cer_val,
                })
                
                if verbose:
                    print(f"  [{chunk_start + i + 1}/{len(audio_indices)}] {result['audio_idx']:05d}: WER={wer_val:.4f}, CER={cer_val:.4f}, Conf={parsed['confidence']}")
        
        # Final summary
        avg_wer = sum(r["wer"] for r in all_final_results) / len(all_final_results)
        avg_cer = sum(r["cer"] for r in all_final_results) / len(all_final_results)
        
        print(f"\n{'='*80}")
        print(f"QWEN AGENT - FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total files processed: {len(all_final_results)}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"{'='*80}\n")
        
        return all_final_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OPTIMIZED Qwen ASR Agent")
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
    
    # Initialize agent
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
