from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from jiwer import wer, cer
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str | None = None,
        load_in_8bit: bool = False,
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
    ) -> Dict[str, Any]:
        """
        Run Whisper, MMS-1B, MMS-ZS on a *single* audio index.

        Uses eval_engine with max_samples=1 and start_idx=audio_idx.
        """

        whisper_cfg = {
            "backend": "whisper",
            "model_name": "small",
            "language": "hi",
            "data_root": str(data_root),
            "transcription_file": "transcriptions.txt",
            "quiet": True,
            "max_samples": 1,       # exactly one file
            "start_idx": audio_idx,
        }

        mms_cfg = {
            "backend": "mms",
            "model_name": "facebook/mms-1b-all",
            "target_lang": "hin",
            "data_root": str(data_root),
            "transcription_file": "transcriptions.txt",
            "quiet": True,
            "max_samples": 1,
            "start_idx": audio_idx,
        }

        mms_zs_cfg = {
            "backend": "mms_zeroshot",
            "model_name": "mms-meta/mms-zeroshot-300m",
            "data_root": str(data_root),
            "transcription_file": "transcriptions_uroman.txt",
            "quiet": True,
            "max_samples": 1,
            "start_idx": audio_idx,
        }

        print(f"Evaluating on {whisper_cfg['model_name']} (idx={audio_idx})")
        whisper_res = evaluate_model(whisper_cfg)
        print(f"Evaluating on {mms_cfg['model_name']} (idx={audio_idx})")
        mms_res = evaluate_model(mms_cfg)
        print(f"Evaluating on {mms_zs_cfg['model_name']} (idx={audio_idx})")
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
          - Whisper (native script or romanized)
          - MMS-1B (native script or romanized)
          - MMS-zeroshot (uroman romanization)
        - The target language is Hindi.
        
        Your task:
        1. Carefully read all candidate transcriptions.
        2. Decide which hypothesis is the most accurate and fluent Hindi sentence.
        3. Optionally combine or lightly correct them if that clearly improves accuracy.
        4. Output your final answer in a STRICT format in English as the conversation language.
        
        Output format (MANDATORY):
        LANGUAGE: <Detected Language of Transcription>
        TRANSCRIPTION: <your best transcription in Devanagari if possible; else consistent romanized Hindi>
        CONFIDENCE: <high|medium|low>
        REASONING: <1-3 sentences in English explaining which system(s) you trusted and why>
        """

    def _build_user_prompt(
        self,
        audio_name: str,
        hypotheses: Dict[str, str],
    ) -> str:
        lines = [f"Audio file: {audio_name}",
                 "Here are the ASR hypotheses:"]

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
                max_new_tokens=512,
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
        verbose: bool = True,
    ) -> Dict[str, Any]:
        baseline_out = self._run_baselines_for_file(audio_idx, data_root)
        whisper_hyp = baseline_out["whisper"]
        mms_hyp = baseline_out["mms_1b"]
        mms_zs_hyp = baseline_out["mms_zs_uroman"]
        ref_dev = baseline_out["ref_dev"]

        audio_name = f"hindi_{audio_idx:03d}.wav"

        if verbose:
            print("=" * 80)
            print(f"Qwen Hindi ASR Agent on {audio_name}")
            print("=" * 80)
            print("Whisper:", whisper_hyp)
            print("MMS-1B:", mms_hyp)
            print("MMS-ZS (uroman):", mms_zs_hyp)
            print("REF (Devanagari):", ref_dev)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            audio_name,
            {
                "whisper": whisper_hyp,
                "mms_1b": mms_hyp,
                "mms_zeroshot": mms_zs_hyp,
            },
        )

        response = self._call_qwen(system_prompt, user_prompt)

        if verbose:
            print("\nQwen raw response:\n", response)

        parsed = self._parse_final_answer(response)
        qwen_transcription = parsed["transcription"]

        wer_val = float(wer([ref_dev], [qwen_transcription]))
        cer_val = float(cer([ref_dev], [qwen_transcription]))

        if verbose:
            print("\nEVAL vs ground truth:")
            print("QWEN TRANSCRIPTION:", qwen_transcription)
            print("REF:", ref_dev)
            print(f"WER: {wer_val:.3f}")
            print(f"CER: {cer_val:.3f}")

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
        max_files: int = 5,
        start_idx: int = 0,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx in range(start_idx, start_idx + max_files):
            res = self.run_on_file(audio_idx=idx, data_root=data_root, verbose=verbose)
            results.append(res)
        return results


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "data" / "hindi_audio"

    agent = QwenASRAgent(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_8bit=False,
    )

    results = agent.run_on_dataset(
        data_root=data_root,
        max_files=5,
        start_idx=0,
        verbose=True,
    )

    out_dir = repo_root / "results" / "qwen_agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        out_path = out_dir / f"hindi_{res['audio_idx']:03d}_qwen_agent.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\nSaved Qwen agent result to {out_path}")
