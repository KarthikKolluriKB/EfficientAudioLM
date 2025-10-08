import argparse
import json 
import os 
from types import SimpleNamespace 
from typing import Any, Dict 

import torch 
import yaml

from omegaconf import OmegaConf

# Internal imports
from models.model import model_builder
from utils.metrics import compute_wer
from utils.utils import load_jsonl, import_from_path 

@torch.no_grad()
def run_eval(
    cfg_path: str, 
    ckpt_path: str,
    split: str = "test",
    device: str = "cuda",
    prompt: str = None,
    max_new_tokens: int = 356,
    save_jsonl: str = None,
):
    # 1. Load config 
    cfg = OmegaConf.load(cfg_path) 
    train_cfg, model_cfg, data_cfg = cfg.train, cfg.model, cfg.data

    # 2. Build Model/tokenizer and load projector weights
    model, tokenizer = model_builder(train_cfg, model_cfg, ckpt_path)
    model.eval()
    model.to(device)

    # 3. Build dataset 
    get_dataset = import_from_path(data_cfg.file)
    dataset = get_dataset(data_cfg, tokenizer, split)

    # Enable Inference Mode & set input type to "mel"
    dataset.inference_mode = True 
    dataset.input_type = "mel"

    # If split is "test", load true test JSONL 
    if split.lower() == "test" and getattr(data_cfg, "test_data_path", None): 
        dataset.data = load_jsonl(data_cfg.test_data_path)

    # 4. Prompt 
    default_prompt = (
        "Transcribe speech to text. Output the transcription directly without redundant content. "
        "Ensure that the output is not duplicated."
    )

    use_prompt = default_prompt if (prompt is None) else prompt

    # 5. Iterate items 
    hyps, refs, rows = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        mel = item.get("audio_mel") # [T, 80]
        ref = item.get("target", "") 
        uid = item.get("key", f"utt_{i}")

        if mel is None:
            print(f"Skipping item {i} as audio_mel is None")
            continue


        # ASR-LLM Inference (accepts [B, T, 80] or [B, 80, T])
        hyp = model.inference(
            audio_mel=mel,
            prompt=use_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
        )

        hyp = hyp.strip() if isinstance(hyp, str) else str(hyp).strip()

        # reference
        ref = str(ref).strip() 

        # Store
        hyps.append(hyp)
        refs.append(ref)
        rows.append({"utt_id": uid, "ref": ref, "hyp": hyp})

        # 6. Compute corpus WER and word Accuracy
        wer_value = float(compute_wer(hyps, refs))
        acc_value = 1.0 - wer_value

        # Saving Results
        if save_jsonl is not None:
            os.makedirs(os.path.dirname(save_jsonl), exist_ok=True)
            with open(save_jsonl, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "wer": wer_value,
            "accuracy": acc_value,
            "num_samples": len(refs),
        }

        print(json.dump(summary, indent=2))
        return summary
    

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True, help="Path to YAML config (OmegaConf)")
    arg_parser.add_argument("--ckpt", type=str, required=True, help="Path to projector checkpoint with 'projector' state_dict")
    arg_parser.add_argument("--split", type=str, default="test", help="train/val/test; test reloads test_data_path")
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--prompt", type=str, default="Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated.")
    arg_parser.add_argument("--max_new_tokens", type=int, default=128)
    arg_parser.add_argument("--save_jsonl", type=str, default=None)
    args = arg_parser.parse_args()

    run_eval(
        cfg_path=args.config,
        ckpt_path=args.ckpt,
        split=args.split,
        device=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        save_jsonl=args.save_jsonl,
    )

if __name__ == "__main__":
    main()
