import argparse
from cProfile import label
import json 
import os 
from pathlib import Path
from types import SimpleNamespace 
from typing import Any, Dict 

import torch 
from tqdm import tqdm
import yaml
import logging

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Internal imports
from models.model import model_builder
from utils.metrics import compute_accuracy, compute_wer, decode_texts_from_outputs
from datamodule.dataset import get_speech_dataset

SPLIT = "test"
BATCH_SIZE = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@torch.no_grad()
def run_eval(args):
    # 1. Load config 
    cfg = OmegaConf.load(args.cfg_path) 
    train_cfg, model_cfg, data_cfg = cfg.train, cfg.model, cfg.data

    # 2. Build Model/tokenizer and load projector weights
    model, tokenizer = model_builder(train_cfg, model_cfg, ckpt_path=args.ckpt_path)
    
    # Move model to device
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # 3. Load Test Dataset
    test_dataset = get_speech_dataset(data_cfg, tokenizer, split="test")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=test_dataset.collator,
        num_workers=4,
        pin_memory=True,
    )
    
    results = []
    total_wer = 0 
    num_samples = 0

    # Evaluation Loop
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating..."):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Generate predictions
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_mel=batch["audio_mel"],
                modality_mask=batch["modality_mask"],
                max_new_tokens=args.max_new_tokens,
            )

            # Get labels
            labels = batch["labels"]

            # Compute Accuracy 
            preds = torch.argmax(outputs.logits, dim=-1)
            batch_acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)
            logger.info(f"Batch Accuracy: {batch_acc:.4f}")
            

            # Decode predictions and references
            pred_texts, ref_texts = decode_texts_from_outputs(
                logits=outputs.logits.detach(),
                labels=labels.detach(),
                tokenizer=tokenizer,
                ignore_label=-100, 
            )

            # Compute WER for the batch
            batch_wer = compute_wer(pred_texts, ref_texts)
            total_wer += batch_wer * len(pred_texts)
            num_samples += len(pred_texts)

            # Save results
            for idx, (pred, ref) in enumerate(zip(pred_texts, ref_texts)):
                result = {
                    "id": batch["ids"][idx],
                    "prediction": pred,
                    "reference": ref,
                }
                results.append(result)

            # Average WER score 
            avg_wer = total_wer / num_samples if num_samples > 0 else float("inf")
            logger.info(f"Test Average WER: {avg_wer:.4f}\n")
            logger.info(f"Test Average Word Accuracy: {1 - avg_wer:.4f}\n")
            # Average Accuracy  
            avg_acc = batch_acc / num_samples if num_samples > 0 else 0.0
            logger.info(f"Test Average Accuracy: {avg_acc:.4f}\n")

            # Save results to JSONL if specified
            if args.save_jsonl:
                output_path = Path(args.save_jsonl)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w") as f: 
                    for res in results:
                        f.write(json.dumps(res) + "\n")

                    # write summary at the end 
                    summary = {
                        "wer": avg_wer,
                        "word_acc": 1 - avg_wer,
                        "accuracy": avg_acc,
                        "num_samples": num_samples,
                    }
                    f.write(json.dumps(summary) + "\n")

                logger.info(f"Saved results to {output_path}")

    logger.info("Evaluation Completed...!")

    return avg_wer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the evaluation on.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--save_jsonl",
        type=str,
        default=None,
        help="Path to save the evaluation results in JSONL format.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # For debugging: create temporary args
    DEBUG = False
    if DEBUG:
        args = SimpleNamespace(
            cfg_path="configs/config.yaml",
            ckpt_path="outputs/ASRLLM_enc_lin_20h/projector_best_wer.pt",
            split="test",
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_new_tokens=256,
            save_jsonl="results.jsonl"
        )
        logger.info("Using debug arguments")
    else:
        args = parse_args()
    run_eval(args)

