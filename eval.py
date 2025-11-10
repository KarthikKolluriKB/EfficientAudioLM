import argparse
import json 
import os 
import gc
import torch 

from tqdm import tqdm
from omegaconf import OmegaConf
from types import SimpleNamespace 
from torch.utils.data import DataLoader

# Internal imports
from models.model import model_builder
from datamodule.dataset import get_speech_dataset
from utils.metrics import decode_texts_from_outputs
from utils.wand_config import init_wandb
from utils.log_config import get_logger
from utils.train_utils import save_and_print_examples

logger = get_logger(log_dir="logs", filename="eval.log")

@torch.no_grad()
def run_eval(args):
    """
    Run evaluation on the test set with improved error handling and memory efficiency.
    """
    run = None
    model = None
    device = None
    
    try:
        # 1. Load config and check files exist
        if not os.path.exists(args.cfg_path):
            raise FileNotFoundError(f"Config file not found: {args.cfg_path}")
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")
        
        cfg = OmegaConf.load(args.cfg_path)
        train_cfg, model_cfg, data_cfg, wandb_cfg = cfg.train, cfg.model, cfg.data, cfg.log
        
        # 2. Initialize wandb if specified
        if wandb_cfg.use_wandb:
            run = init_wandb(
                use_wand=True,
                project=wandb_cfg.wandb_project_name,
                run_name=args.wandb_exp_name if args.wandb_exp_name else wandb_cfg.wandb_experiment_name, # Use arg if provided
                tags=["eval", "asr-llm"],
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        
        # 3. Build Model/tokenizer and load weights
        model, tokenizer = model_builder(train_cfg, model_cfg, ckpt_path=args.ckpt_path)
        
        # 4. Move model to device
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()
        logger.info(f"Model moved to device: {device}")
        
        # 5. Load Test Dataset
        test_dataset = get_speech_dataset(data_cfg, tokenizer, split=args.split)
        logger.info(f"Loaded {len(test_dataset)} samples from {args.split} split")
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=test_dataset.collator,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
        )
        
        # 6. Initialize tracking variables for memory efficiency
        num_samples = 0
        num_batches = 0
        running_loss = 0.0
        running_acc = 0.0
        running_wer = 0.0

        # predicted_texts, target_texts
        all_pred_texts = []
        all_target_texts = []
        
        # 7. Evaluation Loop
        logger.info("Starting evaluation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass through model to get outputs
                model_kwargs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                    "audio_mel": batch["audio_mel"],
                    "modality_mask": batch["modality_mask"]
                }
                
                outputs, metrics = model(**model_kwargs)


                # Reference, Prediction texts
                pred_texts, target_texts = decode_texts_from_outputs(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    tokenizer=tokenizer,
                    ignore_label=-100,
                )

                # Accumulate all texts
                all_pred_texts.extend(pred_texts)
                all_target_texts.extend(target_texts)
                
                # Accumulate metrics with running averages for memory efficiency
                batch_loss = outputs.loss.item()
                batch_acc = metrics.get('acc', 0)
                batch_wer = metrics.get('wer', 0)
                
                running_loss += batch_loss
                running_acc += batch_acc
                running_wer += batch_wer
                num_batches += 1
                
                # Update number of samples
                batch_size = batch["input_ids"].size(0)
                num_samples += batch_size
                
                # Log batch metrics for first few batches
                if num_samples <= (5 * args.batch_size):
                    logger.info(f"\nBatch {batch_idx + 1} metrics:")
                    logger.info(f"Loss: {batch_loss:.4f}")
                    logger.info(f"Accuracy: {batch_acc:.4f}")
                    logger.info(f"WER: {batch_wer:.4f}")
                    logger.info(f"Word Accuracy: {1 - batch_wer:.4f}")

                    # Log predicted vs target texts for first few batches
                    for i in range(min(3, len(pred_texts))):
                        logger.info(f"\nSamples {i + 1} in Batch {batch_idx + 1}:")
                        logger.info(f"Target: {target_texts[i]}")
                        logger.info(f"Predicted: {pred_texts[i]}")
                
                # Clear cache periodically to prevent memory buildup
                if batch_idx % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # 8. Calculate and log final results
        if num_batches > 0:
            avg_metrics = {
                'loss': running_loss / num_batches,
                'acc': running_acc / num_batches,
                'wer': running_wer / num_batches
            }
            
            # Log average metrics
            logger.info("\n" + "="*50)
            logger.info("FINAL EVALUATION RESULTS:")
            logger.info(f"Total samples processed: {num_samples}")
            logger.info(f"Total batches processed: {num_batches}")
            logger.info(f"Avg Loss: {avg_metrics['loss']:.4f}")
            logger.info(f"Avg Accuracy: {avg_metrics['acc']:.4f}")
            logger.info(f"Avg WER: {avg_metrics['wer']:.4f}")
            logger.info(f"Avg Word Accuracy: {1 - avg_metrics['wer']:.4f}")
            logger.info("="*50)
            
            # Log to wandb if enabled
            if run is not None:
                run.log({
                    "test/avg_loss": avg_metrics['loss'],
                    "test/avg_acc": avg_metrics['acc'],
                    "test/avg_wer": avg_metrics['wer'],
                    "test/avg_word_acc": 1 - avg_metrics['wer'],
                    "test/total_samples": num_samples,
                    "test/total_batches": num_batches,
                })

            # Save predictions and targets to JSONL
            save_and_print_examples(all_pred_texts, 
                                    all_target_texts, 
                                    output_path=args.output_dir,
                                    epoch=0,
                                    n_save=10,
                                    n_print=0,
                                    run=run,
                                    seed=42)
        else:
            logger.error("No batches were successfully processed!")
    
    # Handle specific CUDA out of memory errors        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory: {e}")
        logger.error("Try reducing batch size or using CPU")
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        
    # Handle general CUDA runtime errors
    except RuntimeError as e:
        if "cuda" in str(e).lower():
            logger.error(f"CUDA runtime error: {e}")
            if device and device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            logger.error(f"Runtime error: {e}")
    
    # Handle file not found errors
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        
    # Handle OmegaConf configuration errors
    except Exception as e:
        if "OmegaConf" in str(type(e)):
            logger.error(f"Configuration error: {e}")
        else:
            logger.error(f"Model building error: {e}")
            
    # Handle wandb connection issues
    except Exception as e:
        if "wandb" in str(e).lower():
            logger.warning(f"Wandb error (continuing without logging): {e}")
        else:
            logger.exception(f"Unexpected error during evaluation: {e}")
    
    # Handle keyboard interrupt gracefully
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        
    # Catch any other unexpected errors
    except Exception as e:
        logger.exception(f"Unexpected error in run_eval: {e}")
        
    finally:
        # Clean up resources
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        if run is not None:
            try:
                run.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASR-LLM model on test set")
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
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
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--wandb_exp_name",
        type=str,
        default=None,
        help="Wandb experiment name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # For debugging: create temporary args
    DEBUG = False
    if DEBUG:
        args = SimpleNamespace(
            cfg_path="configs/eval_config.yaml",
            ckpt_path="outputs/ASRLLM_enc_lin_20h/projector_best_wer.pt",
            split="test",
            batch_size=32,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_new_tokens=256
        )
        logger.info("Using debug arguments")
    else:
        args = parse_args()
    
    run_eval(args)
