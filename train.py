import os 
import sys
import time 
import argparse 

import torch 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from omegaconf import OmegaConf

# internal imports
from utils.utils import set_seed, get_device, resolve_pad_token, ensure_dir, save_projector
from utils.log_config import get_logger
from utils.wand_config import init_wandb
from models.model import model_builder
from datamodule.dataset import get_speech_dataset
from utils.metrics import decode_texts_from_outputs, compute_wer
from utils.train_utils import save_and_print_examples

from torchtnt.utils.early_stop_checker import EarlyStopChecker
from transformers import get_cosine_schedule_with_warmup

def evaluate(cfg, model, dataloader, device, tokenizer):
    """Evaluate the model on the given dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing validation/test data
        device: Device to run evaluation on 
        tokenizer: Tokenizer for decoding outputs
        
    Returns:
        tuple: Contains (validation loss, accuracy, WER score, word accuracy, 
               hypothesis texts, reference texts)
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_accuracies = []
    all_wer_scores = []
    all_hyp_texts, all_ref_texts = [], []
    
    use_autocast = bool(cfg.train.mixed_precision and getattr(device, "type", str(device)) == "cuda")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            audio_mel = batch["audio_mel"].to(device)
            modality_mask = batch['modality_mask'].to(device)
            

            if use_autocast:
                with torch.autocast(device_type=device, dtype=amp_dtype): 
                    # Forward pass
                    outputs, metrics = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel,
                        modality_mask=modality_mask
                    )
                    
            else: 
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask
                )
            
            # Accumulate loss
            total_loss += outputs.loss.item()
            n_batches += 1

            # Store metrics from model
            if metrics is not None:
                if "acc" in metrics:
                    all_accuracies.append(metrics["acc"])
                if "wer" in metrics:
                    all_wer_scores.append(metrics["wer"])

            # Get decoded texts from logits for final evaluation
            hyp_texts, ref_texts = decode_texts_from_outputs(
                logits=outputs.logits,
                labels=labels,
                tokenizer=tokenizer,
                ignore_label=-100
            )
            
            # Accumulate texts
            all_hyp_texts.extend(hyp_texts)
            all_ref_texts.extend(ref_texts)

    # Calculate final metrics
    val_loss = total_loss / max(n_batches, 1)
    val_acc = sum(all_accuracies) / max(len(all_accuracies), 1) if all_accuracies else 0.0
    val_wer_score = sum(all_wer_scores) / max(len(all_wer_scores), 1) if all_wer_scores else 0.0
    val_word_acc = 1.0 - val_wer_score if val_wer_score >= 0.0 else 0.0
    
    # Reset model to training mode
    model.train()
    
    return val_loss, val_acc, val_wer_score, val_word_acc, all_hyp_texts, all_ref_texts


def main(): 
    """
    Main training loop for training the projector module.
    """
    # For better matmul performance on GPUs with Tensor Cores 
    torch.set_float32_matmul_precision("high")   

    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, default="configs/config.yaml", required=True, help="Path to the config file.")
    args = parser.parse_args() 
    #args = argparse.Namespace( config='configs/config.yaml')

    # load YAML into an OmegaConf dict
    cfg = OmegaConf.load(args.config)

    # Prepare logging/output directories
    ensure_dir(cfg.train.output_dir)
    log_dir = os.path.join(cfg.log.log_dir) if cfg.log.log_dir else "."
    ensure_dir(log_dir)
    logger = get_logger(log_dir=log_dir, filename=os.path.basename(cfg.log.log_filename))
    logger.info(f"Loaded configuration file from {args.config}")

    # Seed and device
    set_seed(cfg.train.seed)
    logger.info(f"Set random seed to {cfg.train.seed}")
    device = get_device() 
    logger.info(f"Device: {device}")

    # Early Stopping 
    if not cfg.early_stopping.get("enabled", True):
        early_stop = EarlyStopChecker(
            mode=cfg.early_stopping.mode,
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            threshold_mode="abs"  
        )
    
        logger.info(f"Early stopping configured with mode={cfg.early_stopping.mode}, patience={cfg.early_stopping.patience}, min_delta={cfg.early_stopping.min_delta}, threshold_mode=abs")

    # Model (ASR + LLM)
    model, tokenizer = model_builder(cfg.train, cfg.model)
    pad_id = resolve_pad_token(tokenizer, model.llm)
    logger.info(f"Resolved pad_token_id: {pad_id}")

    # Move model to device
    model.to(device)

    # Freeze modules per config
    if cfg.train.freeze_llm:
        for p in model.llm.parameters():
            p.requires_grad = False

    # Optimizer only for projector
    projector_params = [p for p in model.projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(projector_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Dataset and DataLoader
    # FIXME: currently only support train dataset )
    split = cfg.data.get("train_split", cfg.data.get("test_split", "train"))

    train_ds = get_speech_dataset(cfg.data, tokenizer, split=split)
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=train_ds.collator,
        num_workers=cfg.train.num_workers,
        # FIXME: check if device is cuda or cuda:0 
        pin_memory=(device == "cuda"),
        persistent_workers=True if cfg.train.num_workers > 0 else False
    )

    logger.info(f"Train dataset: {len(train_ds)} samples, {len(train_dataloader)} batches")

    # Validation dataloader 
    val_split = cfg.data.get("val_split", "dev")
    vald_ds = get_speech_dataset(cfg.data, tokenizer, split=val_split)
    val_dataloader = DataLoader(
        vald_ds, 
        batch_size=cfg.train.val_batch_size,
        shuffle=False,
        collate_fn=vald_ds.collator,
        num_workers=cfg.train.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True if cfg.train.num_workers > 0 else False
    )

    logger.info(f"Validation dataset: {len(vald_ds)} samples, {len(val_dataloader)} batches")

    # Calculate total training steps HERE
    total_training_steps = cfg.train.num_epochs * len(train_dataloader)
    logger.info(f"Total training steps: {total_training_steps}")
    
    # Initializing the LR Scheduler
    if cfg.scheduler.get("enabled", True):
        num_warmup_steps = cfg.train.get("num_warmup_steps", 1000)
        num_cycles = cfg.train.get("num_cycles", 0.5)
        
        lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=num_cycles
            )
        logger.info(f"Using Cosine LR Scheduler with {num_warmup_steps} warmup steps and {num_cycles} cycles")

    # W&B init 
    run = None 
    if cfg.log.use_wandb:
        run = init_wandb(
            use_wand=True, 
            project=cfg.log.wandb_project_name,
            run_name=cfg.log.wandb_exp_name,
            tags=[cfg.model.llm_model_name, cfg.model.projector, "projector-only"],
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        logger.info("Initialized W&B run")

    # Mixed precision and scaler
    use_autocast = bool(cfg.train.mixed_precision and device == "cuda")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Grad Scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=bool(cfg.train.use_fp16))

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    best_val_wer = float("inf")
    start_time = time.time()
    model.train()

    logger.info("Starting training...")

    for epoch in range(cfg.train.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Incrementing global step 
            global_step += 1

            # Move batch to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            audio_mel = batch["audio_mel"].to(device, non_blocking=True)
            modality_mask = batch['modality_mask'].to(device, non_blocking=True)

            # Optimizer zero grad
            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type=device, dtype=amp_dtype): 
                    # Forward pass
                    outputs, metrics = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel,
                        modality_mask=modality_mask
                    )
                    loss = outputs.loss
                    
            else: 
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask
                )
                loss = outputs.loss

            # Accuracy for logging
            if metrics is not None and "acc" in metrics:
                acc = metrics["acc"]

            # Compute WER per batch 
            hyp_texts, ref_texts = decode_texts_from_outputs(
                logits=outputs.logits,
                labels=labels,
                tokenizer=tokenizer,
                ignore_label=-100
            )

            batch_wer = compute_wer(hyp_texts, ref_texts)

            # Word-level accuracy
            word_acc = 1.0 - batch_wer if batch_wer >= 0.0 else 0.0

            # Backward pass and optimization step
            if scaler.is_enabled(): 
                scaler.scale(loss).backward()
                if cfg.train.grad_clip is not None: 
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(projector_params, cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.grad_clip is not None: 
                    clip_grad_norm_(projector_params, cfg.train.grad_clip)
                optimizer.step()

            # Step the LR scheduler 
            if cfg.scheduler.get("enabled", True):
                lr_scheduler.step()

            if global_step % cfg.log.log_interval == 0: 
                elapsed = time.time() - start_time
                # Get current learning rate
                if cfg.scheduler.get("enabled", True):
                    current_lr = lr_scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch={epoch} | Step={global_step} | WER={batch_wer:.4f} | W_ACC={word_acc:.4f} | Loss={loss.item():.4f} | Acc={float(acc):.4f} | LR={current_lr:.6e} | Time={elapsed:.2f}s")
                if run is not None: 
                    run.log({
                        "train/wer": batch_wer,
                        "train/word_acc": word_acc,
                        "train/loss": loss.item(),
                        "train/acc": acc,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/time_elapsed": elapsed
                    }, step=global_step)
                # Reset start time
                start_time = time.time()

        # Validation at the end of each epoch
        val_loss, val_acc, val_wer_score, val_word_acc, all_hyp_texts, all_ref_texts = evaluate(cfg, model, val_dataloader, device, tokenizer=tokenizer)
        logger.info(f"Epoch {epoch} | Val WER: {val_wer_score:.4f} | Val Word Acc: {val_word_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if run is not None: 
            run.log({
                "val/wer": val_wer_score,
                "val/word_acc": val_word_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/epoch": epoch
            }, step=global_step)


        # Save hyp and ref texts for a few examples
        save_and_print_examples(
            hyp_texts=all_hyp_texts,
            ref_texts=all_ref_texts,
            output_path=cfg.train.output_dir,
            epoch=epoch,
            n_save=10,
            n_print=5,
            run=run,
            seed=cfg.train.seed
        )

        # Save best model
        if val_wer_score < best_val_wer:
            best_val_wer = val_wer_score
            best_val_path = os.path.join(cfg.train.output_dir, "projector_best_wer.pt")
            save_projector(model, best_val_path, global_step)
            logger.info(f"New best model saved at step {global_step} to {best_val_path} with val_wer {best_val_wer:.4f}")

        # Check early stopping
        # Only if enabled in config
        if cfg.early_stopping.get("enabled", True):
            if early_stop.check(val_loss):
                logger.info(f"Early stopping at epoch: {epoch} (patience={early_stop.patience})")
                break

    # Final model checkpoint (for reference)
    final_path = os.path.join(cfg.train.output_dir, "projector_final.pt")
    save_projector(model, final_path, global_step)
    logger.info(f"Final model checkpointed at: {final_path}")

    # End of training     
    logger.info("Training completed.....")
    logger.info("Training Time: {:.2f} minutes".format((time.time() - start_time) / 60))
    logger.info(f"Best validation wer: {best_val_wer:.4f}")
    logger.info(f"Final projector model saved to: {best_val_path}")

    if run is not None: 
        run.finish()


if __name__ == "__main__":

    # Debugging flag
    DEBUG = False

    if DEBUG:
        sys.argv = [
            'train.py',
            '--config', 'configs/test_config.yaml'
        ]

        main()
    else:
        main()

