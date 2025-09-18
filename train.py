import os 
import time 
import argparse 

import torch 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from omegaconf import OmegaConf

# internal imports
import warnings
from utils.utils import set_seed, get_device, resolve_pad_token, ensure_dir, save_projector
from utils.log_config import get_logger
from utils.wand_config import init_wandb
from models.model import model_builder
from datamodule.dataset import get_speech_dataset
from torchtnt.utils.early_stop_checker import EarlyStopChecker
#from torch.optim.lr_scheduler import ReduceLROnPlateau

# warnings
warnings.filterwarnings(
    "ignore",
    message=".*will be changed to use torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)


def evaluate(cfg, model, dataloader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    num_correct, num_total = 0, 0

    use_autocast = bool(cfg.train.mixed_precision and getattr(device, "type", str(device)) == "cuda")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            audio_mel = batch["audio_mel"].to(device)
            if use_autocast:
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
                    outputs, metrics = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel,
                    )
            else:
                outputs, metrics = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel,
                )
            total_loss += outputs.loss.item()
            n_batches += 1
            if metrics is not None:
                num_correct += metrics.get("num_correct", 0)
                num_total  += metrics.get("num_total", 0)

    val_loss = total_loss / max(n_batches, 1)
    val_acc = num_correct / max(num_total, 1) if num_total > 0 else 0.0

    model.train()
    return val_loss, val_acc


# Early Stopping 
early_stop = EarlyStopChecker(
    mode="min",
    patience=2,
    min_delta=0.001, 
    threshold_mode="abs"
)

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
    train_split = cfg.data.get("train_split", cfg.data.get("test_split", "train"))

    train_ds = get_speech_dataset(cfg.data, tokenizer, split=train_split)
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

    # W&B init 
    run = None 
    if cfg.log.use_wandb:
        run = init_wandb(
            use_wand=True, 
            project=cfg.log.wandb_project_name,
            run_name=cfg.log.wandb_exp_name,
            tags=[cfg.model.llm_model_name, cfg.model.projector, "patched-linear-projector"],
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        logger.info("Initialized W&B run")

    # Mixed precision and scaler
    use_autocast = bool(cfg.train.mixed_precision and getattr(device, "type", str(device)) == "cuda")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Grad Scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=bool(cfg.train.use_fp16))

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
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

            # Optimizer zero grad
            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast): 
                    # Forward pass
                    outputs, metrics = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel
                    )
                    loss = outputs.loss
                    
            else: 
                # Forward pass
                outputs, metrics= model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel
                )
                loss = outputs.loss

            # Accuracy for logging
            if metrics is not None and "acc" in metrics:
                acc = float(metrics["acc"])

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

            if global_step % cfg.log.log_interval == 0: 
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch={epoch} Step={global_step} Loss={loss.item():.4f} Acc={float(acc):.4f} LR={lr:.6e} Time={elapsed:.2f}s")
                if run is not None: 
                    run.log({
                        "train/loss": loss.item(),
                        "train/acc": acc,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/time_elapsed": elapsed
                    }, step=global_step)
                # Reset start time
                start_time = time.time()


        # Validation at the end of each epoch
        val_loss, val_acc = evaluate(cfg, model, val_dataloader, device)
        logger.info(f"Epoch {epoch} Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        if run is not None: 
            run.log({
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/epoch": epoch
            }, step=global_step)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_path = os.path.join(cfg.train.output_dir, "projector_best.pt")
            save_projector(model, best_val_path, global_step)
            logger.info(f"New best model saved at step {global_step} to {best_val_path} with val_loss {best_val_loss:.4f}")

        # early stopping 
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
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final projector model saved to: {best_val_path}")

    if run is not None: 
        run.finish()


if __name__ == "__main__":
    main()