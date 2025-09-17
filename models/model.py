import os 
import types 
import torch 
import soundfile as sf
import torch.nn as nn
import logging 
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig


from utils.metrics import compute_accuracy
from utils.train_utils import print_model_size, print_module_size

logger = logging.getLogger(__name__)


def model_builder(train_config, model_config, **kwargs):
    """"""
    # 1. tokenizer
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    # 2. llm 
    llm = setup_llm(train_config, model_config, **kwargs)

    # 3. projector 
    projector = setup_projector(train_config, model_config, **kwargs).to(torch.bfloat16)
    #encoder_projector = setup_encoder_projector(train_config, model_config, **kwargs) # fp32

    # 2. model
    model = ASRLLM(
        llm,
        projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs
    )

    # load ckpt 
    ckpt_path = kwargs.get("ckpt_path", None) 
    # TODO: check models is loading correctly
    if ckpt_path is not None:
        logger.info(f"Load checkpoint from {ckpt_path}")
        ckpt_dir = torch.load(ckpt_path, map_location="cpu")
        model.projector.load_state_dict(ckpt_dir['projector'], strict=True)
    
    print_model_size(model, train_config)

    return model, tokenizer


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.llm_model
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_llm(train_config, model_config, **kwargs):
    
    model = AutoModelForCausalLM.from_pretrained(
            model_config.llm_model,
            torch_dtype=torch.bfloat16 if train_config.mixed_precision else torch.float32,
            attn_implementation="sdpa",
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
    )

    print_module_size(model, model_config.llm_model_name)

    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm: 
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()

     # TODO: No PEFT

    return model


def setup_projector(train_config, model_config, **kwargs):
    from models.projector import MelProjectorLinear
    projector = MelProjectorLinear(model_config)
    print_module_size(projector, "mel-linear")
    return projector


class ASRLLM(nn.Module):
    
    def __init__(self,
                 llm: nn.Module,
                 projector: Optional[nn.Module],
                 tokenizer,
                 train_config,
                 model_config,
                 **kwargs
    ):
        super().__init__()

        # llm 
        self.llm = llm

        # projector
        self.projector = projector

        # tokenizer
        self.tokenizer = tokenizer

        self.train_config = train_config
        self.model_config = model_config


    @staticmethod
    def _masked_next_token_accuracy(logits: torch.FloatTensor,
                                labels: torch.LongTensor,
                                ignore_label: int = -100,
                                return_counts: bool = True):
        """
        Compute masked next-token accuracy with standard causal shift:
        compare logits at t vs labels at t+1, ignoring positions == ignore_label.
        """
        # logits: [B, T, V], labels: [B, T]
        preds = logits.argmax(dim=-1)                # [B, T]
        preds = preds[:, :-1]                        # [B, T-1]
        tgt = labels[:, 1:]                          # [B, T-1]
        mask = tgt.ne(ignore_label)                  # [B, T-1]
        if mask.sum().item() == 0:
            return (preds.new_tensor(0.0), 0, 0) if return_counts else preds.new_tensor(0.0)
        num_correct = (preds.eq(tgt) & mask).sum()
        denom = mask.sum()
        acc = num_correct.float() / denom.float()
        return (acc, num_correct.item(), denom.item()) if return_counts else acc


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        
        audio_mel = kwargs.get("audio_mel", None)
        if audio_mel is None: 
            raise ValueError("audio_mel is required for direct spectorgram pipeline")

        # 1. Projector 
        with torch.no_grad() if getattr(self.train_config, "freeze_llm", False) else torch.enable_grad():
            enc_tok = self.projector(audio_mel)  # [B, T_a, D_llm]

        # 2. Token embedding 
        token_embeds = None 
        if input_ids is not None: 
            # Santize any placeholder ids for embedding lookup
            input_ids = input_ids.clone()
            input_ids[input_ids == -1] = 0
    
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, "embed_tokens"):
                token_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "embed_tokens"):
                token_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                token_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Cast to LLM dtype (e.g., bfloat16)
        llm_dtype = next(self.llm.parameters()).dtype
        if enc_tok.dtype != llm_dtype:
            enc_tok = enc_tok.to(llm_dtype)
        if token_embeds is not None and token_embeds.dtype != llm_dtype:
            token_embeds = token_embeds.to(llm_dtype)

        # 4. Concat encoder feature and token embeddings (audio prefix + text token embeddings)
        # Concatenate audio tokens + prompt tokens
        if token_embeds is not None:
            inputs_embeds = torch.cat([enc_tok, token_embeds], dim=1)  # [B, T_a + T_text, D]
        else:
            inputs_embeds = enc_tok

        # 5. Build attention mask aligned with inputs_embeds
        # Build attention mask
        B, T_total, _ = inputs_embeds.size()
        T_audio = enc_tok.size(1)
        if attention_mask is not None and input_ids is not None:
            audio_attn = torch.ones((B, T_audio), dtype=attention_mask.dtype, device=inputs_embeds.device)
            attention_mask = torch.cat([audio_attn, attention_mask], dim=1)
        else:
            attention_mask = torch.ones((B, T_total), dtype=torch.long, device=inputs_embeds.device)

        # 6. Labels: ignore loss on audio prefix (mask with ignore_index e.g: -100)
        if labels is not None: 
            if labels.dim() != 2: 
                raise ValueError("Labels should be of shape 2D (B, T_text) for cross-entropy loss.")
            ignore_pad = torch.full((B, T_audio), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([ignore_pad, labels], dim=1) # [B, T_audio + T_text]

        # Fast path for generation setup
        if kwargs.get("inference_mode", False): 
            return inputs_embeds, attention_mask
        
        # Forward through LLM using inputs_embeds only 
        llm_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

        model_outputs = self.llm(**llm_kwargs)

        # Metrics
        metrics = None 
        if labels is not None and hasattr(model_outputs, "logits"):
            with torch.no_grad():
                acc, num_correct, denom = self._masked_next_token_accuracy(
                    model_outputs.logits, labels, ignore_label=-100, return_counts=True
                )
            metrics = {"acc": float(acc.item()), "num_correct": num_correct, "num_total": denom}

        return model_outputs, metrics
    
    @torch.no_grad()
    def inference(self, audio_mel: torch.Tensor, prompt: str = "", max_new_tokens: int = 64,
                  temperature: float = 0.7, top_p: float = 0.9, num_beams: int = 1,
                  do_sample: bool = None, device: str = None, **gen_kwargs):
        if device is None:
            device = next(self.parameters()).device
        llm_dtype = next(self.llm.parameters()).dtype
        if audio_mel.dim() == 2:
            audio_mel = audio_mel.unsqueeze(0)
        audio_mel = audio_mel.to(device, dtype=llm_dtype)

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)  # plain prompt for base Llama
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        inputs_embeds, attn_mask = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, audio_mel=audio_mel, inference_mode=True
        )

        if do_sample is None:
            do_sample = num_beams == 1 and temperature > 0.0

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams if not do_sample else 1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **gen_kwargs
        )
        # When passing inputs_embeds, return is only newly generated tokens
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        if audio_mel.size(0) == 1:
            texts = texts
        return texts