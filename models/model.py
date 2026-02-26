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


from utils.metrics import compute_accuracy, compute_wer, decode_texts_from_outputs
from utils.train_utils import print_model_size, print_module_size

logger = logging.getLogger(__name__)


def model_builder(train_config, model_config, **kwargs):
    """
    Build ASR-LLM model components.

    Model components:
        1. Tokenizer
        2. Projector
        3. LLM
        4. ASR-LLM model

    Args: 
        - train_config: training configuration
        - model_config: model configuration
        - kwargs: additional arguments
    
    Returns:
        - model: ASR-LLM model
        - tokenizer: tokenizer for LLM
    """
    # 1. tokenizer
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    # 2. projector
    projector = setup_projector(train_config, model_config, **kwargs)


    # 3. llm 
    llm = setup_llm(train_config, model_config, **kwargs)

    
    # 4. model
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
    if model_config.projector == "cnn-fuyu-v2":
        from models.projector import CNNFuyuProjectorV2
        projector = CNNFuyuProjectorV2(model_config)
    elif model_config.projector == "cnn-fuyu":
        from models.projector import CNNFuyuProjector
        projector = CNNFuyuProjector(model_config)
    elif model_config.projector == "fuyu":
        from models.projector import FuyuAudioProjector
        projector = FuyuAudioProjector(model_config)
    elif model_config.projector == "linear":
        from models.projector import MelProjectorConcat
        projector = MelProjectorConcat(model_config)
    elif model_config.projector == "patched-linear":
        from models.projector import PatchedLinearProjectorV1
        projector = PatchedLinearProjectorV1(model_config)
    elif model_config.projector == "patched-linear-v2":
        from models.projector import PatchedLinearProjectorV2
        projector = PatchedLinearProjectorV2(model_config)
    elif model_config.projector == "context-aware":
        from models.projector import ContextAwareProjector
        projector = ContextAwareProjector(model_config)
    elif model_config.projector == "context-aware-glu":
        from models.projector import ContextAwareProjectorGLU
        projector = ContextAwareProjectorGLU(model_config)
    elif model_config.projector == "cov1d-linear":
        from models.projector import EncoderProjectorCov1d
        projector = EncoderProjectorCov1d(model_config)
    elif model_config.projector == "q-former":
        from models.projector import EncoderProjectorQFormer
        projector = EncoderProjectorQFormer(model_config)
    else:
        return None
    print_module_size(projector, model_config.projector)
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
        self.dataset_config = kwargs.get("data_config", None)
        
        # Initialize metric flag for accuracy computation
        self.metric = kwargs.get("metric", True)  # Default to True if not specified

        # Label smoothing for better generalization
        self.label_smoothing = getattr(model_config, "label_smoothing", 0.0)


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
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # optional, downsampled mask for whisper
        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)

        modality_mask = kwargs.get("modality_mask", None)


        # Create post mask if not provided
        if audio_mel_post_mask is None:
            audio_mel_post_mask = torch.ones(audio_mel.size()[:-1], dtype=torch.long, device=audio_mel.device)

        # 2. Projector (Project to LLM embedding space)
        if self.model_config.projector == "q-former":
            # Q-former
            encoder_outputs = self.projector(audio_mel, audio_mel_post_mask) # [B, T_enc_proj, D_llm]

        elif self.model_config.projector in ["linear", "patched-linear", "patched-linear-v2", "context-aware", "context-aware-glu", "cov1d-linear", "fuyu", "cnn-fuyu", "cnn-fuyu-v2"]:
            # linear or conv1d + linear
            encoder_outputs = self.projector(audio_mel)  # [B, T_enc_proj, D_llm]


        # 3. Token embedding 
        if input_ids is not None: 
            # Santize any placeholder ids for embedding lookup
            input_ids[input_ids == -1] = 0
    
            # Resolve embedding layer across model architectures
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, "embed_tokens"):
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "embed_tokens"):
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        
        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outputs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outputs.shape[0]):
                encoder_outs_pad[
                    i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                ] = encoder_outputs[i][:modality_lengths[i]]
            
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

        # Fast path for generation setup
        if kwargs.get("inference_mode", False): 
            return inputs_embeds, attention_mask
        
        # Default path for training / evaluation
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        # Override loss with label smoothing if configured
        if self.label_smoothing > 0 and labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)
            shift_logits = model_outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            model_outputs.loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Metrics
        metrics = {}
        if self.metric:
            with torch.no_grad():
                # Compute token accuracy
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)
                metrics["acc"] = float(acc.item())

                # Compute WER
                if hasattr(model_outputs, "logits"):
                    # First decode the texts using decode_texts_from_outputs
                    hyp_texts, ref_texts = decode_texts_from_outputs(
                        logits=model_outputs.logits,
                        labels=labels,
                        tokenizer=self.tokenizer,
                        ignore_label=-100
                    )
                    
                    # Then compute WER using the decoded texts
                    wer_score = compute_wer(
                        hyp_texts=hyp_texts,
                        ref_texts=ref_texts
                    )
                    metrics["wer"] = float(wer_score)

        return model_outputs, metrics
    
    @torch.no_grad()
    def generate(self,
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
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            # max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return model_outputs
    
    @torch.no_grad()
    def inference(
        self,
        audio_path=None,
        prompt=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        # inference for asr model

        device = kwargs.get("device", "cuda")
        if os.path.exists(audio_path):  # Audio-Text QA
            import whisper

            audio_raw = whisper.load_audio(audio_path)
            audio_raw = whisper.pad_or_trim(audio_raw)

            mel_size = getattr(
                self.dataset_config, "mel_size", 80
            )  # 80 for large v1 and v2, 128 for large v3
            audio_mel = (
                whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
                .permute(1, 0)[None, :, :]
                .to(device)
            )

            encoder_outs = self.encoder.extract_variable_length_features(
                audio_mel.permute(0, 2, 1)
            )

            
            projector_dtype = next(self.projector.parameters()).dtype
            encoder_outs = encoder_outs.to(projector_dtype)

            if self.model_config.projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = self.projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.projector == "linear":
                encoder_outs = self.projector(encoder_outs)
        else:  # Text QA
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            ).to(device)

        prompt = "USER: {}\n ASSISTANT:".format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)

        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat(
            (encoder_outs, inputs_embeds[None, :, :]), dim=1
        )  # [audio,prompt]

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            inputs_embeds.device
        )

        # generate
        model_outputs = self.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        # Decode output ids to text
        output_text = self.tokenizer.decode(
            model_outputs[0], skip_special_tokens=True
        )

        return output_text