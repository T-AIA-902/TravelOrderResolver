"""
Quantization utilities for QLoRA fine-tuning.

Provides configuration and loading utilities for 4-bit quantized models
using bitsandbytes and PEFT.
"""

from typing import Any, Dict, Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

DeviceType = Literal["auto", "cuda", "cpu"]


def get_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_use_double_quant: bool = False,
) -> Any:
    """
    Create a BitsAndBytesConfig for 4-bit quantization.

    Args:
        load_in_4bit: Whether to load model in 4-bit
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_compute_dtype: Compute dtype for 4-bit base models
        bnb_4bit_use_double_quant: Use nested quantization

    Returns:
        BitsAndBytesConfig instance
    """
    from transformers import BitsAndBytesConfig

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    task_type: str = "CAUSAL_LM",
    bias: str = "none",
) -> Any:
    """
    Create a LoRA configuration for PEFT.

    Args:
        r: LoRA rank (dimension of the low-rank matrices)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        task_type: Task type ("CAUSAL_LM", "SEQ_CLS", "TOKEN_CLS")
        bias: Bias type ("none", "all", "lora_only")

    Returns:
        LoraConfig instance
    """
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
    }

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type_map.get(task_type, TaskType.CAUSAL_LM),
    )


def load_quantized_model(
    model_name: str,
    model_type: Literal["causal_lm", "seq_cls"] = "causal_lm",
    num_labels: int = 3,
    quantization_config: Optional[Any] = None,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
) -> Any:
    """
    Load a quantized model for fine-tuning or inference.

    For pre-quantized models (e.g., from Unsloth), quantization_config
    can be None.

    Args:
        model_name: HuggingFace model name or path
        model_type: Type of model ("causal_lm" or "seq_cls")
        num_labels: Number of labels for sequence classification
        quantization_config: Optional BitsAndBytesConfig
        device_map: Device mapping strategy
        torch_dtype: Optional torch dtype
        trust_remote_code: Trust remote code in model

    Returns:
        Loaded model
    """
    if torch_dtype is None:
        torch_dtype = torch.float16

    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    if model_type == "seq_cls":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )

    return model


def load_tokenizer(
    model_name: str,
    padding_side: str = "right",
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """
    Load a tokenizer for a model.

    Args:
        model_name: HuggingFace model name or path
        padding_side: Side for padding ("left" or "right")
        trust_remote_code: Trust remote code

    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def prepare_model_for_training(model: Any) -> Any:
    """
    Prepare a quantized model for k-bit training.

    This enables gradient checkpointing and prepares the model
    for PEFT fine-tuning.

    Args:
        model: The loaded model

    Returns:
        Model prepared for training
    """
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    return model


def apply_lora(model: Any, lora_config: Any) -> Any:
    """
    Apply LoRA adapters to a model.

    Args:
        model: The model to apply LoRA to
        lora_config: LoRA configuration

    Returns:
        Model with LoRA adapters
    """
    from peft import get_peft_model

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def merge_and_save_lora(
    model: Any,
    tokenizer: AutoTokenizer,
    output_dir: str,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
) -> None:
    """
    Merge LoRA weights with base model and save.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        output_dir: Directory to save merged model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_id: Repository ID for Hub
    """
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub and hub_repo_id:
        merged_model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)

    print(f"Model saved to {output_dir}")
