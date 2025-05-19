""" Finetuning the seq2seq models on downstream tasks."""
import math
import os
import sys
import logging
import functools
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Union, List
import numpy as np

import torch
import transformers
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    T5ForConditionalGeneration,
    RobertaForSequenceClassification,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    set_seed,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    IA3Config,
    AdaLoraConfig,
    AdaDHPConfig,
    AdaDHPModel,
)

from src.tasks import AutoTask, num_labels_mapping
from src.metrics import TASK_TO_METRICS
from src.trainer import PEFTSeq2SeqTrainer
from src.postprocessors import AutoPostProcessor, PostProcessor
from AdaTrainer import AdaTrainer
from AdaSeq2seqTrainer import AdaptiveSeq2SeqTrainer

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    lang_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    add_prefix: bool = field(
        default=False,
        metadata={"help": "Whether add the prefix before each example, typically using the name of the dataset."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    truncate_head: bool = field(
        default=False, metadata={"help": "Truncate the head or tail of the sequence."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    split_validation_test: Optional[bool] = field(default=True,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    k_shot_example: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of examples to use for the k-shot learning."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    peft_model_id: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model using PEFT."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    prefix_length: int = field(
        default=100,
        metadata={"help": "Defines the length for prompt tuning."}
    )
    text_init: bool = field(
        default=True,
        metadata={"help": "Whether to use text initialization for prompt tuning or not."}
    )
    prompt_tuning_init_text: str = field(
        default="Classify this text is postive or not:",
        metadata={"help": "The text used for prompt tuning initialization."}
    )
    num_transformer_submodules: int = field(
        default=1,
        metadata={
            "help": "Set to 1 to add the prompt only to the encoder. Set to 2 to add the prompt to both the encoder and decoder."}
    )
    hidden_size: int = field(
        default=768,
        metadata={"help": "The hidden size of the model."}
    )
    peft_type: str = field(
        default="PROMPT_TUNING",
        metadata={"help": "PROMPT_TUNING."}
    )
    ###################################################ia3###################################################
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with (IA)³."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    feedforward_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or a regex expression of module names which are feedforward"
                    "For example, ['output.dense']"
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from (IA)^3 layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_ia3_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the vectors in the (IA)^3 layers."},
    )
    #####################################################################################################################
    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Initial Lora matrix dimension."})
    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    deltaT: int = field(default=1, metadata={"help": "Step interval of rank allocation."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    orth_reg_weight: float = field(default=0.5, metadata={"help": "The orthogonal regularization coefficient."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    rank_pattern: Optional[dict] = field(default=None, metadata={"help": "The saved rank pattern."})
    #####################################################################################################################
    target_num: int = field(default=96, metadata={"help": "Target trainable parameters."})


@dataclass
class DynamicTrainingArguments(Seq2SeqTrainingArguments):
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Whether to use generate to get the predictions."}
    )
    generation_max_length: Optional[int] = field(
        default=20,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=1, metadata={"help": "Number of beams to use for evaluation."})
    save_lora_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to save the lora embeddings."}
    )
    load_lora_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to load the lora embeddings."}
    )
    load_lora_embedding_B: bool = field(
        default=True,
        metadata={"help": "Whether to load the lora embedding B, which is initialized from zeros."},
    )
    quantization: bool = field(
        default=False,
        metadata={"help": "Whether to quantize the model."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(training_args.output_dir, 'output.log'), mode='w'),
                  logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"\n\nModel parameters {model_args}")
    logger.info(f"\n\nData parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_args.seed = training_args.seed
    training_args.metric_for_best_model = TASK_TO_METRICS[data_args.task_name][0]
    training_args.greater_is_better = True

    # Preprocessing datasets
    def encoder_preprocess_function(examples, max_target_length=None, task_id=None):
        model_inputs = tokenizer(examples['source'],
                                 max_length=data_args.max_seq_length,
                                 padding=padding,
                                 truncation=True)
        # Setup the tokenizer for targets
        labels = torch.tensor([int(i) for i in examples['target']])
        model_inputs["labels"] = labels
        return model_inputs

    def seq2seq_preprocess_function(examples, max_target_length=None, task_id=None):
        model_inputs = tokenizer(examples['source'],
                                 max_length=data_args.max_seq_length,
                                 padding=padding,
                                 truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def decoder_preprocess_function(examples, max_target_length=None, task_id=None):
        batch_size = len(examples['source'])
        inputs = [f"{x} Label : " for x in examples['source']]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(examples['target'])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    data_args.max_seq_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (data_args.max_seq_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (data_args.max_seq_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:data_args.max_seq_length])
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:data_args.max_seq_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:data_args.max_seq_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def decoder_test_preprocess_function(examples):
        batch_size = len(examples['source'])
        inputs = [f"{x} Label : " for x in examples['source']]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(examples['target'])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    data_args.max_seq_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (data_args.max_seq_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (data_args.max_seq_length - len(label_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:data_args.max_seq_length])
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:data_args.max_seq_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:data_args.max_seq_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics_encoder(p: EvalPrediction, processor: PostProcessor, metrics: Callable):
        preds, labels = p
        num_logits = preds.shape[-1]
        if num_logits == 1:
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
        result = {}
        for metric in metrics:
            result.update(metric(preds, labels))
        return result

    def compute_metrics_seq2seq(p: EvalPrediction, processor: PostProcessor, metrics: Callable):
        # preds, labels, data_info = p
        preds, labels = p
        decoded_preds, decoded_labels = processor.process(preds, labels, data_info=None)
        result = {}
        for metric in metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    def compute_metrics_decoder(p: EvalPrediction, processor: PostProcessor, metrics: Callable):
        # preds, labels, data_info = p
        output_sequences, labels = p
        preds = output_sequences[:, data_args.max_seq_length:]
        # output_sequences_decode = []
        # for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #     print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        #     # args.stop_token = tokenizer.eos_token
        #     generated_sequence = generated_sequence.tolist()

        #     # Decode text
        #     text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        #     output_sequences_decode.append(text)
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds, decoded_labels = processor.process(preds, labels, data_info=None)
        result = {}
        for metric in metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    if any(x in model_args.model_name_or_path for x in ["bert", "roberta", "albert"]):
        logger.info(f"\n\nLoading enocder model from {model_args.model_name_or_path}.\n\n")
        task_type = TaskType.SEQ_CLS
        preprocess_function = encoder_preprocess_function
        metrics_fn = compute_metrics_encoder
    elif any(x in model_args.model_name_or_path for x in ["t5"]):
        logger.info(f"\n\nLoading seq2seq model from {model_args.model_name_or_path}.\n\n")
        task_type = TaskType.SEQ_2_SEQ_LM
        preprocess_function = seq2seq_preprocess_function
        metrics_fn = compute_metrics_seq2seq
    elif any(x in model_args.model_name_or_path for x in ["gpt", "llama"]):
        logger.info(f"\n\nLoading decoder model from {model_args.model_name_or_path}.\n\n")
        task_type = TaskType.CAUSAL_LM
        preprocess_function = decoder_preprocess_function
        metrics_fn = compute_metrics_decoder
        training_args.generation_max_length = data_args.max_seq_length + training_args.generation_max_length
    else:
        raise NotImplementedError

    if model_args.peft_model_id:
        logger.info(f"\n\nLoading model {model_args.peft_model_id} for prompt tuning.\n\n")
        peft_config = PeftConfig.from_pretrained(model_args.peft_model_id)
        # peft_config.load_lora_embeddings=training_args.load_lora_embeddings
        # peft_config.load_lora_embedding_B=training_args.load_lora_embedding_B
        if task_type == TaskType.SEQ_CLS:
            model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
        elif task_type == TaskType.SEQ_2_SEQ_LM:
            model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
        elif task_type == TaskType.CAUSAL_LM and "gpt" in model_args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        elif task_type == TaskType.CAUSAL_LM and "llama" in model_args.model_name_or_path:
            # model = LlamaForCausalLM.from_pretrained(
            #     model_args.model_name_or_path,
            #     return_dict=True,
            #     load_in_8bit=training_args.quantization,
            #     device_map="auto",
            #     low_cpu_mem_usage=True,
            # )
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                return_dict=True,
                load_in_8bit=True if training_args.quantization else False,
                device_map="auto",
                # low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            model_args.hidden_size = model.config.hidden_size
        model = PeftModel.from_pretrained(model, model_args.peft_model_id)
        model.peft_config[model.active_adapter].inference_mode = False
    else:
        if task_type == TaskType.SEQ_CLS:
            model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)
        elif task_type == TaskType.SEQ_2_SEQ_LM:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
        elif task_type == TaskType.CAUSAL_LM and "gpt" in model_args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
        elif task_type == TaskType.CAUSAL_LM and "llama" in model_args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                return_dict=True,
                load_in_8bit=training_args.quantization,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
        model_args.hidden_size = model.config.hidden_size
        if model_args.peft_type == "PROMPT_TUNING":
            logger.info(f"\n\nLoading model for prompt tuning.\n\n")
            peft_config = PromptTuningConfig(
                task_type=task_type,
                prompt_tuning_init=PromptTuningInit.TEXT if model_args.text_init else PromptTuningInit.RANDOM,
                num_virtual_tokens=model_args.prefix_length,
                num_transformer_submodules=model_args.num_transformer_submodules,
                prompt_tuning_init_text=model_args.prompt_tuning_init_text,
                tokenizer_name_or_path=model_args.model_name_or_path,
            )
        elif model_args.peft_type == "IA3":
            logger.info(f"\n\nLoading model for ia3.\n\n")
            peft_config = IA3Config(
                task_type=task_type,
                target_modules=None,
                feedforward_modules=None,
                modules_to_save=None
            )
        elif model_args.peft_type == "ADALORA":
            logger.info(f"\n\nLoading model for adalora.\n\n")
            peft_config = AdaLoraConfig(
                task_type=task_type,
                target_r=model_args.target_r,
                init_r=model_args.init_r,
                target_modules=model_args.target_modules,
                tinit=model_args.tinit,
                tfinal=model_args.tfinal,
                deltaT=model_args.deltaT,
                beta1=model_args.beta1,
                beta2=model_args.beta2,
                total_step=model_args.total_step,
            )
        elif model_args.peft_type == "ADADHP":
            logger.info(f"\n\nLoading model for adadhp.\n\n")
            peft_config = AdaDHPConfig(
                task_type=task_type,
                target_num=model_args.target_num,
                tinit=model_args.tinit,
                tfinal=model_args.tfinal,
                deltaT=model_args.deltaT,
                beta1=model_args.beta1,
                beta2=model_args.beta2,
                total_step=model_args.total_step,
                target_modules=model_args.target_modules
            )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info(model)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # TODO: check if this is correct
    padding = "max_length" if data_args.pad_to_max_length else False

    column_names = ['source', 'target']
    if training_args.do_train:
        # Load datasets from files if your target datasets are not in huggingface datasets.
        data_processor = AutoTask.get(data_args.task_name, data_args.dataset_config_name, seed=42)
        max_target_length = data_processor.get_max_target_length(tokenizer=tokenizer,
                                                                 default_max_length=training_args.generation_max_length)
        train_dataset = data_processor.get(
            split="train",
            split_validation_test=data_args.split_validation_test,
            add_prefix=True if data_args.add_prefix else False,
            n_obs=data_args.max_train_samples,
            lang=data_args.lang_name,
            file_name=data_args.train_file)
        if data_args.task_name == "yelp_polarity":
            train_dataset = train_dataset.select(list(range(100000)))
        if data_args.k_shot_example is not None:
            logger.info(f"\n\nUsing Seed {training_args.seed} for sampling.\n")
            logger.info(f"\nUsing {data_args.k_shot_example} examples for training.\n\n")
            class_num_dct = num_labels_mapping[data_args.task_name]
            num_example_per_class = data_args.k_shot_example // len(class_num_dct)
            shuffled_train_dataset = train_dataset.shuffle(seed=training_args.seed)
            index_lst = []
            for i, data in enumerate(shuffled_train_dataset):
                if sum(class_num_dct.values()) == data_args.k_shot_example:
                    break
                label = data["target"]
                if data_args.task_name == "stsb":
                    label = "0" if float(label) <= 2.5 else "1"
                if class_num_dct[label] < num_example_per_class or sum(
                        class_num_dct.values()) == data_args.k_shot_example - 1:
                    class_num_dct[label] += 1
                    index_lst.append(i)
            train_dataset = shuffled_train_dataset.select(index_lst)
        train_dataset = train_dataset.map(
            functools.partial(preprocess_function, max_target_length=max_target_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        data_processor = AutoTask.get(data_args.task_name, data_args.dataset_config_name, seed=42)
        max_target_length = data_processor.get_max_target_length(tokenizer=tokenizer,
                                                                 default_max_length=training_args.generation_max_length)
        eval_dataset = data_processor.get(
            split="validation",
            split_validation_test=data_args.split_validation_test,
            add_prefix=True if data_args.add_prefix else False,
            n_obs=data_args.max_eval_samples,
            lang=data_args.lang_name,
            file_name=data_args.validation_file)
        eval_dataset = eval_dataset.map(
            decoder_test_preprocess_function if task_type == TaskType.CAUSAL_LM else functools.partial(
                preprocess_function, max_target_length=max_target_length),
            # functools.partial(preprocess_function, max_target_length=max_target_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_predict:
        data_processor = AutoTask.get(data_args.task_name, data_args.dataset_config_name, seed=42)
        max_target_length = data_processor.get_max_target_length(tokenizer=tokenizer,
                                                                 default_max_length=training_args.generation_max_length)
        test_dataset = data_processor.get(
            split="test",
            split_validation_test=data_args.split_validation_test,
            add_prefix=True if data_args.add_prefix else False,
            n_obs=data_args.max_predict_samples,
            lang=data_args.lang_name,
            file_name=data_args.test_file)
        test_dataset = test_dataset.map(
            decoder_test_preprocess_function if task_type == TaskType.CAUSAL_LM else functools.partial(
                preprocess_function, max_target_length=max_target_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Get the metric function
    eval_metrics = AutoTask.get(data_args.task_name, data_args.dataset_config_name).metric
    post_processor = AutoPostProcessor.get(data_args.task_name, tokenizer, data_args.ignore_pad_token_for_loss)

    logger.info(f"\nUsing the consistent learning rate for all parameters\n\n")
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-8)
    # 这个地方算的有点问题
    # num_training_steps = len(train_dataset) * training_args.num_train_epochs // (
    #         training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) if training_args.max_steps == -1 else training_args.max_steps

    num_training_steps = math.ceil(
        len(train_dataset) / (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)) \
                         * training_args.num_train_epochs if training_args.max_steps == -1 else training_args.max_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps
    )

    # Initialize our Trainer
    if task_type == TaskType.SEQ_CLS:
        logger.info(f"\nUsing the default trainer for {task_type}\n\n")
        trainer = AdaTrainer(
            model=model,
            args=training_args,
            data_collator=default_data_collator,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            compute_metrics=functools.partial(metrics_fn, processor=post_processor, metrics=eval_metrics),
        )
    elif task_type == TaskType.SEQ_2_SEQ_LM or task_type == TaskType.CAUSAL_LM:
        logger.info(f"\nUsing the PEFTSeq2SeqTrainer for {task_type}\n\n")
        trainer = AdaptiveSeq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=default_data_collator,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            compute_metrics=functools.partial(metrics_fn, processor=post_processor, metrics=eval_metrics),
        )
    else:
        raise NotImplementedError

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_predict:
        logger.info("*** Predict ***")
        test_output = trainer.predict(test_dataset)
        test_metrics = test_output.metrics
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "nlp tasks"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = data_args.task_name
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = data_args.task_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
