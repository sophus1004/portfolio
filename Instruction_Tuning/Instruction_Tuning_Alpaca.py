import copy
import torch
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer, BitsAndBytesConfig

from datasets import Dataset, load_dataset

prompt_input = "아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{instruction}\n\n### 응답:\n"

model_name_or_path = "EleutherAI/polyglot-ko-1.3b"
data_name_or_path = "maywell/ko_wikidata_QA"
train_ds = load_dataset(data_name_or_path)['train']

use_lora = False # True or False
use_qlora = False # True or False
gradient_checkpointing=False # True or False
training_args = TrainingArguments(
    output_dir="./test",
    save_strategy="no",
    num_train_epochs=1,
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    optim="adamw_torch" if not use_qlora == True else "paged_adamw_8bit",
    learning_rate=1e-5,
    weight_decay=0,
    lr_scheduler_type="cosine",
    bf16=True
    )

if use_lora == True or use_qlora == True:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
        )

if use_qlora == True:
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float32 if not use_qlora == True else 'auto',
    quantization_config=bnb_config if use_qlora == True else None,
    device_map='auto'
    )

if gradient_checkpointing is True:
    model.enable_input_require_grads() if use_lora == True or use_qlora == True else None
    model.config.use_cache=False
    training_args.gradient_checkpointing=True
    training_args.gradient_checkpointing_kwargs={'use_reentrant':False}
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"]=True

if use_lora == True or use_qlora == True:
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if use_qlora == True:
        print(model.config.quantization_config)

else:
    print(model.config.torch_dtype)

def tokenize_fn(strings, tokenizer):
    
    tokenized_input = tokenizer(strings)
        
    input_ids_lens = labels_lens = [
        len(tokenized) for tokenized in tokenized_input['input_ids']
    ]

    return dict(
        input_ids=tokenized_input['input_ids'],
        labels=tokenized_input['input_ids'],
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def create_source_example(example):
    prompt_formatted = prompt_input.format(instruction=example['instruction'])
    return prompt_formatted, example['output'], prompt_formatted + example['output']

def process_labels(label, source_len):
    label[:source_len] = [IGNORE_INDEX] * source_len
    return label

IGNORE_INDEX = -100
sources = []
targets = []
examples = []

mapped_results = list(map(create_source_example, train_ds))
sources, targets, examples = zip(*mapped_results)

sources_tokenized = tokenize_fn(sources, tokenizer)
examples_tokenized = tokenize_fn(examples, tokenizer)

input_ids = examples_tokenized['input_ids']
labels = copy.deepcopy(input_ids)

labels = list(map(process_labels, labels, sources_tokenized['input_ids_lens']))

data_dict = dict(input_ids=input_ids, labels=labels)
input_train_ds = Dataset.from_dict(data_dict)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, return_tensors='pt')

trainer = Trainer(
    model,
    training_args,
    data_collator=data_collator,
    train_dataset=input_train_ds
    )

trainer.train()