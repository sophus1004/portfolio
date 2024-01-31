import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_dataset

from peft import LoraConfig, get_peft_model

sys_prompt = "아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요."
instruction_template = "\n\n### 명령어:\n"
response_template = "\n\n### 응답:\n"
prompt_input = sys_prompt + instruction_template + "{instruction}" + response_template + "{output}"

model_name_or_path = "EleutherAI/polyglot-ko-1.3b"
data_name_or_path = "maywell/ko_wikidata_QA"
train_ds = load_dataset(data_name_or_path)

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

def formatting_prompts_func(example, prompt_input):

    output_texts = list(map(lambda inst, out: prompt_input.format(instruction=inst, output=out), 
                            example['instruction'], 
                            example['output']))
    return output_texts

data_collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer, mlm=False
    )

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds['train'],
    max_seq_length=2048,
    formatting_func=lambda example: formatting_prompts_func(example, prompt_input=prompt_input),
    data_collator=data_collator
    )

trainer.train()