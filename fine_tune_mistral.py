import os
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import re
import torch
import transformers
import bitsandbytes as bnb
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser, Trainer,
                          TrainingArguments, logging, pipeline)



def main():
    #Importing the dataset from HUGGINGFACE
    dataset = load_dataset("berquetR/train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Create the prompt
    dataset = dataset.map(lambda example : propmt(example), remove_columns = ['source', 'original_source', 'source_links', 'target', 'optimal_choice'])

    #Import the model
    base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    new_model = "Enlighten_Instruct"

    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.bos_token, tokenizer.eos_token
    model = prepare_model_for_kbit_training(model)

    # Load PEFT model
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
    model = get_peft_model(model, peft_config)

    # Load the trainer
    training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    group_by_length=True,
    lr_scheduler_type="constant",
)
    # Setting sft parameters
    trainer = SFTTrainer( 
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
    
    trainer.train()

    trainer.model.save_pretrained(new_model)
    model.config.use_cache = True
    model.eval()

    # Save the model
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)
    model.config.save_pretrained(new_model)
    peft_config.save_pretrained(new_model)



def propmt (example):
    return {'text' : "<s>[INST]" + 'Given the current Wikipedia page we are on : '+ example['source'] + ' the original wikipedia page from where we started' + example['original_source'] + ' the proposed links to other Wikipedia pages are :' + example['source_links'] + ' In order to arrive the fastest, in terms of the number of Wikipedia pages visited, to the Wikipedia page :'+ example['target']+ ' On which link should we click on ?' "[/INST]" +  example["optimal_choice"] + '</s>'}
