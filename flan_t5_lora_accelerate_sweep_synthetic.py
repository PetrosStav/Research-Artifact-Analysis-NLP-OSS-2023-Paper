import os
import torch
import yaml
import argparse
import wandb

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm

def parse_args():
    ##############################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", default="./yaml_files/config_flan_t5_base_lora_sweep_synthetic.yaml",type=str, help="The path where the yaml config file exists", required=False)
    args = parser.parse_args()
    return args
    ##############################################

# parse args
args = parse_args()

# load yaml
yaml_path = args.yaml_path
yaml_file = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)
########################################
# variables
model_name_or_path = yaml_file['model_name']['value']
use_wandb = yaml_file['log_bool']['value']
output_dir = yaml_file['output_dir']['value']
checkpoint_name = yaml_file['checkpoint_name']['value']
########################################

# defines
########################################
accelerator = Accelerator(gradient_accumulation_steps=64)
########################################
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, cache_dir="./models_cache")


def build_model(r, lora_alpha, lora_dropout):
    ########################################
    # init peft
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout, 
        target_modules=['q', 'v'] # this adds lora adapters only in the q, v of the attention module
    )
    ########################################
    # init t5 model
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir="./models_cache")
    model = get_peft_model(model, peft_config)
    accelerator.print(model.print_trainable_parameters())
    return model, peft_config


def build_dataset(max_length, batch_size, model, train_data, val_data):
    ########################################
    data = load_dataset("json", data_files={"train": train_data, "val": val_data})
    ########################################
    def preprocess_function(data_points, padding="max_length"):
        # tokenize inputs
        model_inputs = tokenizer(data_points['input'], max_length=max_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=data_points['output'], max_length=max_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    ########################################
    with accelerator.main_process_first():
        train_data = data["train"].shuffle(seed=1993).map(
                preprocess_function,
                batched=True,
                num_proc=1,
                load_from_cache_file=False,
                remove_columns=data["train"].features,
                desc="Running tokenizer on dataset",
        )
        val_data = data["val"].map(
                preprocess_function,
                batched=True,
                num_proc=1,
                load_from_cache_file=False,
                remove_columns=data["val"].features,
                desc="Running tokenizer on dataset",
        )
    ########################################
    data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model)
    train_dataloader = DataLoader(
        train_data, shuffle=True, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        val_data, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )
    return train_dataloader, eval_dataloader


def build_optimizer(model_parameters, num_of_instances, lr, num_epochs):
    optimizer = torch.optim.AdamW(model_parameters, lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(num_of_instances * num_epochs),
    )
    return optimizer, lr_scheduler


def train(config=None):
    global yaml_file
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        my_model, my_peft_config = build_model(config.r, config.lora_alpha, config.lora_dropout)
        train_loader, dev_loader = build_dataset(
            yaml_file['max_length']['value'], 
            yaml_file['batch_size']['value'], 
            my_model,
            yaml_file['train_data']['value'],
            yaml_file['val_data']['value']
        )
        my_optimizer, my_lr_scheduler = build_optimizer(my_model.parameters(), len(train_loader), config.learning_rate, config.epochs)
        
        # early stopping parameters
        patience = 3  # number of epochs to wait for improvement before stopping
        min_delta = 0.0001  # minimum improvement to qualify as an improvement
        best_eval_loss = 999
        no_improvement = 0
        
        ########################################
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            my_model, train_loader, dev_loader, my_optimizer, my_lr_scheduler
        )
        accelerator.print(model)
        # accelerator.state.deepspeed_plugin.zero_stage == 3
        # accelerator.state.num_processes = torch.cuda.device_count()
        ########################################

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            ########################################
            # train
            for batch in tqdm(train_dataloader, desc=f"Training for epoch: {epoch}"):
                outputs = model(**batch)
                loss = outputs.loss
                if use_wandb:
                    wandb.log({"batch loss": loss.item()})
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            accelerator.wait_for_everyone()
            model.eval()
            eval_loss = 0
            ########################################
            # eval
            for batch in tqdm(eval_dataloader, desc=f"Evaluating for epoch: {epoch}"):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            ########################################
            eval_epoch_loss = eval_loss / len(eval_dataloader)

            print('Saving model...')

            accelerator.wait_for_everyone()
            ########################################
            # saving model
            peft_model_id = f"{checkpoint_name}_{my_peft_config.peft_type}_{my_peft_config.task_type}_{wandb.run.name}_{config.r}_{config.lora_alpha}_{config.lora_dropout}_{epoch}"
            model.save_pretrained(f"{output_dir}/{peft_model_id}")
            ########################################

            # check for improvement
            if best_eval_loss is None or eval_epoch_loss < best_eval_loss - min_delta:
                best_eval_loss = eval_epoch_loss
                no_improvement = 0
            else:
                no_improvement += 1
            
            print(f"Epoch: {epoch}, Train Loss: {total_loss/len(train_dataloader)}", f"Eval Loss: {eval_epoch_loss}", f"Best Eval Loss: {best_eval_loss}")

            if use_wandb:
                wandb.log({"Epoch": epoch, "train_loss": total_loss/len(train_dataloader), "eval_loss": eval_epoch_loss, "best_eval_loss": best_eval_loss})
            
            if no_improvement >= patience:
                print("No improvement for {} epochs. Stopping training.".format(patience))
                break
        
    
def main():
    global yaml_file
    # define sweep configuration
    sweep_config = {
        'method': 'bayes'
    }
    metric = {
        'name': 'best_eval_loss',
        'goal': 'minimize'   
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'optimizer': {
            'value': 'adamw'
            },
        'r': {
            'values': [4, 8, 16, 32]
        },
        'lora_alpha': {
            'values': [8, 16, 32, 64]
        },
        'lora_dropout': {
            'values': [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.000005, 
            'max': 0.0005
        },
        'epochs': {
            'values': [5, 10, 15]
        },
        'batch_size': {
            'value': yaml_file['batch_size']['value']
        },
        'max_length': {
            'value': 1024
        },
        'model_name': {
            'value': yaml_file['model_name']['value']
        }
    }
    sweep_config['parameters'] = parameters_dict
    # init the SWEEP
    sweep_id = wandb.sweep(sweep_config, project=yaml_file['wandb_project']['value'])
    wandb.agent(sweep_id, train, count=yaml_file['wandb_trials']['value'], project=yaml_file['wandb_project']['value'])


if __name__ == '__main__':
    main()
