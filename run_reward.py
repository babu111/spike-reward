import gzip
import json
from datasets import load_dataset, load_metric, load_from_disk
from transformers import PreTrainedTokenizerFast,AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from src.model import GPT, GPTConfig
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from src.spikingjelly.clock_driven import functional
from transformers import DataCollatorWithPadding
import logging, wandb
from datetime import datetime

now = datetime.now()
print(f'runs/{now.strftime("%Y_%m_%d_%H_%M")}_log.txt')

logging.basicConfig(filename=f'./runs/{now.strftime("%Y_%m_%d_%H_%M")}_log.txt', level=logging.WARNING, filemode='w', force=True)

wandb.login(key="d1404233fdb9f8caf6f207dae8cf113a180e3882")
# run = None

run = wandb.init(
        # set the wandb project where this run will be logged
        project="spike-reward",
        
        # track hyperparameters and run metadata
        config={
        "architecture": "SpikeGPT",
        "dataset": "rlhf-reward",
        "sigma_in_loss": "sigmoid",
        }
    )


device = 'cuda:2'
torch.cuda.set_device(device)
torch.manual_seed(1234)

tokenizer = PreTrainedTokenizerFast(tokenizer_file='./20B_tokenizer.json', padding_side='left', model_max_length=1024)
tokenizer.pad_token = "<|padding|>"

model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=24, n_embd=768, reward=True))
m2 = torch.load('./checkpoints/finetuned--2023-12-31-12-10-09.pth',map_location=torch.device('cpu'))

# filtered_state_dict = {k: v for k, v in m2.items() if k not in ['head.weight']}
model.load_state_dict(m2, strict=False)
# model.head.weight.data.normal_(mean=0.0, std=0.01)
model = model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# if os.path.exists(root_path+dir_names[0]+"ddd"):
    # chosen_train_dataset = load_from_disk(root_path+dir_names[0]+"ddd")

# else:
    # Tokenize
# train_dataset = load_dataset('json', data_files=root_path+dir_names[0]+'train.jsonl.gz')

root_path = "/soft/datasets/"
dir_names = ["rlhf-reward-datasets"]
dataset = load_dataset(root_path+dir_names[0])

# import IPython; IPython.embed()
train_dataset = dataset['train']
test_dataset = dataset['test']

def tokenize_function(examples):
    examples['chosen_full'] = [examples['prompt'][i]+examples['chosen'][i] for i in range(len(examples['chosen']))]
    examples['rejected_full'] = [examples['prompt'][i]+examples['rejected'][i] for i in range(len(examples['rejected']))]
    tokenized_preferred = tokenizer(examples["chosen_full"], padding='max_length', truncation=True)
    tokenized_rejected = tokenizer(examples["rejected_full"], padding='max_length', truncation=True)

    return {
        "input_ids_preferred": tokenized_preferred['input_ids'],
        "input_ids_rejected": tokenized_rejected['input_ids']
    }


# chosen_train_dataset = train_dataset.map(lambda batch: tokenizer(batch["prompt"]+batch["chosen"], truncation=True), batched=True, batch_size=16)
# rejected_train_dataset = train_dataset.map(lambda batch: tokenizer(batch["rejected"], truncation=True), batched=True, batch_size=16)
# train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=16)

# chosen_test_dataset = test_dataset.map(lambda batch: tokenizer(batch["chosen"], truncation=True), batched=True, batch_size=16)
# rejected_test_dataset = test_dataset.map(lambda batch: tokenizer(batch["rejected"], truncation=True), batched=True, batch_size=16)
test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=16)
# chosen_train_dataset.save_to_disk(root_path+dir_names[0])


def collate_fn(examples):
    batch_preferred = []
    batch_rejected = []
    for i in range(len(examples)):
        batch_preferred.append(torch.tensor(examples[i]['input_ids_preferred']))
        batch_rejected.append(torch.tensor(examples[i]['input_ids_rejected']))
        
    chosen = torch.stack(batch_preferred, dim=0)
    rejected = torch.stack(batch_rejected, dim=0)
    # import IPython; IPython.embed(); exit()
    return chosen, rejected

batch_size = 16

# chosen_train_loader = DataLoader(chosen_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# rejected_train_loader = DataLoader(rejected_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# chosen_test_loader = DataLoader(chosen_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# rejected_test_loader = DataLoader(rejected_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=1e-4)
def loss_fn(chosen_r, rejected_r):
    # losses = -torch.log(torch.relu(chosen_r - rejected_r))
    losses = -torch.log(torch.sigmoid(chosen_r - rejected_r))
    return torch.mean(losses)


def evaluate(test_loader, model, iters):
    logging.warning("Start evaluating...")
    # Initialize the metrics
    total_loss = 0
    total_correct = 0
    total_samples = 0
    dataset_size = test_loader.dataset.__len__()
    loop_size = int(dataset_size / iters)
    
    eval_bar = tqdm(total=iters)
    n = 0

    # Loop over the validation data in batches
    for batch in test_loader:
        # import IPython; IPython.embed(); exit()
        chosen = batch[0]; rejected = batch[1]
        n += 1
        if n % loop_size != 0:
            continue
        # Forward pass
        chosen = chosen.to(device); rejected = rejected.to(device)
        with torch.no_grad():
            chosen_r = model(chosen)[:,-1]
            rejected_r = model(rejected)[:,-1]
            loss = loss_fn(chosen_r, rejected_r)

        # Update the metrics
        total_loss += loss.item()
        total_correct += (chosen_r>rejected_r).sum().item()
        total_samples += chosen.size(0)
        functional.reset_net(model)
        eval_bar.update(chosen.size(0))
        # if n > iters:
            # break

    # Compute the metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    print({"val": {"loss": avg_loss, "accuracy": avg_acc}})

    # Print the metrics
    logging.warning(f"Dataset {dir_names[0]}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}, Total Samples = {total_samples}")
    run.log({"val": {"loss": avg_loss, "accuracy": avg_acc}})


model.eval()        
evaluate(test_loader, model=model, iters = len(test_dataset))

    



"""
emb.weight',
 'blocks.0.ln1.weight',
 'blocks.0.ln1.bias',
 'blocks.0.ln2.weight',
 'blocks.0.ln2.bias',
 'blocks.0.ln0.weight',
 'blocks.0.ln0.bias',
 'blocks.0.att.time_decay',
 'blocks.0.att.time_first',
 'blocks.0.att.time_mix_k',
 'blocks.0.att.time_mix_v',
 'blocks.0.att.time_mix_r',
 'blocks.0.att.key.weight',
 'blocks.0.att.value.weight',
 'blocks.0.att.receptance.weight',
 'blocks.0.att.output.weight',
 'blocks.0.ffn.time_mix_k',
 'blocks.0.ffn.time_mix_r',
 'blocks.0.ffn.key.weight',
 'blocks.0.ffn.receptance.weight',
 'blocks.0.ffn.value.weight',
 'ln_out.weight',
 'ln_out.bias',
 'head.weight'
"""