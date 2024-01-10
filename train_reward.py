import gzip
import json

# Replace 'your_file.jsonl.gz' with your file path
root_path = "/soft/datasets/hh-rlhf/"
dir_names = ['harmless-base/', 'helpful-base/', 'helpful-online/', 'helpful-rejection-sampled/']


# Open the gzip file in read mode
def view_data():
    for dir in dir_names:
        # with gzip.open("/raid/czy/datasets/hh-rlhf/"+dir+'train.jsonl.gz', 'rt', encoding='utf-8') as f:
        with gzip.open("/raid/czy/datasets/hh-rlhf/"+dir+'test.jsonl.gz', 'rt', encoding='utf-8') as f:
            for i,line in enumerate(f):
                print(f"===================={dir}=====================")
                data = json.loads(line)
                print(data)
                if i>0:
                    break
    with open("/raid/czy/datasets/hh-rlhf/test_full.jsonl", 'w') as f_out:            
        for dir in dir_names:
            with gzip.open("/raid/czy/datasets/hh-rlhf/"+dir+'test.jsonl.gz', 'rt', encoding='utf-8') as f:
                for i,line in enumerate(f):
                    data = json.loads(line)
                    f_out.write(line)
                    if i % 5000 == 0:
                        print(i)
                    
                
# view_data()

# exit()


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
run = wandb.init(
        # set the wandb project where this run will be logged
        project="spike-reward",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-4,
        "architecture": "SpikeGPT",
        "dataset": "hh-rlhf",
        "epochs": 1,
        "sigma_in_loss": "sigmoid",
        }
    )

device = 'cuda:2'
torch.cuda.set_device(device)
torch.manual_seed(1234)

tokenizer = PreTrainedTokenizerFast(tokenizer_file='./20B_tokenizer.json', padding_side='left', model_max_length=1024)
tokenizer.pad_token = "<|padding|>"

model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=24, n_embd=768, reward=True))
m2 = torch.load('./checkpoints/SpikeGPT-216M.pth',map_location=torch.device('cpu'))

filtered_state_dict = {k: v for k, v in m2.items() if k not in ['head.weight']}
# model.load_state_dict(filtered_state_dict, strict=False)
model.head.weight.data.normal_(mean=0.0, std=0.01)
model = model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# if os.path.exists(root_path+dir_names[0]+"ddd"):
    # chosen_train_dataset = load_from_disk(root_path+dir_names[0]+"ddd")

# else:
    # Tokenize
# train_dataset = load_dataset('json', data_files=root_path+dir_names[0]+'train.jsonl.gz')
train_dataset = load_dataset('json', data_files=root_path+'full.jsonl')
test_dataset = load_dataset('json', data_files=root_path+'test_full.jsonl')

chosen_train_dataset = train_dataset.map(lambda batch: tokenizer(batch["chosen"], truncation=True), batched=True, batch_size=16)['train']
rejected_train_dataset = train_dataset.map(lambda batch: tokenizer(batch["rejected"], truncation=True), batched=True, batch_size=16)['train']

chosen_test_dataset = test_dataset.map(lambda batch: tokenizer(batch["chosen"], truncation=True), batched=True, batch_size=16)['train']
rejected_test_dataset = test_dataset.map(lambda batch: tokenizer(batch["rejected"], truncation=True), batched=True, batch_size=16)['train']
# chosen_train_dataset.save_to_disk(root_path+dir_names[0])


def collate_fn(examples):
    examples = tokenizer.pad(
            examples,
            padding=True,
        )
    new_batch_data = []
    # new_batch_label = []

    for i in range(len(examples['input_ids'])):
        new_batch_data.append(torch.tensor(examples['input_ids'][i]))
        # new_batch_label.append(torch.tensor(examples['label'][i], dtype=torch.long))
    data = torch.stack(new_batch_data, dim=0)
    # label = torch.stack(new_batch_label, dim=0)
    # return data, label
    return data

batch_size = 1

chosen_train_loader = DataLoader(chosen_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
rejected_train_loader = DataLoader(rejected_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# TODO: shuffle two train loader together

chosen_test_loader = DataLoader(chosen_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
rejected_test_loader = DataLoader(rejected_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=1e-4)
def loss_fn(chosen_r, rejected_r):
    # losses = -torch.log(torch.relu(chosen_r - rejected_r))
    losses = -torch.log(torch.sigmoid(chosen_r - rejected_r))
    return torch.mean(losses)


num_epochs = 1

def evaluate(chosen_test_loader, rejected_test_loader, model, iters):
    logging.warning("Start evaluating...")
    # Initialize the metrics
    total_loss = 0
    total_correct = 0
    total_samples = 0
    dataset_size = chosen_test_loader.dataset.__len__()
    loop_size = int(dataset_size / iters)
    
    eval_bar = tqdm(total=iters)
    n = 0

    # Loop over the validation data in batches
    for (chosen, rejected) in zip(chosen_test_loader, rejected_test_loader):
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
        eval_bar.update(1)
        # if n > iters:
            # break

    # Compute the metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Print the metrics
    logging.warning(f"Dataset {dir_names[0]}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}, Total Samples = {total_samples}")
    run.log({"val": {"loss": avg_loss, "accuracy": avg_acc}})

# Train Reward
for epoch in range(num_epochs):

    model.train()

    bar = tqdm(total=len(chosen_train_dataset)//batch_size)

    for (chosen, rejected) in zip(chosen_train_loader, rejected_train_loader):
        bar.update(1)
        # if bar.n > 2000:
            # continue
        optimizer.zero_grad()
        # Forward pass
        # import IPython; IPython.embed()
        # chosen = batch[0]; rejected = batch[1]
        chosen = chosen.to(device); rejected = rejected.to(device)
        i = 4
        # print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        # print(f" Memory Allocated: {torch.cuda.memory_allocated(i)/1e9} GB")
        # print(f" Memory Cached: {torch.cuda.memory_reserved(i)/1e9} GB")
        chosen_r = model(chosen)[:,-1]
        rejected_r = model(rejected)[:,-1]
        # import IPython; IPython.embed()
        loss = loss_fn(chosen_r, rejected_r)
        functional.reset_net(model)

        loss.backward()
        optimizer.step()
        
        run.log({"train":{"loss": loss.item()}}, step=bar.n, commit=False)
        
        if bar.n % 500 == 0:
            # acc = (chosen_r>rejected_r).sum().item() / batch_size
            # logging.warning(f"Iteration {bar.n}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
            model.eval()        
            evaluate(chosen_test_loader, rejected_test_loader, model=model, iters = 1000)
            model.train()
    
    model.eval()        
    evaluate(chosen_test_loader, rejected_test_loader, model=model, iters = len(chosen_test_dataset)//batch_size)

    torch.save(model.state_dict(), './checkpoints/finetuned-' + 
               '-' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
    



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