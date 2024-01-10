import gzip
import json
import torch.distributed as dist
from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets
from transformers import PreTrainedTokenizerFast,AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from src.model import GPT, GPTConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AdamW
from src.spikingjelly.clock_driven import functional
from transformers import DataCollatorWithPadding
import os
import logging
from datetime import datetime

def setup():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()
    
def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

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

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
    
class DDPTrainer:
    def __init__(self, snapshot_path) -> None:
        if os.path.exists(snapshot_path):
            self._load_snapshot(snapshot_path)
            
    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def evaluate(chosen_test_loader, rejected_test_loader, iters, rank, model):
        logging.info("Start evaluating...")
        # Initialize the metrics
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        bar = tqdm(total=iters)

        # Loop over the validation data in batches
        for (chosen, rejected) in tqdm(zip(chosen_test_loader, rejected_test_loader)):
            # Forward pass
            chosen = chosen.to(rank); rejected = rejected.to(rank)
            with torch.no_grad():
                chosen_r = model(chosen)[:,-1]
                rejected_r = model(rejected)[:,-1]
                loss = loss_fn(chosen_r, rejected_r)

            # Update the metrics
            total_loss += loss.item()
            total_correct += (chosen_r>rejected_r).sum().item()
            total_samples += chosen.size(0)
            functional.reset_net(model)
            bar.update(1)
            if bar.n > iters:
                break

        # Compute the metrics
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # Print the metrics
        logging.info(f"Dataset {dir_names[0]}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")

    def loss_fn(chosen_r, rejected_r):
        losses = -torch.log(torch.relu(chosen_r - rejected_r))
        return torch.mean(losses)


    def train(
        self,
        train_dataset,
        test_dataset,
        wandb_run=None,
    ):
        setup()
        rank = int(os.environ["LOCAL_RANK"])
        print(f"Running DDP reward training on rank {rank}.")
        torch.manual_seed(1234)

        # create model and move it to GPU with id rank
        model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=24, n_embd=768, reward=True))
        ddp_model = DDP(model, device_ids=[rank], output_device=i)
        self.model = ddp_model
        m2 = torch.load('./checkpoints/SpikeGPT-216M.pth',map_location=torch.device('cpu'))

        filtered_state_dict = {k: v for k, v in m2.items() if k not in ['head.weight']}
        ddp_model.load_state_dict(filtered_state_dict, strict=False)
        ddp_model.module.head.weight.data.normal_(mean=0.0, std=0.01)

        size = dist.get_world_size()
        partition_sizes = [1.0 / size for _ in range(size)]
        chosen_train_dataset = DataPartitioner(chosen_train_dataset, partition_sizes)
        chosen_train_dataset = chosen_train_dataset.use(dist.get_rank())
        rejected_train_dataset = DataPartitioner(rejected_train_dataset, partition_sizes)
        rejected_train_dataset = rejected_train_dataset.use(dist.get_rank())
        chosen_test_dataset = DataPartitioner(chosen_test_dataset, partition_sizes)
        chosen_test_dataset = chosen_test_dataset.use(dist.get_rank())    
        rejected_test_dataset = DataPartitioner(rejected_test_dataset, partition_sizes)
        rejected_test_dataset = rejected_test_dataset.use(dist.get_rank())

        batch_size = 1 # micro batch size, total bs = mbs * num_gpus

        chosen_train_loader = DataLoader(chosen_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        rejected_train_loader = DataLoader(rejected_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        # TODO: shuffle two train loader together

        chosen_test_loader = DataLoader(chosen_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        rejected_test_loader = DataLoader(rejected_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        optimizer = AdamW(model.parameters(), lr=1e-4)

        num_epochs = 1
        # Train Reward
        for epoch in range(num_epochs):

            model.train()

            bar = tqdm(total=len(chosen_train_dataset)//batch_size)

            for (chosen, rejected) in zip(chosen_train_loader, rejected_train_loader):
                bar.update(1)
                if bar.n < 13180:
                    continue
                optimizer.zero_grad()
                # Forward pass
                # import IPython; IPython.embed()
                # chosen = batch[0]; rejected = batch[1]
                chosen = chosen.to(rank); rejected = rejected.to(device)
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
                
                if bar.n % 100 == 0:
                    acc = (chosen_r>rejected_r).sum().item() / batch_size
                    # logging.info(f"Iteration {bar.n}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
                    model.eval()        
                    evaluate(chosen_test_loader, rejected_test_loader, iters = 50, rank=rank, model=ddp_model)
            
            model.eval()        
            evaluate(chosen_test_loader, rejected_test_loader, iters = len(chosen_test_dataset)//batch_size, 
                    rank=rank, model=ddp_model)
            
            cleanup()

    
if __name__ == "__main__":
    import wandb
    import random

    # start a new wandb run to track this script
    wandb.login(key="d1404233fdb9f8caf6f207dae8cf113a180e3882")
    run = None
    '''
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="spike-reward",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-4,
        "architecture": "SpikeGPT",
        "dataset": "all",
        "epochs": 1,
        }
    )
    '''
        
    # Replace 'your_file.jsonl.gz' with your file path
    root_path = "/soft/datasets/"
    dir_names = ['hh-rlhf/','rlhf-reward-datasets']
    datasets = []
    train_dataset = load_dataset('json', data_files=root_path+dir_names[0]+'train.jsonl.gz')['train']
    test_dataset = load_dataset('json', data_files=root_path+dir_names[0]+'test.jsonl.gz')['train']
    hh_rlhf_dataset = {
        'train': train_dataset,
        'test': test_dataset
    }
    concatenate_datasets(datasets)

    now = datetime.now()
    print(f'runs/{now.strftime("%Y_%m_%d_%H_%M")}_log.txt')

    logging.basicConfig(filename=f'./runs/{now.strftime("%Y_%m_%d_%H_%M")}_log.txt', level=logging.INFO, filemode='w', force=True)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file='./20B_tokenizer.json', padding_side='left', model_max_length=1024)
    tokenizer.pad_token = "<|padding|>"

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # if os.path.exists(root_path+dir_names[0]+"ddd"):
        # chosen_train_dataset = load_from_disk(root_path+dir_names[0]+"ddd")

    # else:
        # Tokenize
    train_dataset = load_dataset('json', data_files=root_path+dir_names[0]+'train.jsonl.gz')['train']
    test_dataset = load_dataset('json', data_files=root_path+dir_names[0]+'test.jsonl.gz')['train']

    chosen_train_dataset = train_dataset.map(lambda batch: tokenizer(batch["chosen"], truncation=True), batched=True, batch_size=16)['train']
    rejected_train_dataset = train_dataset.map(lambda batch: tokenizer(batch["rejected"], truncation=True), batched=True, batch_size=16)['train']

    chosen_test_dataset = test_dataset.map(lambda batch: tokenizer(batch["chosen"], truncation=True), batched=True, batch_size=16)
    rejected_test_dataset = test_dataset.map(lambda batch: tokenizer(batch["rejected"], truncation=True), batched=True, batch_size=16)
    # chosen_train_dataset.save_to_disk(root_path+dir_names[0])
    
    
    DDPTrainer.train()
    
# torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py