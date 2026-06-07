import os
import sys
import time
import math
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import BertTokenizer
from model import GPT, GPTConfig
from config import Config

def train():
    # Setup DDP variables
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set random seeds for reproducibility and diversity across ranks
    torch.manual_seed(1337 + seed_offset)
    torch.cuda.manual_seed_all(1337 + seed_offset)
    np.random.seed(1337 + seed_offset)

    conf = Config()
    model_config = GPTConfig(
        vocab_size=conf.vocab_size,
        block_size=conf.block_size,
        n_layer=conf.n_layer,
        n_head=conf.n_head,
        n_embd=conf.n_embd
    )

    if master_process:
        print("Initializing model...")
    model = GPT(model_config).to(device)

    # Wrap model with DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Load data
    if not os.path.exists("train.bin"):
        if master_process:
            print("Error: train.bin not found. Please run prepare_data.py first.")
        if ddp:
            destroy_process_group()
        sys.exit(1)
        
    data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    if master_process:
        print(f"Data has {len(data):,} tokens.")

    def get_batch():
        max_idx = len(data) - conf.block_size - 1
        # Each rank will sample different sequences because their random seeds are offset
        ix = torch.randint(0, max_idx, (conf.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+conf.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+conf.block_size]).astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    # Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # Load checkpoint if exists
    start_step = 0
    if os.path.exists(conf.checkpoint_path):
        if master_process:
            print(f"Loading checkpoint from {conf.checkpoint_path}...")
        try:
            checkpoint = torch.load(conf.checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(conf.checkpoint_path, map_location=device)
        raw_model = model.module if ddp else model
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step'] + 1
        if master_process:
            print(f"Resumed training from step {start_step}")

    # Cosine learning rate scheduler with warmup
    max_iters = conf.max_iters
    warmup_iters = 1000
    learning_rate = conf.learning_rate
    min_lr = learning_rate / 10.0

    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > max_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    tokenizer = BertTokenizer.from_pretrained(conf.tokenizer_path)

    # Time tracking for Kaggle timeout guard (8.3 hours = 29,880 seconds)
    start_time = time.time()
    max_training_seconds = 8 * 3600 + 20 * 60 

    if master_process:
        print("Starting training loop...")
        
    model.train()
    for step in range(start_step, max_iters):
        t0 = time.time()
        
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        x, y = get_batch()
        
        with torch.amp.autocast('cuda'):
            logits, loss = model(x, y)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if step % 50 == 0:
            loss_val = loss.item()
            if ddp:
                # Average loss across all DDP ranks for logging
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss_val = loss.item() / ddp_world_size
                
            if master_process:
                dt = (time.time() - t0) * 1000
                print(f"Step {step} | Loss {loss_val:.4f} | LR {lr:.2e} | Time {dt:.2f}ms")
                
        # Eval & Save
        if step > start_step and step % conf.eval_interval == 0:
            if master_process:
                model.eval()
                raw_model = model.module if ddp else model
                # Check performance via sample generation
                context = torch.tensor([tokenizer.encode("人工智能", add_special_tokens=False)], device=device)
                gen = raw_model.generate(context, max_new_tokens=40)
                print(f"\n--- Step {step} Gen Test: {tokenizer.decode(gen[0].tolist())} ---\n")
                
                # Save checkpoint
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'config': model_config
                }
                torch.save(checkpoint, conf.checkpoint_path)
                print(f"Saved checkpoint to {conf.checkpoint_path}")
                model.train()
                
        # Safe exit guard: check elapsed time
        elapsed = time.time() - start_time
        if elapsed > max_training_seconds:
            if master_process:
                print(f"\n[Timeout Guard] Training time elapsed: {elapsed/3600:.2f} hours.")
                print("Saving final checkpoint and exiting to prevent Kaggle timeout cleanup...")
                raw_model = model.module if ddp else model
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'config': model_config
                }
                torch.save(checkpoint, conf.checkpoint_path)
                print(f"Checkpoint saved to {conf.checkpoint_path}. Exit successful.")
            if ddp:
                destroy_process_group()
            sys.exit(0)

    # If training completes naturally
    if master_process:
        print("Training reached max iterations!")
        raw_model = model.module if ddp else model
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': max_iters - 1,
            'config': model_config
        }
        torch.save(checkpoint, conf.checkpoint_path)
        print("Final checkpoint saved.")

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    train()
