import os
import torch
import time
import math
from GPT2_Basic_Skeleton import GPT,GPTConfig, DataLoaderLite

print("Resuming training")

#Define paths and parameters

checkpoint_path = "log/model_00100.pt"  # Update with your checkpoint path
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)
print("checkpoint loaded")

config = checkpoint['config']
print(config)
print("config loaded")

model = GPT(config)
print(model)
print("model loaded")

state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}
model.load_state_dict(state_dict)


start_step = checkpoint['step']
print(f"Resuming training from step {start_step}")

# Set RNG states
torch.set_rng_state(checkpoint['rng_state'])
if checkpoint['cuda_rng_state'] is not None:
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

###############################################################################

# Initialize data loaders
B = 2  # Micro batch size
T = 1024  # Context length
total_batch_size = 524288  # 2^19 = 0.5M tokens
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"total_desired_batch_size: {total_batch_size}")
print(f"gradient accumulation steps: {grad_accum_steps}")
torch.set_float32_matmul_precision('high')

train_loader = DataLoaderLite(B=B, T=T, split='train')
val_loader = DataLoaderLite(B=B, T=T, split='val')


model.to('cuda') 
model = torch.compile(model)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type='cuda')
optimizer.load_state_dict(checkpoint['optimizer'])

# Learning rate schedule
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 215
max_steps = 5723

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def update_log_file(log_file, step, loss_type, loss_value):
    # Read the existing log file
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Check if the step is already recorded
    step_str = f"{step} {loss_type}"
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(step_str):
            lines[i] = f"{step_str} {loss_value:.6f}\n"
            updated = True
            break

    # If not updated, append the new entry
    if not updated:
        lines.append(f"{step_str} {loss_value:.6f}\n")

    # Write back to the log file
    with open(log_file, "w") as f:
        f.writelines(lines)

# Resume training
for step in range(start_step + 1, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    if step % 10 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 5
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to('cuda'), y.to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            print(f"validation loss: {val_loss_accum.item():.4f}")
            update_log_file(log_file, step, "val", val_loss_accum.item())
            if step > 0 and (step % 10 == 0 or last_step):
                # Save model checkpoints with optimizer state and step
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                }
                torch.save(checkpoint, checkpoint_path)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to('cuda'), y.to('cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Adjust learning rate based on the current step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # milliseconds

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_second = tokens_processed / (t1 - t0)
    print(f"step {step} loss: {loss_accum.item()}, lr {lr:.6f}, norm: {norm.item():.6f}, time: {dt:.2f}ms, tokens per second: {tokens_per_second:.2f}")
    update_log_file(log_file, step, "train", loss_accum.item())
