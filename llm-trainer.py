import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

# Custom dataset for log files
class LogDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.examples = []
        for line in lines:
            if line.strip():
                tokenized = tokenizer(line, truncation=True, max_length=block_size, padding="max_length")
                self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.examples[idx].items()}
        item["labels"] = item["input_ids"].clone()
        return item

def train(model, tokenizer, dataset, num_epochs=20, base_batch_size=12, early_stopping_patience=3, learning_rate=5e-5, model_save_dir="models"):
    accelerator = Accelerator()
    device = accelerator.device
    
    if not os.path.exists(model_save_dir) and accelerator.is_main_process:
        os.makedirs(model_save_dir)

    # Escala o batch size de acordo com o número de processos
    batch_size = base_batch_size * accelerator.num_processes
    num_workers = min(4 * accelerator.num_processes, os.cpu_count())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    total_steps = len(dataloader) * num_epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    epoch_losses = []
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        accelerator.print(f"\nEpoch {epoch}")
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        accelerator.print(f"Average epoch loss {epoch}: {avg_loss:.4f}")
        
        if accelerator.is_main_process:
            save_path = os.path.join(model_save_dir, "intermediate_model")
            model.module.save_pretrained(save_path) if hasattr(model, "module") else model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            accelerator.print(f"Model from epoch {epoch} saved to {save_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            accelerator.print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= early_stopping_patience:
                accelerator.print("Early stopping triggered. Stopping training.")
                break

    if accelerator.is_main_process:
        final_model_path = os.path.join(model_save_dir, "final_model")
        model.module.save_pretrained(final_model_path) if hasattr(model, "module") else model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        accelerator.print(f"\nFinal model saved to {final_model_path}")

    return epoch_losses

def main():
    file_path = "/home/vberta/projects/def-aloise/vberta/Paper3/gpt2_logs.log"
    model_dir = Path("/home/vberta/projects/def-aloise/vberta/Paper3/hf_models/gpt2").resolve()

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)
    dataset = LogDataset(file_path, tokenizer, block_size=1024)
    
    losses = train(model, tokenizer, dataset,
                   num_epochs=20,
                   base_batch_size=4,  # ← será multiplicado pelo número de GPUs
                   early_stopping_patience=3,
                   learning_rate=5e-5,
                   model_save_dir="models")
    
    accelerator = Accelerator()
    if accelerator.is_main_process:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker="o")
        plt.title("Função de Erro (Loss) por Época")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("training_loss.png")
        plt.show()
        print("\nLoss graph saved as 'training_loss.png'.")

if __name__ == "__main__":
    main()