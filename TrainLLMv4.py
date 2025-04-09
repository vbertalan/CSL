import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.optim import AdamW
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

def train(model, tokenizer, dataset, num_epochs=20, batch_size=8, early_stopping_patience=3, learning_rate=5e-5, model_save_dir="models"):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = len(dataloader) * num_epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    epoch_losses = []
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}")
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()

            # Updates progress bar with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Average epoch loss {epoch}: {avg_loss:.4f}")
        
        save_path = os.path.join(model_save_dir, "intermediate_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model from epoch {epoch} saved to {save_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered. Stopping training.")
                break

    final_model_path = os.path.join(model_save_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    return epoch_losses

def main():

    # Define path to file
    file_path = "/home/vbertalan/Downloads/gpt2_logs_mini.log"
    model_name = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Define max_length (here, as block_size)
    dataset = LogDataset(file_path, tokenizer, block_size=1024)
    
    # Define other parameters
    losses = train(model, tokenizer, dataset,
                   num_epochs=5,
                   batch_size=2,
                   early_stopping_patience=3,
                   learning_rate=5e-5,
                   model_save_dir="models")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title("Função de Erro (Loss) por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()
    print("\nGráfico de perda salvo como 'training_loss.png'.")

if __name__ == "__main__":
    main()