import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

# Define the path for the template file
#log_file = "/home/vbertalan/Downloads/gpt2_logs_mini.log"
log_file = "/home/vbertalan/Downloads/input.txt"

# Auxiliary function to read lines from a raw log file
def read_lines_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Step 1: Load GPT-2 tokenizer and add custom log templates
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
log_templates = read_lines_from_file(log_file)

# Add log templates as tokens to the tokenizer
tokenizer.add_tokens(log_templates)

# Set padding token to EOS
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Prepare log sequences directly from log templates
sequences = log_templates  # Use log templates as sequences

# Tokenize sequences
tokenized_sequences = tokenizer(
    sequences,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

# Step 3: Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Step 4: Prepare Dataset and DataLoader for training
class LogSequenceDataset(Dataset):
    def __init__(self, tokenized_sequences):
        self.input_ids = tokenized_sequences['input_ids']
        self.attention_mask = tokenized_sequences['attention_mask']

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# Create the dataset and dataloader
dataset = LogSequenceDataset(tokenized_sequences)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 6: Enable gradient updates on embeddings
model.get_input_embeddings().requires_grad_(True)

# Step 7: Train the model (Continual Pretraining)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using {} for training".format(device))

max_epochs = 5  # Maximum number of epochs
patience = 2  # Number of epochs to wait for improvement
best_loss = float('inf')  # Initialize best loss to infinity
epochs_without_improvement = 0  # Counter for epochs without improvement

# Initialize lists to store losses
train_losses = []

# Open log file to record epoch losses
with open("training_log.txt", "w") as log_file:
    try:
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{max_epochs}", leave=True)

            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                optimizer.zero_grad()

                # Forward pass (language modeling task)
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accumulate loss
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

                # Frequent logging after each batch
                log_file.write(f"Epoch {epoch + 1}, Batch Loss: {loss.item():.4f}\n")

            # Calculate average loss for the epoch
            average_loss = total_loss / len(dataloader)
            train_losses.append(average_loss)
            log_file.write(f"Epoch {epoch + 1}/{max_epochs} - Average Loss: {average_loss:.4f}\n")
            print(f"Epoch {epoch + 1}/{max_epochs} completed. Average Loss: {average_loss:.4f}")

            # Early stopping logic
            if average_loss < best_loss:
                best_loss = average_loss
                epochs_without_improvement = 0  # Reset counter
                # Save the last intermediate model whenever there is an improvement
                model.save_pretrained("fine_tuned_intermediate")
                tokenizer.save_pretrained("fine_tuned_intermediate")
                print("Intermediate model saved as 'fine_tuned_intermediate'.")
            else:
                epochs_without_improvement += 1

            # Stop training if no improvement for 'patience' epochs
            if epochs_without_improvement >= patience:
                print("Early stopping triggered. No improvement in loss.")
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        log_file.write(f"Training interrupted due to an error: {e}\n")

# Step 8: Save the final fine-tuned model
model.save_pretrained("fine_tuned_gpt2_final")
tokenizer.save_pretrained("fine_tuned_gpt2_final")

print("Fine-tuning completed and final model saved.")

# Step 9: Plotting the loss trend
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title('Training Loss Trend')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.xticks(range(1, len(train_losses) + 1))
plt.grid()

# Save the plot as a PNG file
plt.savefig("fine_tuned_gpt2_final/loss_plot.png")
plt.close()  # Close the plot to free memory