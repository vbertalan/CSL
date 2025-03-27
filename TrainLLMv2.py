import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# === Step 0: Login on Hugging Face (coloque seu token aqui se necessário) ===
# Step 0: Code for reading HuggingFace token
def get_huggingface_token():
    f = open("huggingface_token.txt", "r")
    return (f.read())

login(token=get_huggingface_token())  # Descomente se preferir login via código

# === Step 1: Define LLaMA 2 model ===
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# === Step 2: Add custom tokens ===
log_templates = [
    "Error encountered in module X",
    "Error encountered in",
    "Unexpected behavior in network communication",
    "System rebooted successfully",
    "Segmentation fault in memory allocation"
]
tokenizer.add_tokens(log_templates)

# === Step 3: Prepare sequences ===
sequences = [
    "Error encountered in module X The weather is great today. I am working hard.",
    "The system rebooted successfully after the error."
]

tokenized_sequences = tokenizer(
    sequences,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

# === Step 4: Load LLaMA 2 model ===
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Step 5: Resize embeddings to include new tokens ===
model.resize_token_embeddings(len(tokenizer))

# === Step 6: Prepare Dataset & Dataloader ===
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

dataset = LogSequenceDataset(tokenized_sequences)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# === Step 7: Optimizer ===
optimizer = AdamW(model.parameters(), lr=5e-5)

# === Step 8: Train the model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 3
for epoch in range(epochs):
    model.train()
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

# === Step 9: Save fine-tuned model ===
model.save_pretrained("fine_tuned_llama2")
tokenizer.save_pretrained("fine_tuned_llama2")
print("✅ Fine-tuning completed and model saved.")