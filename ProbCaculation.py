import torch
import torch.nn.functional as F

# Assuming MyTrainedModel is already defined elsewhere
# model = MyTrainedModel(vocab_size=1000)  # Adjust vocab_size based on your model
# model.load_state_dict(torch.load("path_to_trained_model.pth"))  # Load trained weights
# model.eval()  # Set the model to evaluation mode

# Example: Define sequences for testing
batch_size, seq_length, vocab_size = 5, 10, 1000

# Define the computeProb function
def computeProb(i, j, logits):
    # Compute probability distributions
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_length, vocab_size)

    # Compute token occurrence probabilities across all positions
    p_token_j = probs.sum(dim=1)  # Shape: (batch_size, vocab_size)

    # Compute conditional probabilities P(token_i | token_j)
    conditional_probs = torch.zeros(vocab_size, vocab_size)
    window_size = 3  # Look at 3 tokens before

    # Compute joint probability of token_j appearing before token_i
    num = torch.sum(probs[:, :-window_size, token_j] * probs[:, window_size:, token_i])
    denom = torch.sum(probs[:, :, token_j])  # Total probability of token_j

    conditional_probs[token_i, token_j] = num / (denom + 1e-8)  # Avoid division by zero
    
    return conditional_probs

# Function to check conditional independence
def check_conditional_independencies(model):
    conditional_independencies = []    
    
    for token_j in range(vocab_size):
        for token_i in range(vocab_size):
            if token_j != token_i:
                for different C:
                    sequences_with_C = # Sequences having j before i and having C between them
                    sequences_without_C =   # Sequences having j before i and not having C between them
  

                    # Get logits from the model for both sequences
                    logits_without_C = model(sequences_without_C)  # Shape: (batch_size, seq_length, vocab_size)
                    logits_with_C = model(sequences_with_C)  # Shape: (batch_size, seq_length, vocab_size)
    
                    # Compute probabilities for both cases
                    Pij = computeProb(token_i, token_j, logits_without_C)
                    PijC = computeProb(token_i, token_j, logits_with_C)
                
                    # Check if probabilities are almost the same (i.e., conditional independence)
                    if torch.allclose(Pij, PijC, atol=1e-4):
                        conditional_independencies.append((token_i, token_j))

    return conditional_independencies

# Example usage
conditional_independencies = check_conditional_independencies(model)

# Print out the result
print("Conditional Independencies: ", conditional_independencies)
