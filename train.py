import torch
import torch.nn as nn
import torch.optim as optim
from models.ntm import NTM
import numpy as np
from tqdm import tqdm

# Config
INPUT_SIZE = 1
OUTPUT_SIZE = 1
SEQ_LEN = 10
BATCH_SIZE = 1
EPOCHS = 10000
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
ntm = NTM(INPUT_SIZE, OUTPUT_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(ntm.parameters(), lr=LEARNING_RATE)

def generate_batch(seq_len):
    """
    Generates a batch of random sequences and their sorted versions.
    """
    seq = np.random.rand(BATCH_SIZE, seq_len, 1).astype(np.float32)
    sorted_seq = np.sort(seq, axis=1)

    return torch.tensor(seq).to(device), torch.tensor(sorted_seq).to(device)

print("Starting training...\n")

for epoch in tqdm(range(1, EPOCHS + 1)):
    ntm.train()

    x_batch, y_batch = generate_batch(SEQ_LEN)
    output = ntm(x_batch)

    loss = criterion(output, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"\nEpoch {epoch}, Loss: {loss.item():.6f}")
        print("Input:     ", x_batch.squeeze().detach().cpu().numpy())
        print("Target:    ", y_batch.squeeze().detach().cpu().numpy())
        print("Prediction:", output.squeeze().detach().cpu().numpy())

print("Training complete!")
torch.save(ntm.state_dict(), "ntm_sort.pth")