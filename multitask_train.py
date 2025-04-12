import torch
import torch.nn as nn
import torch.optim as optim
from models.ntm import NTM
import numpy as np
from tqdm import tqdm

TASKS = ['copy', 'reverse', 'duplicate']
TASK_TOKENS = {
    'copy':      [1, 0, 0],
    'reverse':   [0, 1, 0],
    'duplicate': [0, 0, 1],
}
TASK_TOKEN_SIZE = len(TASK_TOKENS['copy'])

INPUT_SIZE = 1 + TASK_TOKEN_SIZE
OUTPUT_SIZE = 1
SEQ_LEN = 5
BATCH_SIZE = 1
EPOCHS = 6000
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntm = NTM(INPUT_SIZE, OUTPUT_SIZE).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(ntm.parameters(), lr=LEARNING_RATE)

def generate_batch(task):
    seq = np.random.rand(BATCH_SIZE, SEQ_LEN, 1).astype(np.float32)
    token = np.tile(np.array(TASK_TOKENS[task]), (BATCH_SIZE, SEQ_LEN, 1)).astype(np.float32)
    x = np.concatenate((token, seq), axis=-1)

    if task == 'copy':
        y = seq
    elif task == 'reverse':
        y = seq[:, ::-1, :]
    elif task == 'duplicate':
        y = np.repeat(seq, 2, axis=1)
    else:
        raise ValueError("Unknown task")

    return torch.tensor(x).to(device), torch.tensor(y).to(device)

print("🎯 Training multi-task NTM")

for epoch in tqdm(range(1, EPOCHS + 1)):
    ntm.train()

    task = np.random.choice(TASKS)
    x, y = generate_batch(task)
    output = ntm(x)

    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"\n📦 Epoch {epoch}, Task: {task}, Loss: {loss.item():.6f}")
        print("Input Seq:    ", x[0, :, TASK_TOKEN_SIZE:].detach().cpu().numpy())
        print("Target Seq:   ", y[0].detach().cpu().numpy())
        print("Prediction:   ", output[0].detach().cpu().numpy())

torch.save(ntm.state_dict(), "ntm_multitask.pth")