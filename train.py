import torch
import torch.nn as nn
import torch.optim as optim
from models.ntm import NTM
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['reverse', 'duplicate', 'repeat'], default='reverse')
parser.add_argument('--repeat_n', type=int, default=3)
args = parser.parse_args()

INPUT_SIZE = 1
OUTPUT_SIZE = 1
SEQ_LEN = 5
BATCH_SIZE = 1
EPOCHS = 5000
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NTMWrapper(nn.Module):
    def __init__(self, ntm):
        super().__init__()
        self.ntm = ntm
        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def forward(self, x, output_length=None):
        if output_length is None:
            output_length = x.size(1)

        outputs = []
        for i in range(output_length):
            if i < x.size(1):
                inp = x[:, i:i + 1, :]
            else:
                inp = torch.zeros_like(x[:, 0:1, :])

            out = self.ntm(inp)
            outputs.append(out)

        return torch.cat(outputs, dim=1)


base_ntm = NTM(INPUT_SIZE, OUTPUT_SIZE).to(device)
ntm = NTMWrapper(base_ntm).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(base_ntm.parameters(), lr=LEARNING_RATE)


def generate_batch(task):
    x = np.random.rand(BATCH_SIZE, SEQ_LEN, 1).astype(np.float32)

    if task == 'reverse':
        y = x[:, ::-1, :].copy()
        output_length = SEQ_LEN
    elif task == 'duplicate':
        y = np.repeat(x, 2, axis=1)
        output_length = SEQ_LEN * 2
    elif task == 'repeat':
        y = np.repeat(x, args.repeat_n, axis=1)
        output_length = SEQ_LEN * args.repeat_n
    else:
        raise ValueError("Invalid task")

    return (torch.tensor(x).to(device),
            torch.tensor(y).to(device),
            output_length)


print(f"Training task: {args.task.upper()}")

for epoch in tqdm(range(1, EPOCHS + 1)):
    ntm.train()
    ntm.reset_state()

    x, y, output_length = generate_batch(args.task)
    output = ntm(x, output_length=output_length)

    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"\nEpoch {epoch}, Loss: {loss.item():.6f}")
        print("Input:     ", x.squeeze().cpu().numpy())
        print("Target:    ", y.squeeze().cpu().numpy())
        print("Prediction:", output.squeeze().detach().cpu().numpy())


