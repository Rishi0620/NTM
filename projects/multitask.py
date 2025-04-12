import torch
import numpy as np

task_tokens = {
    'copy': torch.tensor([[1, 0, 0]]),
    'reverse': torch.tensor([[0, 1, 0]]),
    'duplicate': torch.tensor([[0, 0, 1]]),
}

model = torch.load('ntm_multitask.pth')
model.eval()

task = 'reverse'
task_token = task_tokens[task].unsqueeze(1)
input_seq = torch.rand(1, 10, 1)

input_with_token = torch.cat([task_token, input_seq], dim=1)

with torch.no_grad():
    output = model(input_with_token)

print(f"Task: {task}")
print("Input: ", input_seq.squeeze().numpy())
print("Output:", output.squeeze().numpy())