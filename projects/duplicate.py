import torch
import numpy as np

model = torch.load('ntm_duplicate.pth')
model.eval()

input_seq = torch.rand(1, 10, 1)

with torch.no_grad():
    output = model(input_seq)

input_np = input_seq.squeeze().numpy()
output_np = output.squeeze().numpy()
ground_truth = np.concatenate([input_np, input_np])

print("Input Sequence:   ", input_np)
print("Duplicated Output:", output_np)
print("Ground Truth:     ", ground_truth)