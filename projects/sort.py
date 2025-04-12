import torch
import numpy as np

model = torch.load('ntm_sort.pth')
model.eval()

input_seq = torch.rand(1, 10, 1)

with torch.no_grad():
    output = model(input_seq)

input_np = input_seq.squeeze().numpy()
output_np = output.squeeze().numpy()
ground_truth = np.sort(input_np)

print("Original Input: ", input_np)
print("Sorted Output:  ", output_np)
print("Ground Truth:   ", ground_truth)