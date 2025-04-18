import torch
import numpy as np

model = torch.load('ntm_copy.pth')
model.eval()

input_seq = torch.rand(1, 10, 1)

with torch.no_grad():
    output = model(input_seq)

input_np = input_seq.squeeze().numpy()
output_np = output.squeeze().numpy()

print("Input Sequence: ", input_np)
print("Copied Output:  ", output_np)