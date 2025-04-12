import torch
from models.ntm import NTM
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntm = NTM(input_size=1, output_size=1)
ntm.load_state_dict(torch.load("ntm_sort.pth", map_location=device))  # load saved weights
ntm.eval()

def test_sequence(seq):
    input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = ntm(input_tensor)
    return output.squeeze().cpu().numpy()

test_seq = np.random.rand(10, 1)
print("Test Input:      ", test_seq.squeeze())
predicted = test_sequence(test_seq)
print("Model Output:    ", predicted.squeeze())
print("Ground Truth:    ", np.sort(test_seq.squeeze()))