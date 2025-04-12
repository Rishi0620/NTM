import numpy as np
import torch


def int_to_one_hot(n, max_val):
    vec = np.zeros(max_val)
    vec[n] = 1
    return vec


def generate_sorting_batch(batch_size=32, seq_len=5, max_val=10):
    seqs = np.random.randint(0, max_val, size=(batch_size, seq_len))

    input_seq = np.zeros((batch_size, seq_len, max_val))
    target_seq = np.zeros((batch_size, seq_len, max_val))

    for i in range(batch_size):
        for j in range(seq_len):
            input_seq[i, j, seqs[i, j]] = 1
            target_seq[i, j, np.sort(seqs[i])[j]] = 1

    input_seq = torch.from_numpy(input_seq).float()
    target_seq = torch.from_numpy(target_seq).float()

    return input_seq, target_seq

if __name__ == "__main__":
    inp, out = generate_sorting_batch()
    print("Input shape:", inp.shape)  # (32, 5, 10)
    print("Output shape:", out.shape)  # (32, 5, 10)