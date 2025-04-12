import torch
import torch.nn as nn

class NTMMemory(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M
        self.register_buffer('mem_bias', torch.Tensor(N, M))
        nn.init.uniform_(self.mem_bias, -0.1, 0.1)
        self.reset()

    def reset(self, batch_size=1):
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def read(self, weights):
        return torch.bmm(weights.unsqueeze(1), self.memory).squeeze(1)

    def write(self, weights, erase_vector, add_vector):
        w = weights.unsqueeze(2)
        e = erase_vector.unsqueeze(1)
        a = add_vector.unsqueeze(1)

        self.memory = self.memory * (1 - torch.bmm(w, e)) + torch.bmm(w, a)