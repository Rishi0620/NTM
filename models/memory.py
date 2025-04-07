import torch
import torch.nn as nn

class NTMMemory(nn.Module):
    def __init__(self, N, M):
        """
        N: Number of memory locations
        M: Vector size at each location
        """
        super().__init__()
        self.N = N
        self.M = M
        self.register_buffer('mem_bias', torch.Tensor(N, M))
        nn.init.uniform_(self.mem_bias, -0.1, 0.1)
        self.reset()

    def reset(self, batch_size=1):
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)  # (batch, N, M)

    def read(self, weights):
        """
        weights: (batch, N)
        returns: (batch, M)
        """
        return torch.bmm(weights.unsqueeze(1), self.memory).squeeze(1)

    def write(self, weights, erase_vector, add_vector):
        """
        weights: (batch, N)
        erase_vector: (batch, M)
        add_vector: (batch, M)
        """
        w = weights.unsqueeze(2)  # (batch, N, 1)
        e = erase_vector.unsqueeze(1)  # (batch, 1, M)
        a = add_vector.unsqueeze(1)    # (batch, 1, M)

        self.memory = self.memory * (1 - torch.bmm(w, e)) + torch.bmm(w, a)