import torch
import torch.nn as nn
import torch.nn.functional as F

class ReadHead(nn.Module):
    def __init__(self, memory, controller_output_size, shift_range=1):
        super().__init__()
        self.memory = memory
        self.N, self.M = memory.N, memory.M

        self.key_gen = nn.Linear(controller_output_size, self.M)
        self.beta_gen = nn.Linear(controller_output_size, 1)  # strength

    def forward(self, controller_output):
        key = torch.tanh(self.key_gen(controller_output))            # (batch, M)
        beta = F.softplus(self.beta_gen(controller_output)) + 1e-5   # (batch, 1)

        # Cosine similarity
        mem = self.memory.memory  # (batch, N, M)
        key = key.unsqueeze(1)    # (batch, 1, M)
        norm_mem = F.normalize(mem, dim=2)
        norm_key = F.normalize(key, dim=2)
        sim = torch.bmm(norm_mem, norm_key.transpose(1, 2)).squeeze(2)  # (batch, N)

        weights = F.softmax(beta * sim, dim=1)
        read_vec = self.memory.read(weights)  # (batch, M)

        return read_vec, weights


class WriteHead(nn.Module):
    def __init__(self, memory, controller_output_size):
        super().__init__()
        self.memory = memory
        self.N, self.M = memory.N, memory.M

        self.key_gen = nn.Linear(controller_output_size, self.M)
        self.beta_gen = nn.Linear(controller_output_size, 1)
        self.erase_gen = nn.Linear(controller_output_size, self.M)
        self.add_gen = nn.Linear(controller_output_size, self.M)

    def forward(self, controller_output):
        key = torch.tanh(self.key_gen(controller_output))
        beta = F.softplus(self.beta_gen(controller_output)) + 1e-5

        mem = self.memory.memory
        key = key.unsqueeze(1)
        norm_mem = F.normalize(mem, dim=2)
        norm_key = F.normalize(key, dim=2)
        sim = torch.bmm(norm_mem, norm_key.transpose(1, 2)).squeeze(2)
        weights = F.softmax(beta * sim, dim=1)

        erase = torch.sigmoid(self.erase_gen(controller_output))  # (batch, M)
        add = torch.tanh(self.add_gen(controller_output))          # (batch, M)

        self.memory.write(weights, erase, add)
        return weights