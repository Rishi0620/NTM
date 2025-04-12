import torch
import torch.nn as nn
from .controller import LSTMController
from .memory import NTMMemory
from .heads import ReadHead, WriteHead


class NTM(nn.Module):
    def __init__(self, input_size, output_size, controller_size=100, N=128, M=20):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.N = N
        self.M = M

        self.memory = NTMMemory(N, M)
        self.controller = LSTMController(input_size + M, controller_size, controller_size)

        self.read_head = ReadHead(self.memory, controller_size)
        self.write_head = WriteHead(self.memory, controller_size)

        self.output_layer = nn.Linear(controller_size + M, output_size)

    def reset(self, batch_size=1):
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        self.read_vector = torch.zeros(batch_size, self.M)

    def forward(self, x_seq):
        batch_size, seq_len, _ = x_seq.size()
        self.reset(batch_size)

        outputs = []

        for t in range(seq_len):
            x = x_seq[:, t, :]  # (batch, input_size)
            controller_input = torch.cat([x, self.read_vector], dim=1)

            ctrl_out, _ = self.controller(controller_input)

            _ = self.write_head(ctrl_out)

            self.read_vector, _ = self.read_head(ctrl_out)

            ntm_output = self.output_layer(torch.cat([ctrl_out, self.read_vector], dim=1))
            outputs.append(ntm_output.unsqueeze(1))

        return torch.cat(outputs, dim=1)