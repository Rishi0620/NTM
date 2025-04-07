import torch
import torch.nn as nn

class LSTMController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.reset()

    def reset(self, batch_size=1):
        self.h = torch.zeros(batch_size, self.hidden_size)
        self.c = torch.zeros(batch_size, self.hidden_size)

    def forward(self, x):
        self.h, self.c = self.lstm(x, (self.h, self.c))
        out = self.output(self.h)
        return out, self.h