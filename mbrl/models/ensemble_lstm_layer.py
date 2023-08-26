import torch
from torch import nn

from typing import List, Sequence

class EnsembleLSTMLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(self, num_members: int, in_size: int, out_size: int):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size

        # self.lstm = [
        #     nn.LSTM(in_size, out_size, batch_first=True)
        #     for _ in range(num_members)
        # ]
        
        self.lstm = nn.LSTM(in_size, out_size, batch_first=True)

        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        print('Entering in LSTM:', x.shape)
        _, (x, _) = self.lstm(x)
        print('Leaving LSTM:', x.squeeze(0).shape)
        return x.squeeze(0)
        # if self.use_only_elite:
        #     torch.cat(
        #         [
        #             lstm(x)
        #             for lstm in self.lstm
        #         ]
        #     )
            
        # else:
        #     torch.cat(
        #         [
        #             lstm(x)
        #             for lstm in self.lstm[self.elite_models]
        #         ]
        #     )

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite