#pylint: disable=no-member
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


# Numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# Use the values as observed in Rlkit
B_INIT_VALUE = 0.1
W_INIT_VALUE = 3e-3

class RecurrentPolicyModel(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 hidden_size: int = 256):
        super(RecurrentPolicyModel, self).__init__()

        self.recurrent_layer = nn.LSTM(
            num_inputs, hidden_size, batch_first=True)

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        for layer in [self.mean_linear, self.log_std_linear]:
            layer.weight.data.uniform_(-W_INIT_VALUE, W_INIT_VALUE)
            layer.bias.data.uniform_(-W_INIT_VALUE, W_INIT_VALUE)
            
        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)

    def forward(self, state):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        _, (state, _) = self.recurrent_layer(state)

        state = state.squeeze(0)

        state = F.relu(self.linear1(state))
        state = F.relu(self.linear2(state))

        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)

        std = log_std.exp()
        normal = Normal(mean, std)
        
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(RecurrentPolicyModel, self).to(device)