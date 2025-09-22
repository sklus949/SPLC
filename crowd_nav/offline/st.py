import torch.nn as nn
import torch


class FE_layer(nn.Module):
    def __init__(self, input_size):
        super(FE_layer, self).__init__()
        self.FE_layer = nn.Linear(input_size, 128)
        self.Relu = nn.ReLU()

    def forward(self, input_state):
        x_fe = self.FE_layer(input_state)
        x = self.Relu(x_fe)
        return x


class ST_layer(nn.Module):
    def __init__(self):
        super(ST_layer, self).__init__()
        self.LayerNorm1 = nn.LayerNorm(128)

        self.MutilHeadAtten_layer = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.LayerNorm2 = nn.LayerNorm(128)

        self.FowardFeed_layer = nn.Sequential(nn.Linear(128, 512),
                                              nn.ReLU(),
                                              nn.Linear(512, 128),
                                              )

    def forward(self, input_state):
        # x_norm = self.LayerNorm1(input_state)
        x_atten, _ = self.MutilHeadAtten_layer(input_state, input_state, input_state)
        add = x_atten + input_state
        add_norm = self.LayerNorm1(add)
        add_fowardfeed = self.FowardFeed_layer(add_norm)
        return torch.mean(self.LayerNorm2(add_fowardfeed + add), dim=1)


class ST(nn.Module):
    def __init__(self, input_size):
        super(ST, self).__init__()
        self.FE = FE_layer(input_size=input_size)
        self.ST = ST_layer()

    def forward(self, input_state):
        state_fe = self.FE(input_state)
        state_st = self.ST(state_fe)

        return state_st


if __name__ == '__main__':
    st = ST(13)
    x = torch.zeros(256, 6, 13)
    x = st(x)
