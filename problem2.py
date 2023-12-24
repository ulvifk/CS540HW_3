from torch import nn

class MLP(nn.Module):
    def __init__(self, structure_dict: dict[int, int], input_dim):
        super().__init__()

        layers = []

        prev_layer_units = input_dim
        for layer, n_unit in structure_dict.items():
            layers.append(nn.Linear(prev_layer_units, n_unit))
            layers.append(nn.ReLU())
            prev_layer_units = n_unit

        layers.append(nn.Linear(prev_layer_units, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

