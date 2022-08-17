from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, in_channel, hidden_dims, dropout_rates, num_class):
        super().__init__()

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropouts = nn.ModuleList()

        previous_dim = in_channel
        for dim, rate in zip(hidden_dims, dropout_rates):
            self.linears.append(nn.Linear(previous_dim, dim))
            self.bns.append(nn.BatchNorm1d(dim))
            self.dropouts.append(nn.Dropout(rate))
            previous_dim = dim

        self.classification = nn.Linear(previous_dim, num_class)

    def forward(self, x):
        for linear, bn, dropout in zip(self.linears, self.bns, self.dropouts):
            x = dropout(self.relu(bn(linear(x))))

        x = self.classification(x)

        return x
