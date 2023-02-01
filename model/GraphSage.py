import torch_geometric.nn.aggr
from torch import nn
from torch_geometric.nn import SAGEConv

class GraphSage_Net(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim):
        super(GraphSage_Net, self).__init__()
        self.sage1 = SAGEConv(input_dim, hidden_dim,'add')
        self.sage2 = SAGEConv(hidden_dim, output_dim,'add')
    def forward(self, Features,A,E):
        Features = self.sage1(Features, A)
        Features = nn.functional.relu(Features)
        Features = nn.functional.dropout(Features, training=self.training)
        Features = self.sage2(Features, A)
        src=Features[E[0]]
        dst=Features[E[1]]
        result=(src*dst).sum(dim=-1)
        result=nn.functional.sigmoid(result)
        return result