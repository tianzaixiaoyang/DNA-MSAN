import torch
import config
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_node, node_emd_init):
        super(Generator, self).__init__()
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        self.embedding_matrix = nn.Parameter(torch.tensor(self.node_emd_init))
        self.bias = nn.Parameter(torch.zeros([self.n_node]))

        self.node_embedding = None
        self.node_neighbor_embedding = None
        self.node_neighbor_bias = None

    def all_score(self):

        return (torch.matmul(self.embedding_matrix, torch.transpose(self.embedding_matrix, 0, 1)) + self.bias).detach()

    def score(self, node_id, node_neighbor_id):

        self.node_embedding = self.embedding_matrix[node_id, :]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        self.node_neighbor_bias = self.bias[node_neighbor_id]

        return torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1) + self.node_neighbor_bias

    def loss(self, prob, reward):

        l2_loss = lambda x: torch.sum(x * x) / 2
        prob = torch.clamp(input=prob, min=1e-5, max=1) 

        regularization = l2_loss(self.node_embedding) + l2_loss(self.node_neighbor_embedding) + l2_loss(self.node_neighbor_bias)

        _loss = -torch.mean(torch.log(prob) * reward) + config.lambda_gen * regularization

        return _loss
