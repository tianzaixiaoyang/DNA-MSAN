import torch
import config
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MSAN(nn.Module):

    def __init__(self, input_dim, out_dim, raw_features, adj_lists, normal_MSAN_weighted, self_loop=False):
        super(MSAN, self).__init__()

        self.input_size = input_dim
        self.out_size = out_dim
        self.raw_features = raw_features
        self.adj_lists = adj_lists
        self.reWeighted_matrix = torch.tensor(normal_MSAN_weighted, dtype=torch.float)
        self.self_loop = self_loop

        self.Linear = nn.Linear(self.input_size, self.out_size)
        init.kaiming_uniform_(self.Linear.weight)
        init.zeros_(self.Linear.bias)

    def forward(self, nodes_batch):

        aggregate_feats = self.aggregate(nodes_batch)

        embed_matrix = F.relu(self.Linear(aggregate_feats))

        return embed_matrix

    def _get_unique_neighs_list(self, nodes_batch):

        nodes_neighbors = [self.adj_lists[int(node)] for node in nodes_batch]

        nodes_neighbors = [set(neighbor) | {nodes_batch[i]} for i, neighbor in enumerate(nodes_neighbors)]

        unique_nodes_list = list(set.union(*nodes_neighbors))
        i = list(range(len(unique_nodes_list)))
        node2index = dict(list(zip(unique_nodes_list, i)))

        return nodes_neighbors, node2index, unique_nodes_list

    def aggregate(self, nodes):

        nodes_neighbors, node2index, unique_nodes_list = self._get_unique_neighs_list(nodes)

        assert len(nodes) == len(nodes_neighbors) 

        indicator = [(nodes[i] in nodes_neighbors[i]) for i in range(len(nodes_neighbors))]
        assert (False not in indicator)

        if not self.self_loop:
            nodes_neighbors = [(nodes_neighbors[i] - {nodes[i]}) for i in range(len(nodes_neighbors))]

        embed_matrix = self.raw_features[torch.LongTensor(unique_nodes_list)]

        mask = torch.zeros(len(nodes_neighbors), len(node2index))

        column_indices = [node2index[n] for nodes_neighbor in nodes_neighbors for n in nodes_neighbor]
        row_indices = [i for i in range(len(nodes_neighbors)) for _ in range(len(nodes_neighbors[i]))]

        nodes_extend = [nodes[i] for i in range(len(nodes_neighbors)) for _ in range(len(nodes_neighbors[i]))]
        neighbors_extend = [n for nodes_neighbor in nodes_neighbors for n in nodes_neighbor]

        mask[row_indices, column_indices] = self.reWeighted_matrix[nodes_extend, neighbors_extend]

        aggregate_feats = mask.mm(embed_matrix)

        return aggregate_feats


class Discriminator(nn.Module):
    def __init__(self, n_node, features, adj_lists, normal_MSAN_weighted):
        super(Discriminator, self).__init__()

        self.n_node = n_node
        self.features = torch.FloatTensor(features)
        self.adj_lists = adj_lists
        self.normal_MSAN_weighted = normal_MSAN_weighted
        input_dim = self.features.shape[1]
        output_dim = config.n_emb

        self.bias = nn.Parameter(torch.zeros([self.n_node]))
        self.MSAN = MSAN(input_dim, output_dim, self.features, self.adj_lists, self.normal_MSAN_weighted)

        self.embedding_matrix = None

        self.node_embedding = None
        self.node_neighbor_embedding = None
        self.neighbor_bias = None

    def score(self, node_id, node_neighbor_id):

        self.node_embedding = self.embedding_matrix[node_id, :] 
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        self.neighbor_bias = self.bias[node_neighbor_id]

        return torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1) + self.neighbor_bias

    def loss(self, node_id, node_neighbor_id, label):

        l2_loss = lambda x: torch.sum(x * x) / 2

        center_embeddings = self.MSAN(np.asarray(node_id))
        neighbor_embeddings = self.MSAN(np.asarray(node_neighbor_id))
        bias = self.bias[node_neighbor_id]
        score = torch.sum(input=center_embeddings * neighbor_embeddings, dim=1) + bias
        prob = torch.sigmoid(score)

        criterion = nn.BCELoss()

        regularization = l2_loss(center_embeddings) + l2_loss(neighbor_embeddings) + l2_loss(bias)

        _loss = criterion(prob, torch.tensor(label).float()) + config.lambda_dis * regularization

        return _loss

    def reward(self, node_id, node_neighbor_id):

        return torch.log(1 + torch.exp(self.score(node_id, node_neighbor_id))).detach()

    def get_probs(self, inputs1, inputs2):

        x = torch.pow(self.score(inputs1, inputs2), 0.25)  # (E(u).E(v))^a
        probs = torch.sigmoid(x).detach()
        return probs
