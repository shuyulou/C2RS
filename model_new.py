import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv, GATConv
from node2vec import Node2Vec


class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()
        
        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        # original: 0.3, 0.4, 0.5
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))
        
        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)
       
        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)
        return x

class GraphEncoder_GCN(nn.Module):
    def __init__(self, graph, rels, in_dim, hidden_dim, out_dim, num_node, num_rel, device='cuda'):
        super(GraphEncoder, self).__init__()
        self.graph = graph.to(device)
        self.relations = torch.tensor(rels).to(device)
        self.node_emb = nn.Parameter(self.generate_node2vec_embeddings(graph, in_dim).to(device))
        print(self.node_emb.size())
        # self.node_emb = nn.Parameter(torch.Tensor(num_node, in_dim).to(device))
        self.conv1 = RelGraphConv(in_dim, hidden_dim, num_rel, regularizer='basis', num_bases=2, activation=nn.Tanh())
        self.conv2 = RelGraphConv(hidden_dim, out_dim, num_rel, regularizer='basis', num_bases=2, activation=nn.ReLU())
        nn.init.xavier_uniform_(self.node_emb)
    
    def generate_node2vec_embeddings(self, dgl_graph, dimensions, walk_length=5, num_walks=20):
        nx_graph = dgl_graph.to_networkx().to_undirected()
        node2vec = Node2Vec(nx_graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = model.wv
        embeddings_tensor = torch.tensor([embeddings[str(i)] for i in range(len(embeddings))])
        return embeddings_tensor
    
    def forward(self):
        # print(self.graph.device, self.node_emb.device, self.relations.device)
        h = self.conv1(self.graph, self.node_emb, etypes=self.relations)
        out = self.conv2(self.graph, h, etypes=self.relations)
        return out

class GraphEncoder_GAT(nn.Module):
    def __init__(self, graph, rels, in_dim, hidden_dim, out_dim, num_node, num_rel, device='cuda'):
        super(GraphEncoder_GAT, self).__init__()
        self.graph = graph.to(device)
        self.relations = torch.tensor(rels).to(device)
        # self.node_emb = nn.Parameter(torch.Tensor(num_node, in_dim).to(device))
        self.node_emb = nn.Parameter(self.generate_node2vec_embeddings(graph, in_dim).to(device))
        # print(self.node_emb.size())

        # Initialize GAT layers
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=4, activation=nn.ReLU())
        self.conv2 = GATConv(hidden_dim * 4, out_dim, num_heads=1, activation=None)  # Adjust output based on heads

        nn.init.xavier_uniform_(self.node_emb)
    
    def generate_node2vec_embeddings(self, dgl_graph, dimensions, walk_length=5, num_walks=20):
        nx_graph = dgl_graph.to_networkx().to_undirected()
        node2vec = Node2Vec(nx_graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = model.wv
        embeddings_tensor = torch.tensor([embeddings[str(i)] for i in range(len(embeddings))])
        return embeddings_tensor
    
    def forward(self):
        # print(self.graph.device, self.node_emb.device, self.relations.device)
        h = self.conv1(self.graph, self.node_emb).flatten(1)
        out = self.conv2(self.graph, h).mean(1)
        return out