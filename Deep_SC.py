import numpy as np
import torch
import torch.nn as nn
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole_gnn.model.abstract_recommender import SocialRecommender
from torch_geometric.nn.conv import SuperGATConv
import torch.nn.functional as F
from recbole_gnn.model.layers import LightGCNConv
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax

class HypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = 'node',
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=['num_edges'])
    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            # alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = F.gelu(alpha)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


class fusion_triple_feature(nn.Module):
    def __init__(self, emb_dim):
        super(fusion_triple_feature, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.linear_final = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, seq_hidden, pos_emb, feature_emb):
        seq_hidden = seq_hidden.unsqueeze(dim=1)
        pos_emb = pos_emb.unsqueeze(dim=1)
        feature_emb = feature_emb.unsqueeze(dim=1)
        seq_hidden = self.linear1(seq_hidden)
        pos_emb = self.linear2(pos_emb)
        feature_emb = self.linear3(feature_emb)
        fusion_feature = torch.cat((seq_hidden, pos_emb, feature_emb), dim=1)
        attn_weight = self.softmax(fusion_feature)
        fusion_feature = torch.sum(attn_weight * fusion_feature, dim=1)
        fusion_feature = self.linear_final(fusion_feature)
        return fusion_feature


class MLP_bn(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, dropout):
        super(MLP_bn, self).__init__()
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(input_dim)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.output_projection(x)
        return x


class LightGCN(nn.Module):
    def __init__(self, emb_dim, gnn_layer_num):
        super(LightGCN, self).__init__()
        self.gnn_layer_num = gnn_layer_num
        self.emb_dim = emb_dim
        self.GnnConv_list = nn.ModuleList(LightGCNConv(
            dim=self.emb_dim
        ) for _ in range(self.gnn_layer_num))

    def forward(self, x, edge_index, edge_weight):
        gnn_feature_list = []
        for gnn_layer_num in range(self.gnn_layer_num):
            x = self.GnnConv_list[gnn_layer_num](x=x, edge_index=edge_index, edge_weight=edge_weight)
            gnn_feature_list.append(x)
        x = torch.stack(gnn_feature_list, dim=1)
        x = torch.mean(x, dim=1)
        return x


class GraphFormer(nn.Module):
    def __init__(self, emb_dim, gnn_layer_num, head_num, drop_probability, neg_sample_ratio):
        super(GraphFormer, self).__init__()
        self.drop_probability = drop_probability
        self.gnn_layer_num = gnn_layer_num
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.neg_sample_ratio = neg_sample_ratio
        # self.neg_sample_ratio = neg_sample_ratio
        self.GnnConv_list = nn.ModuleList(SuperGATConv(
            in_channels=self.emb_dim,
            out_channels=self.emb_dim,
            heads=self.head_num,
            concat=True,
            # beta=True,
            dropout=self.drop_probability,
            attention_type='MX',
            neg_sample_ratio=self.neg_sample_ratio,
            is_undirected=False,
        ) for _ in range(self.gnn_layer_num))
        self.MLP_block_list = nn.ModuleList(
            MLP_bn(
                input_dim=self.head_num * self.emb_dim,
                embed_dim=self.emb_dim,
                output_dim=self.emb_dim,
                dropout=drop_probability)
            for _ in range(self.gnn_layer_num))

    def forward(self, x, edge_index):
        gnn_feature_list = []
        att_loss = 0.0
        for gnn_layer_num in range(self.gnn_layer_num):
            x = self.GnnConv_list[gnn_layer_num](x=x, edge_index=edge_index)
            att_loss = att_loss + self.GnnConv_list[gnn_layer_num].get_attention_loss()
            x = self.MLP_block_list[gnn_layer_num](x)
            gnn_feature_list.append(x)
        x = torch.stack(gnn_feature_list, dim=1)
        x = torch.mean(x, dim=1)
        return x, att_loss


class Hyper_GAT(nn.Module):
    def __init__(self, emb_dim, gnn_layer_num, gnn_head_num, drop_probability):
        super(Hyper_GAT, self).__init__()
        self.drop_probability = drop_probability
        self.gnn_layer_num = gnn_layer_num
        self.emb_dim = emb_dim
        self.gnn_head_num = gnn_head_num
        self.GnnConv_list = nn.ModuleList(HypergraphConv(
            in_channels=self.emb_dim,
            out_channels=self.emb_dim,
            use_attention=True,
            attention_mode='node',  # 'node'/'edge'
            heads=self.gnn_head_num,
            dropout=0,
        ) for _ in range(self.gnn_layer_num))
        self.MLP_block_list = nn.ModuleList(
            MLP_bn(
                input_dim=self.gnn_head_num * self.emb_dim,
                embed_dim=self.emb_dim,
                output_dim=self.emb_dim,
                dropout=drop_probability)
            for _ in range(self.gnn_layer_num))

    def forward(self, x, edge_index, edge_weight, edge_attr):
        gnn_feature_list = []
        for gnn_layer_num in range(self.gnn_layer_num):
            x = self.GnnConv_list[gnn_layer_num](x=x, hyperedge_index=edge_index, hyperedge_weight=edge_weight, hyperedge_attr=edge_attr)
            x = self.MLP_block_list[gnn_layer_num](x)
            gnn_feature_list.append(x)
        x = torch.stack(gnn_feature_list, dim=1)
        x = torch.mean(x, dim=1)
        return x


class Deep_SC(SocialRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(Deep_SC, self).__init__(config, dataset)
        self.ui_edge_index, self.ui_edge_weight = dataset.get_bipartite_inter_mat(row='user')
        self.ui_edge_index, self.ui_edge_weight = self.ui_edge_index.to(self.device), self.ui_edge_weight.to(self.device)
        self.net_edge_index, self.net_edge_weight = dataset.get_norm_net_adj_mat(row_norm=True)
        self.net_edge_index, self.net_edge_weight = self.net_edge_index.to(self.device), self.net_edge_weight.to(self.device)
        self.ii_edge_index, self.ii_edge_weight = torch.load(str(config['dataset']) + '_ii_graph' + '.pth')
        self.ii_edge_index, self.ii_edge_weight = self.ii_edge_index.to(self.device), self.ii_edge_weight.to(self.device)
        self.embedding_size = config['embedding_size']  # int type:the embedding size of DiffNet
        self.main_loss_weight = config['main_loss_weight']
        self.ssl_loss_weight = config['ssl_loss_weight']
        self.ui_gnn_layer_num = config['ui_gnn_layer_num']
        self.ui_gnn_head_num = config['ui_gnn_head_num']
        self.graphformer_layer_num = config['graphformer_layer_num']
        self.graphformer_head_num = config['graphformer_head_num']
        self.dropout = config['dropout']
        self.ligntgcn_layer = config['ligntgcn_layer']
        self.temperature_parameter = config['temperature_parameter']
        self.neg_sample_ratio = config['neg_sample_ratio']
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)
        self.n2v_emb = torch.load(str(config['dataset']) + '_n2v_user_from_sn_' + str(config['embedding_size']) + '.pth')
        self.n2v_emb.requires_grad_(False)
        self.ui_hypergat = Hyper_GAT(
            emb_dim=self.embedding_size,
            gnn_layer_num=self.ui_gnn_layer_num,
            gnn_head_num=self.ui_gnn_head_num,
            drop_probability=self.dropout,
        )
        self.iu_hypergat = Hyper_GAT(
            emb_dim=self.embedding_size,
            gnn_layer_num=self.ui_gnn_layer_num,
            gnn_head_num=self.ui_gnn_head_num,
            drop_probability=self.dropout,
        )
        self.uu_GraphFormer = GraphFormer(
            emb_dim=self.embedding_size,
            gnn_layer_num=self.graphformer_layer_num,
            head_num=self.graphformer_head_num,
            drop_probability=self.dropout,
            neg_sample_ratio=self.neg_sample_ratio
        )
        self.ii_LightGCN = LightGCN(
            emb_dim=self.embedding_size,
            gnn_layer_num=self.ligntgcn_layer
        )
        self.final_user = fusion_triple_feature(self.embedding_size)
        self.final_item = fusion_triple_feature(self.embedding_size)
        self.u_MLP = MLP_bn(
            input_dim=self.embedding_size,
            embed_dim=self.embedding_size,
            output_dim=self.embedding_size,
            dropout=self.dropout
        )
        self.i_MLP = MLP_bn(
                input_dim=self.embedding_size,
                embed_dim=self.embedding_size,
                output_dim=self.embedding_size,
                dropout=self.dropout
        )

        self.pair_loss = BPRLoss()
        self.point_loss = nn.CrossEntropyLoss())
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.emb_drop = nn.Dropout(p=self.dropout)
        self.restore_user_e = None
        self.restore_item_e = None
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def forward(self):
        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight
        user_embedding = self.emb_drop(user_embedding)
        item_embedding = self.emb_drop(item_embedding)
        user_emb_from_ui = self.ui_hypergat(
            x=user_embedding,
            edge_index=self.ui_edge_index,
            edge_weight=self.ui_edge_weight,
            edge_attr=item_embedding[self.ui_edge_index[1]]
        )
        item_emb_from_ui = self.iu_hypergat(
            x=item_embedding,
            edge_index=self.ui_edge_index.flip([0]),
            edge_weight=self.ui_edge_weight,
            edge_attr=user_embedding[(self.ui_edge_index.flip([0])[1])]
        )
        user_from_uu, sn_loss = self.uu_GraphFormer(
            x=user_emb_from_ui + user_embedding + self.n2v_emb.weight,
            edge_index=self.net_edge_index.flip([0])
        )
        item_from_ii = self.ii_LightGCN(
            x=item_embedding + item_emb_from_ui,
            edge_index=self.ii_edge_index,
            edge_weight=self.ii_edge_weight
        )
        final_user_emb = self.final_user(user_embedding, user_emb_from_ui, user_from_uu)
        final_item_emb = self.final_item(item_embedding, item_emb_from_ui, item_from_ii)
        final_user_emb = self.u_MLP(final_user_emb)
        final_item_emb = self.i_MLP(final_item_emb)
        return final_user_emb, final_item_emb, sn_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        user_all_embeddings, item_all_embeddings, sn_loss = self.forward()
        init_item_emb = self.dropout_layer(item_all_embeddings)
        init_item_emb = F.normalize(init_item_emb, dim=-1)
        init_user_emb = self.dropout_layer(user_all_embeddings)
        init_user_emb = F.normalize(init_user_emb, dim=1)
        u_embeddings = init_user_emb[user]
        pos_embeddings = init_item_emb[pos_item]
        neg_embeddings = init_item_emb[neg_item]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1) / self.temperature_parameter
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1) / self.temperature_parameter
        pair_loss = self.pair_loss(pos_scores, neg_scores)
        logits = (torch.matmul(u_embeddings, init_item_emb.transpose(0, 1)) / self.temperature_parameter)
        point_loss = self.point_loss(logits, pos_item)
        ctr_loss = self.main_loss_weight * point_loss + (1 - self.main_loss_weight) * pair_loss
        loss = ctr_loss + self.ssl_loss_weight * sn_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings, _ = self.forward()
        u_embeddings = F.normalize(user_all_embeddings, dim=-1)
        i_embeddings = F.normalize(item_all_embeddings, dim=-1)
        u_embeddings = u_embeddings[user]
        i_embeddings = i_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1) / self.temperature_parameter
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        u_embeddings = F.normalize(self.restore_user_e, dim=-1)
        i_embeddings = F.normalize(self.restore_item_e, dim=-1)
        u_embeddings = u_embeddings[user]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1)) / self.temperature_parameter
        return scores.view(-1)
