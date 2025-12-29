import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GatedFusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.struct_processor = nn.Sequential(
            nn.Linear(1024, 512), #????????
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.seq_processor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gate_generator = nn.Sequential(
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()  
        )
        self.fusion_output = nn.Sequential(
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.LayerNorm(512)
        )

    def forward(self, struct, seq):
        """
        struct: 结构特征 [batch_size, 1024]
        seq: 序列特征 [batch_size, seq_len, 1280]
        """
        batch_size, seq_len, _ = seq.shape
        
        struct_processed = self.struct_processor(struct)  
        
        gate = self.gate_generator(struct_processed)  
        gate = gate.unsqueeze(1).expand(-1, seq_len, -1)  
        
        seq_processed = self.seq_processor(seq) 
        fused = gate * struct_processed.unsqueeze(1).expand(-1, seq_len, -1) + (1 - gate) * seq_processed

        return self.fusion_output(fused)  

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=702):
        super().__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )
        
        
    def forward(self, interaction):
        x = interaction.view(interaction.size(0), -1)   
        return self.fc(x) 
    
class GIN(torch.nn.Module):
    def __init__(self,  param):
        super(GIN, self).__init__()

        self.num_layers = param['ppi_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        
        self.layers.append(GINConv(nn.Sequential(nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']), 
                                                 nn.ReLU(), 
                                                 nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                 nn.ReLU(), 
                                                 nn.BatchNorm1d(param['ppi_hidden_dim'])), 
                                                 aggregator_type='sum', 
                                                 learn_eps=True))

        for i in range(self.num_layers - 1):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                     nn.ReLU(), 
                                                     nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                     nn.ReLU(), 
                                                     nn.BatchNorm1d(param['ppi_hidden_dim'])), 
                                                     aggregator_type='sum', 
                                                     learn_eps=True))

        self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
        self.fc = nn.Linear(1024, param['output_dim'])

        self.alpha = nn.Parameter(torch.tensor(0.7))  # 初始值为 0.7
        self.beta = nn.Parameter(torch.tensor(0.3))   # 初始值为 0.3
        self.linear_transform = nn.Linear(1024, 1280)
        self.fused = 1
        self.attention_fusion = GatedFusionNetwork()
        self.classification_head = ClassificationHead(input_dim=702*702)
        self.theta_initial = 0.08#: 0.05-0.15 之间
        self.res_alpha = 0.1 #: 初始值0.1，让模型自适应学习
        self.input_projection = nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim'])
        self.norms = nn.ModuleList([nn.LayerNorm(param['ppi_hidden_dim']) for _ in range(self.num_layers)])
        self.use_initial_residual = True

    def get_protein_embedding(node_list,string_embedding_esm2):

        embedding_matrix = []
        for idx in node_list:
            row_data = string_embedding_esm2.iloc[idx].values  # 获取该行的数据
            row_array = np.array(row_data, dtype=np.float32)  # 转换为 numpy 数组
            embedding_matrix.append(row_array)

        embedding_matrix = np.stack(embedding_matrix, axis=0)
        return embedding_matrix

    def apply_attention(self, seq_feat, struct_feat):

        return self.attention_fusion(seq_feat, struct_feat)


    def forward(self, g, x,string_embedding_esm2_tensor, ppi_list, idx):

        x_initial_projected = self.input_projection(x)
        
        for l, layer in enumerate(self.layers):
            if l == 0:
                x_new = layer(g, x)
                x_new = x_new + self.theta_initial * x_initial_projected
            else:
                x_prev = x.clone()
                x_new = layer(g, x)
                x_new = x_new + self.res_alpha * x_prev
                if self.use_initial_residual:
                    x_new = x_new + self.theta_initial * x_initial_projected
            
            # 层归一化
            x = self.norms[l](x_new)
            # Dropout
            x = self.dropout(x)


        x = F.dropout(F.relu(self.linear(x)), p=0.2, training=self.training)

        node_id = np.array(ppi_list)[idx]
        x1 = x[node_id[:, 0]] # 5000 * 1024
        x2 = x[node_id[:, 1]]     

        seq_embedding_matrix_x1 = string_embedding_esm2_tensor[node_id[:, 0]].float().to(device)
        seq_embedding_matrix_x2 = string_embedding_esm2_tensor[node_id[:, 1]].float().to(device)

        apply_attention_x1 = self.attention_fusion(x1,seq_embedding_matrix_x1)  # [batch_size, seq_len, feature_dim]
        apply_attention_x2 = self.attention_fusion(x2, seq_embedding_matrix_x2)  # [batch_size, seq_len, feature_dim]

        apply_attention_x1 = F.normalize(apply_attention_x1, p=2, dim=-1)
        apply_attention_x2 = F.normalize(apply_attention_x2, p=2, dim=-1)

        d = apply_attention_x1.size(-1)  # 获取特征维度

        interaction = torch.bmm(apply_attention_x1, apply_attention_x2.transpose(1, 2)) / (d ** 0.5) # (500, 702, 702)
        output = self.classification_head(interaction)

        
        return output,interaction.detach().cpu().numpy()
    

class GCN_Encoder(nn.Module):
    def __init__(self, param, data_loader):
        super(GCN_Encoder, self).__init__()
        
        self.data_loader = data_loader
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
        self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
                                            'STR_KNN' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
                                            'STR_DIS' : GraphConv(param['input_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                                'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                                'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

    def forward(self, vq_layer):

        prot_embed_list = []

        for iter, batch_graph in enumerate(self.data_loader):

            batch_graph.to(device)
            h = self.encoding(batch_graph)
            z, _, _ = vq_layer(h)
            batch_graph.ndata['h'] = torch.cat([h, z], dim=-1)
            prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
            prot_embed_list.append(prot_embed)

        return torch.cat(prot_embed_list, dim=0)


    def encoding(self, batch_graph):

        x = batch_graph.ndata['x']

        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
            if l != self.num_layers - 1:
                x = self.dropout(x)

        return x
        

class GCN_Decoder(nn.Module):
    def __init__(self, param):
        super(GCN_Decoder, self).__init__()
        
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                                'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                                'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['input_dim']))
        self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                            'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                            'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))


    def decoding(self, batch_graph, x):

        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.fc[l](x['amino_acid'])

            if l != self.num_layers - 1:
                x = self.dropout(self.norms[l](F.relu(x)))
            else:
                pass

        return x


class CodeBook(nn.Module):
    def __init__(self, param, data_loader):
        super(CodeBook, self).__init__()

        self.param = param

        self.Protein_Encoder = GCN_Encoder(param, data_loader)
        self.Protein_Decoder = GCN_Decoder(param)

        self.vq_layer = VectorQuantizer(param['prot_hidden_dim'], param['num_embeddings'], param['commitment_cost'])

    def forward(self, batch_graph):
        z = self.Protein_Encoder.encoding(batch_graph)
        e, e_q_loss, encoding_indices = self.vq_layer(z)

        x_recon = self.Protein_Decoder.decoding(batch_graph, e)
        recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

        mask = torch.bernoulli(torch.full(size=(self.param['num_embeddings'],), fill_value=self.param['mask_ratio'])).bool().to(device)
        mask_index = mask[encoding_indices]
        e[mask_index] = 0.0

        x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e)


        x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
        y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
        mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param['sce_scale']))
        
        return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)


class VectorQuantizer(nn.Module):

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x):    
        x = F.normalize(x, p=2, dim=-1)
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)

        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss, encoding_indices
    
    def get_code_indices(self, x):

        distances = (
            torch.sum(x ** 2, dim=-1, keepdim=True) +
            torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
            2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
        )
        
        encoding_indices = torch.argmin(distances, dim=1)
        
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)
    