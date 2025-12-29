import csv
import nni
import time
import json
import math
import copy
import argparse
import warnings
import numpy as np
import esm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import gc
import pandas as pd
from utils import *
from model import *
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import re
from sklearn.metrics import f1_score
import os
import json
import random
import itertools
from collections import defaultdict
import pickle

def get_ppi_label(df_merged3,dataset,split_mode,seed,path,flag='train'):
    df_merged3 = df_merged3.copy()
    df_merged3.loc[:, 'mode'] = 'reaction'
    if os.path.exists(path+"{}_graph{}_ppi.pkl".format(dataset,dataset)):
        with open(path+"{}_graph{}_ppi.pkl".format(dataset,dataset), "rb") as tf:
            ppi_list = pickle.load(tf)
        with open(path+"{}_graph{}_ppi_label.pkl".format(dataset,dataset), "rb") as tf:
            ppi_label_list = pickle.load(tf)

        with open(path+"{}_graph{}_protein_name.pkl".format(dataset,dataset), 'rb') as tf:
            protein_name = pickle.load(tf)
    else:
        name = 0
        ppi_name = 0
        protein_name = {}
        ppi_dict = {}
        ppi_list = []
        ppi_label_list = []
        class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 
                    'inhibition':4, 'catalysis':5, 'expression':6}

        # 所有蛋白质 ID（大小写不敏感)
        all_proteins = set()
        for idx, row in df_merged3.iterrows():
            protein_a = row['item_id_a'].strip().upper()  # 统一小写并去除空格
            protein_b = row['item_id_b'].strip().upper()
            all_proteins.update([protein_a, protein_b])

        # 为所有蛋白质分配唯一编号
        for protein in sorted(all_proteins):  # 排序保证可复现性
            if protein not in protein_name:
                protein_name[protein] = name
                name += 1

        for idx, row in df_merged3.iterrows():
            protein_a = row['item_id_a'].strip().upper()
            protein_b = row['item_id_b'].strip().upper()

            # 生成标准化边标识符（排序后小写）
            sorted_proteins = sorted([protein_a, protein_b])
            temp_data = "__".join(sorted_proteins)

            if temp_data not in ppi_dict:
                ppi_dict[temp_data] = ppi_name
                temp_label = [0] * 7
                temp_label[class_map[row['mode']]] = 1
                ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = ppi_dict[temp_data]
                temp_label = ppi_label_list[index]
                temp_label[class_map[row['mode']]] = 1
                ppi_label_list[index] = temp_label

        ppi_list = []
        for ppi_key in ppi_dict.keys():
            protein_a, protein_b = ppi_key.split('__')
            ppi_list.append([protein_name[protein_a], protein_name[protein_b]])

        # 验证节点映射一致性
        max_node_id = max(protein_name.values())
        for edge in ppi_list:
            assert edge[0] <= max_node_id, f"节点 {edge[0]} 超出范围！"
            assert edge[1] <= max_node_id, f"节点 {edge[1]} 超出范围！"

        assert len(ppi_list) == len(ppi_label_list), "边数量与标签数量不匹配！"

        with open(path+"{}_graph{}_ppi.pkl".format(dataset,dataset), "wb") as tf:
            pickle.dump(ppi_list, tf)
        with open(path+"{}_graph{}_ppi_label.pkl".format(dataset,dataset), "wb") as tf:
            pickle.dump(ppi_label_list, tf)
        with open(path+"{}_graph{}_protein_name.pkl".format(dataset,dataset), 'wb') as f:
            pickle.dump(protein_name, f)
            
    ppi_g = dgl.to_bidirected(dgl.graph(ppi_list)) # 7825# 无向图转换成有向图
    node_list, r_edge_list, k_edge_list = data_processing_new(df_merged3,dataset,path)
    protein_data = ProteinDatasetDGL(r_edge_list, k_edge_list, node_list, dataset,path)
    if flag == 'train': 
        ppi_split_dict,global_edges = split_dataset(ppi_list, dataset, split_mode, seed,path)
        return protein_name,protein_data, ppi_g.to(device), ppi_list, torch.FloatTensor(np.array(ppi_label_list)).to(device),ppi_split_dict,global_edges
    else:
        ppi_split_dict = split_dataset_old(ppi_list, dataset, split_mode, seed,path)
    
    return protein_name,protein_data, ppi_g.to(device), ppi_list, torch.FloatTensor(np.array(ppi_label_list)).to(device),ppi_split_dict

def split_dataset_old(ppi_list, dataset, split_mode, seed,path):
    if not os.path.exists(path+"{}_graph_{}_{}.json".format(dataset, dataset,split_mode)):
        if split_mode == 'random':
            ppi_num = len(ppi_list)
            random_list = [i for i in range(ppi_num)]
            random.shuffle(random_list)

            ppi_split_dict = {}
            ppi_split_dict['train_index'] = random_list[: int(ppi_num*0.6)]
            ppi_split_dict['val_index'] = random_list[int(ppi_num*0.6) : int(ppi_num*0.8)]
            ppi_split_dict['test_index'] = random_list[int(ppi_num*0.8) :]

            jsobj = json.dumps(ppi_split_dict)
            with open(path+"{}_graph_{}_{}.json".format(dataset,dataset, split_mode), 'w') as f:
                f.write(jsobj)
                f.close()
        
        else:
            print("your mode is {}, you should use bfs, dfs or random".format(split_mode))
            return
    else:
        with open(path+"{}_graph_{}_{}.json".format(dataset,dataset, split_mode), 'r') as f:
            ppi_split_dict = json.load(f)
            f.close()

    return ppi_split_dict

class ProteinDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, prot_r_edge, prot_k_edge, prot_node, dataset,path):
        
        if os.path.exists(path+"{}_graph{}_protein_graphs.pkl".format(dataset,dataset)):
            with open(path+"{}_graph{}_protein_graphs.pkl".format(dataset,dataset), "rb") as tf:
                self.prot_graph_list = pickle.load(tf)
        
        else:
            
            # prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
            # prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)
            # prot_node = torch.load(prot_node_path)

            self.prot_graph_list = []

            for i in range(len(prot_r_edge)):
                prot_seq = []
                for j in range(prot_node[i].shape[0]-1):
                    prot_seq.append((j, j+1))
                    prot_seq.append((j+1, j))

                # prot_g = dgl.graph(prot_edge[i]).to(device)
                prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid') : prot_seq, 
                                          ('amino_acid', 'STR_KNN', 'amino_acid') : prot_k_edge[i],
                                          ('amino_acid', 'STR_DIS', 'amino_acid') : prot_r_edge[i]}).to(device)
                prot_g.ndata['x'] = torch.FloatTensor(prot_node[i]).to(device)

                self.prot_graph_list.append(prot_g)

            with open(path+"{}_graph{}_protein_graphs.pkl".format(dataset,dataset), "wb") as tf:
                pickle.dump(self.prot_graph_list, tf)

    def __len__(self):
        return len(self.prot_graph_list)

    def __getitem__(self, idx):
        return self.prot_graph_list[idx]
    
def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)

def match_feature(x, all_for_assign):
    x_p = np.zeros((len(x), 7))
    for j in range(len(x)):
        if x[j] == 'ALA':
            x_p[j] = all_for_assign[0, :]
        elif x[j] == 'CYS':
            x_p[j] = all_for_assign[1, :]
        elif x[j] == 'ASP':
            x_p[j] = all_for_assign[2, :]
        elif x[j] == 'GLU':
            x_p[j] = all_for_assign[3, :]
        elif x[j] == 'PHE':
            x_p[j] = all_for_assign[4, :]
        elif x[j] == 'GLY':
            x_p[j] = all_for_assign[5, :]
        elif x[j] == 'HIS':
            x_p[j] = all_for_assign[6, :]
        elif x[j] == 'ILE':
            x_p[j] = all_for_assign[7, :]
        elif x[j] == 'LYS':
            x_p[j] = all_for_assign[8, :]
        elif x[j] == 'LEU':
            x_p[j] = all_for_assign[9, :]
        elif x[j] == 'MET':
            x_p[j] = all_for_assign[10, :]
        elif x[j] == 'ASN':
            x_p[j] = all_for_assign[11, :]
        elif x[j] == 'PRO':
            x_p[j] = all_for_assign[12, :]
        elif x[j] == 'GLN':
            x_p[j] = all_for_assign[13, :]
        elif x[j] == 'ARG':
            x_p[j] = all_for_assign[14, :]
        elif x[j] == 'SER':
            x_p[j] = all_for_assign[15, :]
        elif x[j] == 'THR':
            x_p[j] = all_for_assign[16, :]
        elif x[j] == 'VAL':
            x_p[j] = all_for_assign[17, :]
        elif x[j] == 'TRP':
            x_p[j] = all_for_assign[18, :]
        elif x[j] == 'TYR':
            x_p[j] = all_for_assign[19, :]
            
    return x_p



def read_atoms(file, chain="."):
    pattern = re.compile(chain) 

    atoms = []
    ajs = []
    
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ajs_id = line[17:20]
                atoms.append((x, y, z))
                ajs.append(ajs_id)
                
    return atoms, ajs # 

def compute_contacts(atoms, threshold): # 计算原子对之间的距离
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
                contacts.append((j, i))
    return contacts

def knn(atoms, k=5):
    
    x = np.zeros((len(atoms), len(atoms)))
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            x[i, j] = dist(atoms[i], atoms[j])
    index = np.argsort(x, axis=-1)
    
    contacts = []
    for i in range(len(atoms)):
        num = 0
        for j in range(len(atoms)):
            if index[i, j] != i and index[i, j] != i-1 and index[i, j] != i+1:
                contacts.append((i, index[i, j]))
                num += 1
            if num == k:
                break
            
    return contacts

def pdb_to_cm(file, threshold):
    atoms, x = read_atoms(file)
    r_contacts = compute_contacts(atoms, threshold) 
    k_contacts = knn(atoms)
    return r_contacts, k_contacts, x

def data_processing_new(df,dataset,path):
    distance = 10
    name_list = df['item_id_a'].tolist() + df['item_id_b'].tolist()  
    name_list = list(set(name_list))  # 删除重复项
    node_list = []
    r_edge_list = []
    k_edge_list = []
    i = 0
    url_pdb = "/data/dengrui/codebase/DeepPPI/swissprot_pdb_v4_20250603_unzipped"
    all_for_assign = np.loadtxt("/data/dengrui/codebase/DeepPPI/data_1/process_data_string/all_assign.txt")
    # 文件路径
    rball_file = path+"{}_graphprotein.rball.edges.{}.npy".format(dataset,dataset)
    knn_file = path+"{}_graphprotein.knn.edges.{}.npy".format(dataset,dataset)
    node_file = path+"{}_graphprotein.nodes.{}.pt".format(dataset,dataset)

    if os.path.exists(rball_file) and os.path.exists(knn_file) and os.path.exists(node_file):
    # i = 1
    # if i == 0:
        print("Files already exist. Loading the existing files...")
        r_edge_list = np.load(rball_file, allow_pickle=True)
        k_edge_list = np.load(knn_file, allow_pickle=True)
        node_list = torch.load(node_file)
    else:
        print("Files do not exist. Generating new files...")
        for name in tqdm(name_list):
            pdb_file_name = 'AF-'+name + '-F1-model_v4.pdb'
            # 假设pdb_to_cm和match_feature是已经定义的函数
            r_contacts, k_contacts, x = pdb_to_cm(open(url_pdb + "/" + pdb_file_name, "r"), distance)
            x = match_feature(x, all_for_assign)

            node_list.append(x)
            r_edge_list.append(r_contacts)
            k_edge_list.append(k_contacts)

        # 保存生成的数据
        np.save(rball_file, np.array(r_edge_list, dtype=object))
        np.save(knn_file, np.array(k_edge_list, dtype=object))
        torch.save(node_list, node_file)
        
    return node_list, r_edge_list, k_edge_list




def split_dataset(ppi_list, dataset, split_mode, seed,path):
    # 生成所有可能的蛋白质对
    all_proteins = sorted({p for edge in ppi_list for p in edge[:2]})
    all_possible_edges = list(itertools.combinations(all_proteins, 2))
    true_edges_set = set(tuple(sorted(edge)) for edge in ppi_list)
    
    # 生成真实负样本池（排除所有正样本）
    negative_pool = [edge for edge in all_possible_edges if tuple(sorted(edge)) not in true_edges_set]
    random.shuffle(negative_pool)
    # path+"{}_graph{}_protein_graphs.pkl".format(dataset,dataset)
    save_path = path + f"{dataset}_graph{dataset}_{split_mode}.json"
    
    if not os.path.exists(save_path):
        # 创建包含元数据的字典
        global_edges = [tuple(sorted(e)) for e in ppi_list] + [tuple(sorted(n)) for n in negative_pool]

        ppi_split_dict = {
            "meta": {
                "all_proteins": all_proteins,
                "negative_edges": negative_pool,
                "global_edges": [tuple(sorted(e)) for e in ppi_list] + [tuple(sorted(n)) for n in negative_pool]
            },
            "train_pos": [],
            "train_neg": [],
            "val_pos": [],
            "val_neg": [],
            "test_pos": [],
            "test_neg": []
        }

        if split_mode == 'random':
            # 随机划分正样本
            random.shuffle(ppi_list)
            total_pos = len(ppi_list)
            
            train_pos = ppi_list[:int(total_pos*0.8)]
            val_pos = ppi_list[int(total_pos*0.8):int(total_pos*0.9)]
            test_pos = ppi_list[int(total_pos*0.9):]
            # 划分负样本（保证不重叠）
            total_neg = len(negative_pool)
            train_neg = negative_pool[:len(train_pos)]
            val_neg = negative_pool[len(train_pos):len(train_pos)+len(val_pos)]
            test_neg = negative_pool[len(train_pos)+len(val_pos):len(train_pos)+len(val_pos)+len(test_pos)]

        def get_indices(edge_list):
            return [ppi_split_dict["meta"]["global_edges"].index(tuple(sorted(edge))) for edge in edge_list]

        ppi_split_dict.update({
            "train_pos": get_indices(train_pos),
            "train_neg": get_indices(train_neg),
            "val_pos": get_indices(val_pos),
            "val_neg": get_indices(val_neg),
            "test_pos": get_indices(test_pos),
            "test_neg": get_indices(test_neg)
        })

        with open(save_path, 'w') as f:
            json.dump(ppi_split_dict, f, default=lambda x: list(x) if isinstance(x, set) else x)        
    else:
        print("Files already exist. Loading the existing files...")
        with open(save_path, 'r') as f:
            ppi_split_dict = json.load(f)

        ppi_split_dict["meta"] = {
            "all_proteins": list(ppi_split_dict["meta"]["all_proteins"]),
            "negative_edges": [tuple(edge) for edge in ppi_split_dict["meta"]["negative_edges"]],
            "global_edges": [tuple(edge) for edge in ppi_split_dict["meta"]["global_edges"]]
        }


    global_edges = ppi_split_dict["meta"]["global_edges"]
    print_split_stats(ppi_split_dict)
    return ppi_split_dict,global_edges


def print_split_stats(split_dict):
    stats = [
        ("Train", len(split_dict["train_pos"]), len(split_dict["train_neg"])),
        ("Val", len(split_dict["val_pos"]), len(split_dict["val_neg"])),
        ("Test", len(split_dict["test_pos"]), len(split_dict["test_neg"]))
    ]
    
    for name, pos, neg in stats:
        print(f"{name} Set: {pos} pos + {neg} neg = {pos+neg} total")
        print(f"  Pos/Neg Ratio: {pos/neg:.2f}:1")


def get_esm_embedding(data,protein_name,file_path):

    if os.path.exists(file_path):
        # 如果文件存在则加载
        empty_matrix  = np.load(file_path)

    else:
        set_uniprot_seq = pd.concat([
        data[['item_id_a', 'seq_A']].rename(columns={'item_id_a': 'uniprot_id', 'seq_A': 'seq'}),
        data[['item_id_b', 'seq_B']].rename(columns={'item_id_b': 'uniprot_id', 'seq_B': 'seq'})
        ], ignore_index=True)

        set_uniprot_seq = set_uniprot_seq.drop_duplicates()
        
        df = pd.DataFrame()
        df['uniprot_id'] = protein_name.keys()
        df = df.merge(set_uniprot_seq,how='left',on='uniprot_id')
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {device}")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        model = model.to(device)
        def get_rep_seq(sequences):

            batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            return token_representations.cpu().detach().numpy()

        def pad_or_truncate(df):
            max_length = 700
            # 将序列填充到最大长度
            df['seq'] = df['seq'].apply(lambda x: x + 'G' * (max_length - len(x)) if len(x) < max_length else x[:max_length])
            return df

        # 生成蛋白表示
        df = pad_or_truncate(df)
        # data = protein_name_seq_df.copy()
        df_data = list(zip(df.uniprot_id.index,df.seq))
        # 分批次运行
        stride =50
        num_iterations = len(df_data) // stride
        if len(df_data) % stride != 0:
            num_iterations += 1

        empty_matrix = np.empty((0, 702, 1280))
        
        for i in tqdm(range(num_iterations)):
            start = i * stride
            end = start + stride

            # 取出要处理的数据
            current_data = df_data[start:end]

            rep33 = get_rep_seq(sequences=current_data)
            empty_matrix = np.concatenate((empty_matrix, rep33), axis=0)
        np.save(file_path, empty_matrix)
    return empty_matrix

def collate(samples):
    # return dgl.batch_hetero(samples)
    return dgl.batch(samples)


def evaluator(model, ppi_g, prot_embed,string_embedding_esm2_tensor, ppi_list, labels, index, batch_size,mode='metric'):

    eval_output_list = []
    eval_labels_list = []
    eval_edges_list = []
    interaction_list = []
    eval_prob_list = []     # 存概率

    batch_num = math.ceil(len(index) / batch_size)


    with torch.no_grad():
        model.eval()
        for batch in range(batch_num):
            if batch == batch_num - 1:
                eval_idx = index[batch * batch_size:]
                eval_labels = labels[batch * batch_size:]

            else:
                eval_idx = index[batch * batch_size : (batch+1) * batch_size]
                eval_labels = labels[batch * batch_size : (batch+1) * batch_size]

            output,interaction = model(ppi_g, prot_embed,string_embedding_esm2_tensor, ppi_list, eval_idx)
            eval_labels = eval_labels.detach().cpu().numpy()

            eval_edges_list.append([ppi_list[i] for i in eval_idx])
            probs = torch.sigmoid(output).detach().cpu().squeeze().numpy()

            # output = torch.sigmoid(output).detach().cpu()
            y_pred = (output > 0.5).int()
            y_pred = y_pred.cpu().squeeze().numpy()
            if y_pred.ndim == 0:  # 如果是 0-d 数组
                y_pred = [y_pred]  # 封装成列表
            # eval_labels = labels[eval_idx].squeeze().numpy()
            if probs.ndim == 0:  # 如果是 0-d 数组
                probs = [probs]  # 封装成列表            
            eval_output_list.append(list(y_pred))
            eval_labels_list.append(eval_labels)
            eval_prob_list.append(list(probs))

            interaction_list.append(interaction)
        # sigmoid_data  = torch.sigmoid(output).detach().cpu()
        eval_output_array = np.concatenate(eval_output_list, axis=0)  # 将列表合并为一维数组
        eval_labels_array = np.concatenate(eval_labels_list, axis=0)  # 将列表合并为一维数组
        eval_edges_array = np.concatenate(eval_edges_list, axis=0)  # New concatenation
        eval_prob_array = np.concatenate(eval_prob_list, axis=0)

        # f1 = f1_score(eval_output_array, eval_labels_array, average='binary')  # 对于二分类任务

        
    if mode == 'metric':
        return 0
    elif mode == 'output':
        return eval_output_array,eval_labels_array,eval_edges_array,eval_prob_array,interaction_list



