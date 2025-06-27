from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch_geometric
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_networkx
import numpy as np
import os
from tqdm import tqdm
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import torch.optim as optim

from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import re
from torch.nn import Linear, Dropout, ReLU, Sequential
from torch_geometric.nn import SAGEConv, global_mean_pool,SAGPooling
from torch_geometric.data import DataLoader
import collections

import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv,
    SAGPooling,
    global_mean_pool as gap,
    global_max_pool as gmp,TransformerConv
)
from torch_geometric.utils import dropout_adj

import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool as gap
from torch_geometric.nn import GATv2Conv, TransformerConv

from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch.nn import Linear, Dropout, ReLU, Sequential
from torch_geometric.nn import SAGEConv, global_mean_pool,SAGPooling,BatchNorm
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR,OneCycleLR
import math

def generate_graph(
    path_expr_csv: str,
    edge_index_global: np.ndarray,  # shape (2, M), indici globali 0..(num_genes-1)
    gene_to_idx: dict,
    soglia: float = 1
) -> list[Data]:
    """
    Per ciascuna cella nel CSV, genera un grafo PyG con:
      - tutti i num_total_genes nodi (feature=0 se < soglia)
      - edge_index filtrato: rimuove archi che coinvolgono nodi sotto‐soglia
      - g.cell_idx = l'indice riga (cell_id) del CSV
    """

    # 1) Leggi il CSV (righe = cellule, colonne = geni; l'indice della riga è cell_id)
    expr_df = pd.read_csv(path_expr_csv, index_col=0)

    num_total_genes = len(gene_to_idx)
    grafi = []

    # Pre‐converto edge_index_global in torch.LongTensor perché lo riutilizziamo interamente
    full_edge_index = torch.tensor(edge_index_global, dtype=torch.long)

    # shape (M, 2) con coppie (u,v)
    edges_M2 = edge_index_global.T  # ora ogni riga è [u,v]

    for cell_id, cell_expr in expr_df.iterrows():

        # 2) Creiamo features=(num_total_genes,1), tutte a zero
        features = torch.zeros((num_total_genes, 1), dtype=torch.float32)

        # 3) salvo i geni sopra th e li tengo
        expressed = cell_expr[cell_expr > soglia]
        valid_genes = [g for g in expressed.index if g in gene_to_idx]
        valid_indices = {gene_to_idx[g] for g in valid_genes}

        # 4) Riempo il vettore di feature per i geni espressi, quindi tutti gli altri hanno zero
        for gene_name, val in expressed.items():
            if gene_name in gene_to_idx:
                gi = gene_to_idx[gene_name]
                features[gi, 0] = float(val)

        # 5) Filtriamo gli archi: manteniamo solo quelli (u,v) in cui sia u che v sono in valid_indices
        if len(valid_indices) > 0:
            # creiamo una maschera boolean (M,) che indica quali archi tenere
            mask = np.array([(u in valid_indices and v in valid_indices) for u, v in edges_M2],
                            dtype=bool)
            # applico la maschera a edge_index_global (2×M)
            filtered_edges = edge_index_global[:, mask]
            edge_index_tensor = torch.tensor(filtered_edges, dtype=torch.long)
        else:
            # Se non ho geni sopra th salto la cella
            continue

        # 6) Creo l’oggetto Data, incluso cell_idx
        pyg_graph = Data(x=features,
                         edge_index=edge_index_tensor)
        pyg_graph.cell_idx = cell_id   # salva l'indice riga del CSV
        grafi.append(pyg_graph)

    return grafi



class Graph_data(Dataset):
    def __init__(self, root, edge_index, gene_to_idx, wt, kras, transform=None, pre_transform=None):
        self.edge_index = edge_index
        self.gene_to_idx = gene_to_idx
        self.drive_folder = root
        super().__init__(root, transform, pre_transform)

        self.graphs_path = os.path.join(self.root, 'graphs.pt')

        if os.path.exists(self.graphs_path):
            print("Carico grafi già salvati")
            self.graphs = torch.load(self.graphs_path, weights_only=False)
        else:
            print("Creo e salvo i grafi")
            self.graphs = []
            self._process_graphs(wt, kras)
            torch.save(self.graphs, self.graphs_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['graphs.pt']

    def download(self):
        pass

    def _process_graphs(self, wt, kras, fileprefix='filtered_annotated'):
        tumor_files = [f for f in os.listdir(self.drive_folder) if f.startswith(file_prefix)]

        #wt = ['filtered_CRC0322_NT_1_3000.csv', 'filtered_CRC0327_NT_2.csv', 'filtered_CRC0542_NT72h_1.csv']
        #kras = ['filtered_CRC1139_NT_1.csv', 'filtered_CRC1502_NT_1.csv', 'filtered_CRC1620_NT_1.csv']

        for file in tumor_files:
            print(f"Processing {file}")
            path_expr = os.path.join(self.drive_folder, file)

            graphs = generate_graph(
                path_expr_csv = path_expr,
                edge_index_global = self.edge_index,
                gene_to_idx = self.gene_to_idx,
                soglia = 0.1
            )

            if file in kras:
                label = 1
            elif file in wt:
                label = 0
            else:
                raise ValueError(f"File {file} non assegnato a mutato o wt!")

            match = re.search(r'(CRC\d{4})', file)
            if match:
                tumor_id = match.group(1)
            else:
                raise ValueError(f"Impossibile estrarre CRC ID da file: {file}")

            for g in graphs:
                g.y = torch.tensor([label], dtype=torch.long)
                g.tumor_id = tumor_id # tolto per le discrasie di prove?
                self.graphs.append(g)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


class GatClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # primo layer: da in_channels → hidden
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        # layer successivi: dim = hidden*heads → hidden
        for _ in range(num_layers-1):
            self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout))

        self.lin = torch.nn.Linear(hidden_channels*heads, 1)
        self.act = torch.nn.ELU()
        self.dropout = dropout

    def forward(self, x, edge_index, batch, return_attn=False):
        attentions = []
        for conv in self.convs:
            x, (edge_index, attn) = conv(x, edge_index, return_attention_weights=True)
            attentions.append(attn)   # shape: [num_edges, heads]
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = gap(x, batch)
        logits = self.lin(x).squeeze(-1)
        return (logits, attentions) if return_attn else logits


# GATv2Net con attn weights, matching GatClassifier API
class GATv2Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # primo layer
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        # layer intermedi e finale
        for _ in range(num_layers-1):
            # mantiene hidden*heads → hidden
            self.convs.append(GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout))
        self.lin = torch.nn.Linear(hidden_channels*heads, 1)
        self.act = torch.nn.ELU()
        self.dropout = dropout

    def forward(self, x, edge_index, batch, return_attn=False):
        attentions = []
        for conv in self.convs:
            x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
            attentions.append(attn)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        logits = self.lin(x).squeeze(-1)
        return (logits, attentions) if return_attn else logits


# GraphTransformerNet con attn weights simile
class GraphTransformerNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=3, num_layers=2, dropout=0.3, use_pos_enc=False):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout, beta=True))
        for _ in range(num_layers-1):
            self.convs.append(TransformerConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout, beta=True))
        self.lin = torch.nn.Linear(hidden_channels*heads, 1)
        self.act = torch.nn.ELU()
        self.dropout = dropout
        self.use_pos_enc = use_pos_enc

    def forward(self, x, edge_index, batch, return_attn=False, pos_enc=None):
        attentions = []
        if self.use_pos_enc and pos_enc is not None:
            x = x + pos_enc
        for conv in self.convs:
            x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
            attentions.append(attn)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        logits = self.lin(x).squeeze(-1)
        return (logits, attentions) if return_attn else logits

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)
        loss = criterion(logits, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += int((preds == data.y).sum())
        total += data.num_graphs
    return total_loss / len(loader.dataset), correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, return_embedding=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_embeds = [] if return_embedding else None
    for data in loader:
        data = data.to(device)
        if return_embedding:
            logits, embeds = model(data.x, data.edge_index, data.batch, True)
            all_embeds.append(embeds.cpu())
        else:
            logits = model(data.x, data.edge_index, data.batch)
        loss = criterion(logits, data.y.float())
        total_loss += loss.item() * data.num_graphs
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += int((preds == data.y).sum())
        total += data.num_graphs
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total
    if return_embedding:
        return avg_loss, acc, torch.cat(all_embeds, dim=0)
    return avg_loss, acc
