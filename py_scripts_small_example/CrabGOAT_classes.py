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

#si generano i grafi partendo da quello generale e tenendo solo gli edge
#corrispondenti ai geni espressi nelle count matrix del sample
def generate_graph(path_expr_csv, edge_index, gene_to_idx, soglia=0.1):
    """
    path_expr_csv: path al file CSV (rows = cellule, cols = geni)
    edge_index: numpy array shape (2, N), già mappato con indici numerici==>dizionario(non nomi di geni)
    gene_to_idx: dict nome gene -> indice numerico
    soglia: valore minimo per considerare un gene espresso
    """
    expr_df = pd.read_csv(path_expr_csv, index_col=0)
    grafi = []
    num_total_genes = len(gene_to_idx)

    # VODKA MAPPING
    gene_names = [x[1] for x in expr_df.index.str.split(':')]
    d = collections.defaultdict(int)
    for x in gene_names:
        d[x] += 1
        gene_names_nodup = [x for x in gene_names if d[x] == 1]

    mask = expr_df.index.to_series().apply(lambda x: any(sub == x.split(':')[1] for sub in gene_names_nodup))
    expr_df = expr_df[mask]
    expr_df.index =  [x[1] for x in expr_df.index.str.split(':')]
    print(expr_df.shape)
    expr_df = expr_df[expr_df.index.isin(gene_to_idx.keys())]
    print(expr_df.shape)
    expr_df.sort_index()

    expr_df = expr_df.T
    
    #  TODO convert to undirected graph adding all bi-directional edges

    # Pre‐converto edge_index_global in torch.LongTensor perché lo riutilizziamo interamente
    full_edge_index = torch.tensor(edge_index, dtype=torch.long)

    # shape (M, 2) con coppie (u,v)
    edges_M2 = edge_index.T  # ora ogni riga è [u,v]
    skipped_cells = 0
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
            filtered_edges = edge_index[:, mask]
            edge_index_tensor = torch.tensor(filtered_edges, dtype=torch.long)
        else:
            # Se non ho geni sopra th salto la cella
            skipped_cells = skipped_cells + 1
            continue

        # 6) Creo l’oggetto Data, incluso cell_idx
        pyg_graph = Data(x=features,
                         edge_index=edge_index_tensor)
        pyg_graph.cell_idx = cell_id   # salva l'indice riga del CSV
        grafi.append(pyg_graph)

    print(f"{skipped_cells} for not expressed genes")
    return grafi

class My_Graph_data(Dataset):
    def __init__(self, root, edge_index, gene_to_idx, wt, kras, transform=None, pre_transform=None): #gli edge_inde e gene_index sono quelli del mapping e poi saranno filtrati per i geni presenti in quella cellula in generate_graph
        self.edge_index = edge_index
        self.gene_to_idx = gene_to_idx
        self.drive_folder = root
        super().__init__(root, transform, pre_transform)


        self.graphs_path = os.path.join(self.root, 'graphs.pt')

        # controllo se ho già creato i grafi o voglio ricaricarli (se voglio ricaricarli devo eliminare la cartella che li contiene)
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
        tumor_files = [f for f in os.listdir(self.drive_folder) if f.startswith(fileprefix)]

        for file in tumor_files:
            print(f"Processing {file}")
            path_expr = os.path.join(self.drive_folder, file)

            tumor_graphs = generate_graph(path_expr, self.edge_index, self.gene_to_idx, soglia=0.1)

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

            for g in tumor_graphs:
                g.y = torch.tensor([label], dtype=torch.long)
                g.tumor_id = tumor_id
                self.graphs.append(g)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

# graph sage with skip connection
class GraphSAGEResNetGraph(torch.nn.Module):
    """
    Graph-level binary classifier with GraphSAGE, residuals, and SAGPooling (TopK-based
    pooling with learnable score via a GNN) after each conv layer.
    Allows extracting graph embeddings from first MLP layer.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_channels : int
        Dimensionality of hidden layers.
    dropout : float, optional
        Dropout probability (default: 0.5).
    pool_ratio : float, optional
        Node retention ratio for each SAGPooling (default: 0.8).
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 dropout: float = 0.5, pool_ratio: float = 0.8):
        super().__init__()
        # GraphSAGE layers
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.pool1 = SAGPooling(hidden_channels, ratio=pool_ratio)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.pool2 = SAGPooling(hidden_channels, ratio=pool_ratio)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.pool3 = SAGPooling(hidden_channels, ratio=pool_ratio)

        # Global pooling for final representation
        self.global_pool = global_mean_pool

        # MLP head split into two stages to extract embeddings
        self.head1 = Linear(hidden_channels, hidden_channels)
        self.head_act = ReLU()
        self.head_drop = Dropout(dropout)
        self.head2 = Linear(hidden_channels, 1)

        # Activation and dropout for conv layers
        self.act = ReLU()
        self.drop = Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor, return_embedding: bool = False):
        # Layer 1: conv + act + drop + pool
        x1 = self.act(self.conv1(x, edge_index))
        x1 = self.drop(x1)
        x1, edge_index1, _, batch1, _, _ = self.pool1(x1, edge_index, None, batch)

        # Layer 2: conv + residual + act + drop + pool
        x2 = self.conv2(x1, edge_index1) + x1
        x2 = self.act(x2)
        x2 = self.drop(x2)
        x2, edge_index2, _, batch2, _, _ = self.pool2(x2, edge_index1, None, batch1)

        # Layer 3: conv + residual + act + drop + pool
        x3 = self.conv3(x2, edge_index2) + x2
        x3 = self.act(x3)
        x3 = self.drop(x3)
        x3, edge_index3, _, batch3, _, _ = self.pool3(x3, edge_index2, None, batch2)

        # Global pooling
        pooled = self.global_pool(x3, batch3)

        # First MLP layer -> graph embeddings
        embed = self.head1(pooled)
        embed = self.head_act(embed)
        embed = self.head_drop(embed)

        # Final MLP -> logits
        logits = self.head2(embed).view(-1)

        #se c'è return_embedding: True ritorna anche embed sennò solo logits
        if return_embedding:
            return logits, embed
        return logits

# perchè fuori dal modello VODKA HERE HERE
#@title TRAIN e TEST
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
