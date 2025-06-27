import CrabGOAT_classes
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
# why double import TODO
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
from tqdm import tqdm
import networkx as nx
import torch
from torch_geometric.data import DataLoader

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Caprettina',
                    description='Palline...')

    parser.add_argument('-d', '--dir')
    parser.add_argument('-s', '--samplesheet')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-e', '--edges')
    parser.add_argument('-m', '--model')
    parser.add_argument('-b', '--batchsize')
    parser.add_argument('-l', '--learnrate')
    parser.add_argument('-i', '--hiddenlayer')
	
    args = parser.parse_args()

    if args.samplesheet is None or args.dir is None or args.prefix is None or args.edges is None or args.model is None or args.batchsize is None or args.learnrate is None or args.hiddenlayer is None:
        parser.print_help()
        sys.exit()
    # TODO check existance and minimal error reporting
    ss = pd.read_csv(args.samplesheet)
    dir = args.dir
    my_params_name = args.model + "_b" + args.batchsize + "-l" + args.learnrate + "-h" + args.hiddenlayer
    path_gg=args.edges

    general_graph=pd.read_csv(path_gg)
    # adding backwards edges to have an undirected graph (super meh way)
    ed_ts = general_graph.apply(tuple, axis=1)
    ed_st = [x[::-1] for x in ed_ts]
    #print(len(ed_st))
    ed_all = ed_st + [x for x in ed_ts]
    #print(ed_all) # wtf...
    und_general_graph = pd.DataFrame(data={'source': [x[0] for x in ed_all],
                                      'target': [x[1] for x in ed_all]})
    geni_unici = pd.unique(und_general_graph[['source', 'target']].values.ravel())
    gene_to_idx = {gene: idx for idx, gene in enumerate(sorted(geni_unici))}
    idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
    edge_index = np.array([
        [gene_to_idx[src] for src in und_general_graph['source']],
        [gene_to_idx[tgt] for tgt in und_general_graph['target']]
    ])

    wt = ss.loc[ss['genotype']=='wt']['file'].values
    kras =  ss.loc[ss['genotype']=='kras']['file'].values

    samples_train = ss.loc[ss['test']=="no"]['sample'].values
    samples_test  = ss.loc[ss['test']=="yes"]['sample'].values
    # TODO minimal consistency checking of sample sheet
    
    dataset = CrabGOAT_classes.My_Graph_data(root=dir, edge_index=edge_index, gene_to_idx=gene_to_idx, wt=wt, kras=kras)


    records = []
    for g in dataset:
        tumor = g.tumor_id
        num_nodes = g.num_nodes
        num_edges = g.num_edges
        records.append({'tumor_id': tumor, 'num_nodes': num_nodes, 'num_edges': num_edges})

    #df = pd.DataFrame(records)
    #plt.figure(figsize=(10, 6))
    #sns.boxplot(x='tumor_id', y='num_nodes', data=df)
    #plt.title('Distribuzione del numero di nodi per CRC')
    #plt.xlabel('ID')
    #plt.ylabel('Numero di nodi')
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.savefig(os.path.join(dir, 'nodes.png'))
    #plt.show()
    #plt.figure(figsize=(10, 6))
    #sns.boxplot(x='tumor_id', y='num_edges', data=df)
    #plt.title('Distribuzione del numero di archi per tumore')
    #plt.xlabel('Tumor ID')
    #plt.ylabel('Numero di archi')
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.savefig(os.path.join(dir, 'edges.png'))
    #plt.show()
    #stats = df.groupby('tumor_id')[['num_nodes', 'num_edges']].describe()
    #print(stats)
    
    all_graphs = dataset.graphs

    train_graphs = [g for g in all_graphs if g.tumor_id in samples_train]
    test_graphs  = [g for g in all_graphs if g.tumor_id in samples_test]

    labels = np.array([int(g.y.item()) for g in train_graphs]) # estraggo le labels wt o KRAS
    print(labels)
    groups = np.array([g.tumor_id for g in train_graphs]) #estraggo l'ID del paziente per ogni grafo
    print(groups)
    unique_ids, inverse_idx = np.unique(groups, return_inverse=True) # rendo unico l'ID perchè è uguale per tutte le cellule appartenete allo stesso sample e inverse_idx ??
    print(unique_ids)
    print(inverse_idx)
    # VODKA non capisce lo [0][0], np.where ritorna un array di array ma perchè? cmq prendiamo la y della prima cellula per ogni campione
    labels_per_id = np.array([labels[np.where(inverse_idx == i)[0][0]] for i in range(len(unique_ids))]) #assegno la labels wt o KRAS all'ID
    print(labels_per_id)

    # VODKA TODO
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42) # TODO PARAM?
    train_id_idx, val_id_idx = next(splitter.split(unique_ids.reshape(-1,1), labels_per_id))
    train_ids = unique_ids[train_id_idx]
    val_ids   = unique_ids[val_id_idx]

    # Crea train/val split mantenendo integrità dei gruppi
    train_split = [g for g in train_graphs if g.tumor_id in train_ids]
    val_split   = [g for g in train_graphs if g.tumor_id in val_ids]


    train_loader = DataLoader(train_split, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_split, batch_size=64) # TODO PARAM
    test_loader  = DataLoader(test_graphs, batch_size=64)
    #  UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO
    if args.model == 'GAT'
        model = CrabGOAT_classes.GraphSAGEResNetGraph(in_channels=dataset.num_node_features,
                                 hidden_channels=64,
                                 dropout=0.1,
                                 pool_ratio=0.9).to(device)
    elif args.model == "transformer":
        model = CrabGOAT_classes.GraphSAGEResNetGraph(in_channels=dataset.num_node_features,
                                 hidden_channels=64,
                                 dropout=0.1,
                                 pool_ratio=0.9).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    num_epochs = 500 #  500 ## TODO PARAM
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(1, num_epochs + 1):
        # Train step
        tr_loss, tr_acc = CrabGOAT_classes.train(model, train_loader, optimizer, criterion, device) ## VODKA HERE
        # Validation step
        v_loss, v_acc = CrabGOAT_classes.evaluate(model, val_loader, criterion, device)

        # Save metrics
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        # Print progress
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}", flush=True)

    torch.save(model, os.path.join(dir, my_params_name + '_model.pt'))
    torch.save(model.state_dict(), os.path.join(dir, my_params_name + '_model_weights.pt')) # optim?
    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running
