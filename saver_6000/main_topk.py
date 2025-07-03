import CrabGOAT_classes_topk
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
import gc
import time

from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import re
import torch
from torch.nn import Linear, Dropout, ReLU, Sequential
from torch_geometric.nn import SAGEConv, global_mean_pool,SAGPooling,BatchNorm
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR,OneCycleLR


import math
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Caprettina',
                    description='Palline...')

    parser.add_argument('-r', '--dir')
    parser.add_argument('-s', '--samplesheet')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-e', '--edges')
    parser.add_argument('-m', '--model')
    parser.add_argument('-b', '--batchsize')
    parser.add_argument('-l', '--learnrate')
    parser.add_argument('-i', '--hiddenlayer')
    parser.add_argument('-d', '--dropout')
    parser.add_argument('-n', '--numlayers')
    parser.add_argument('-k', '--topk')

    args = parser.parse_args()

    if args.samplesheet is None or args.dir is None or args.prefix is None or args.edges is None or args.model is None or args.batchsize is None or args.learnrate is None or args.hiddenlayer is None or args.dropout is None:
        parser.print_help()
        sys.exit()
    if args.model in ['GraphSage_128', 'GraphSage_512', 'GraphSage_topk'] and args.numlayers is None:
        parser.print_help()
        print('Sage has also -n param!')
        sys.exit()
    elif args.model in ['GraphSage_128', 'GraphSage_512', 'GraphSage_topk']:
        par_num_layers = int(args.numlayers)

    if args.model == 'GraphSage_topk' and args.topk is None:
        parser.print_help()
        print('Sage_topk has also -k param!')
        sys.exit()
    elif args.model == 'GraphSage_topk':
        par_topk = float(args.topk)

    ss = pd.read_csv(args.samplesheet)
    dir = args.dir
    if args.model == 'GraphSage_128':
        my_params_name = "GraphSage_b128-l" + args.learnrate + "-h" + args.hiddenlayer + "-d" + args.dropout + '-n' +  args.numlayers
    elif args.model == 'GraphSage_512':
        my_params_name = "GraphSage_b512-l" + args.learnrate + "-h" + args.hiddenlayer + "-d" + args.dropout + '-n' + args.numlayers
    elif args.model == 'GraphSage_topk':
        my_params_name = "GraphSage_topk-b"+ args.batchsize+ "-l" + args.learnrate + "-h" + args.hiddenlayer + "-d" + args.dropout + '-n' + args.numlayers + '-k' + args.topk
    else:        
        my_params_name = args.model + "_b" + args.batchsize + "-l" + args.learnrate + "-h" + args.hiddenlayer + "-d" + args.dropout

    print(my_params_name, flush=True)
    
    par_batch_size = int(args.batchsize)
    par_learn_rate = float(args.learnrate)
    par_hidden_channels = int(args.hiddenlayer)
    par_dropout = float(args.dropout)
    
    path_gg=args.edges
    general_graph=pd.read_csv(path_gg)

    geni_unici = pd.unique(general_graph[['source', 'target']].values.ravel())
    gene_to_idx = {gene: idx for idx, gene in enumerate(sorted(geni_unici))}
    idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
    edge_index = np.array([
        [gene_to_idx[src] for src in general_graph['source']],
            [gene_to_idx[tgt] for tgt in general_graph['target']]
            ])

    wt = ss.loc[ss['genotype']=='wt']['file'].values
    kras =  ss.loc[ss['genotype']=='kras']['file'].values

    samples_train = ss.loc[ss['test']=="no"]['sample'].values
    samples_test  = ss.loc[ss['test']=="yes"]['sample'].values
    # TODO minimal consistency checking of sample sheet
    
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.utils.data import  WeightedRandomSampler
    from collections import Counter

    dataset = CrabGOAT_classes_topk.Graph_data(root=dir, edge_index=edge_index, gene_to_idx=gene_to_idx, wt=wt, kras=kras)
    all_graphs = dataset.graphs

    # 1) Separa i test
    test_ids     = samples_test
    test_graphs  = [g for g in all_graphs if g.tumor_id in test_ids]
    rem_graphs   = [g for g in all_graphs if g.tumor_id not in test_ids]

    # 2) Split 70/30 a livello di grafo stratificando per label
    labels = np.array([int(g.y.item()) for g in rem_graphs])
    train_graphs, val_graphs = train_test_split(
        rem_graphs,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    print(f"# train raw: {len(train_graphs)}, # val: {len(val_graphs)}", flush=True)
    print(" Train label distrib:", Counter(int(g.y.item()) for g in train_graphs), flush=True)
    print(" Val   label distrib:", Counter(int(g.y.item()) for g in val_graphs), flush=True)

    # 3) Costruisci un WeightedRandomSampler per bilanciare 50/50 nel loader
    train_labels = np.array([int(g.y.item()) for g in train_graphs])
    class_counts = np.bincount(train_labels)                # [#WT, #KRAS]
    class_weights = 1.0 / class_counts                       # peso inverso
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # uso tutti i train_graphs
       replacement=True
    )
    
    gc.collect()
    torch.cuda.empty_cache()

    num_epochs = 500
    best_val_loss = 10
    patience_counter = 0
    patience_limit = 50
    batch_size=par_batch_size
    hidden_channels=par_hidden_channels
    warmup_epochs=10 # was 20 for Transformers
    base_lr=par_learn_rate   #base_lr=[0,001,0.0005,0.00001]
    
    from torch_geometric.loader import DataLoader

    # Se hai un sampler pesato:
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,          # WeightedRandomSampler definito da te
        drop_last=False,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False
    )


    test_loader  = DataLoader(test_graphs, batch_size=batch_size)

    if args.model in  ["GraphSage_128", "GraphSage_512"]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    elif args.model == "GraphSage_topk":
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'GAT':
        model = CrabGOAT_classes_topk.GATv2Net(
        in_channels=dataset.num_node_features,
        hidden_channels=hidden_channels,
        heads=2,
        num_layers=2,
        dropout=par_dropout
        ).to(device)    
    elif args.model == "transformer_3t_2l":
        model = CrabGOAT_classes_topk.GraphTransformerNet(
            in_channels=dataset.num_node_features,
            hidden_channels=hidden_channels,
            heads=3,
            num_layers=2,
            dropout=par_dropout,
            use_pos_enc=True).to(device) ### CAAAVOLO 
    elif args.model == "transformer_2t_2l":
        model = CrabGOAT_classes_topk.GraphTransformerNet(
            in_channels=dataset.num_node_features,
            hidden_channels=hidden_channels,
            heads=2,
            num_layers=2,
            dropout=par_dropout,
            use_pos_enc=False).to(device)
    elif args.model in ["GraphSage_128", "GraphSage_512", "GraphSage_topk"]:
        model = CrabGOAT_classes_topk.GraphSAGESkipNet(
            in_channels=dataset.num_node_features,
            hidden_channels=hidden_channels,
            num_layers=par_num_layers,
            dropout=par_dropout, ratio=par_topk,
            use_pos_enc=False).to(device)

    def make_scheduler(optimizer, total_epochs=num_epochs, warmup_epochs=warmup_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            # normalized epoch per cosine (0 → 1)
            progress = float(epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)

    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, total_epochs=num_epochs, warmup_epochs=warmup_epochs)
    num_neg = Counter(int(g.y.item()) for g in train_graphs)[0]
    num_pos = Counter(int(g.y.item()) for g in train_graphs)[1]
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []


    for epoch in range(1, num_epochs + 1):
        # Train step
        tr_loss, tr_acc = CrabGOAT_classes_topk.train(model, train_loader, optimizer, criterion, device)
        # Validation step
        v_loss, v_acc = CrabGOAT_classes_topk.evaluate(model, val_loader, criterion, device)

        # Step del scheduler sulla val loss
        scheduler.step(v_loss)

        # Save metrics
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        # Early stopping + checkpoint
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(dir, my_params_name+'best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1

        # Print cose
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.8f}", flush=True)

        # se on scende mi fermo
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience_limit} epochs)", flush=True)
            break
    
    # Alla fine, carica i pesi del miglior modello (e ripulisce la GPU sperando di starci)
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(os.path.join(dir, my_params_name+'best_model.pt')))

    def test(model, loader, criterion, device):
        """Evaluate the model on the test set, returning loss, accuracy and confusion per tumor_id."""
        model.eval()
        total_loss = correct = total = 0
        # Initialize confusion dict: {tumor_id: [TP, FP, FN, TN]}
        tumor_confusion = {}

        for data in loader:
            # data.tumor_id is a list of tumor_id strings for each graph in the batch
            tumor_ids = data.tumor_id
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y.float())
            total_loss += loss.item() * data.num_graphs
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += int((preds == data.y).sum())
            total += data.num_graphs

            # update confusion per tumor
            y_true = data.y.cpu().long().tolist()
            y_pred = preds.cpu().long().tolist()
            for tid, true_label, pred_label in zip(tumor_ids, y_true, y_pred):
                if tid not in tumor_confusion:
                    tumor_confusion[tid] = [0, 0, 0, 0]  # TP, FP, FN, TN
                tp, fp, fn, tn = tumor_confusion[tid]
                if true_label == 1 and pred_label == 1:
                    tp += 1
                elif true_label == 0 and pred_label == 1:
                    fp += 1
                elif true_label == 1 and pred_label == 0:
                    fn += 1
                elif true_label == 0 and pred_label == 0:
                    tn += 1
                tumor_confusion[tid] = [tp, fp, fn, tn]

        avg_loss = total_loss / total
        acc = correct / total
        return avg_loss, acc, tumor_confusion
    
    # ==== Run test ====
    test_loss, test_acc, tumor_confusion = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}", flush=True)
    print("Confusion matrix per tumor_id:", flush=True)
    for tid, (tp, fp, fn, tn) in tumor_confusion.items():
        print(f"{tid}: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)

    # ==== Confusion matrix plot ====
    import matplotlib.pyplot as plt

    # Prepare data for plotting
    tumor_ids = list(tumor_confusion.keys())
    metrics = ['TP', 'FP', 'FN', 'TN']
    # Build matrix: rows=tumor_ids, cols=metrics
    conf_matrix = [[tumor_confusion[tid][i] for i in range(4)] for tid in tumor_ids]


    x = range(len(tumor_ids))
    width = 0.2
    plt.figure()
    for i, metric in enumerate(metrics):
        values = [row[i] for row in conf_matrix]
        plt.bar([xi + i*width for xi in x], values, width=width)

    plt.xticks([xi + 1.5*width - width/2 for xi in x], tumor_ids)
    plt.xlabel('Tumor ID')
    plt.ylabel('Count')
    plt.title('Confusion matrix components per Tumor ID')
    plt.legend(metrics)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, my_params_name+'-confusion.png'))
    plt.show()

    def plot_metrics(train_losses, val_losses, train_accs, val_accs):
        """
        Plots loss and accuracy curves.

        Parameters:
            - train_losses, val_losses: lists of float
            - train_accs, val_accs: lists of float
        """
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, label='Train Acc')
        plt.plot(epochs, val_accs, label='Val Acc')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(dir, my_params_name+'-loss.png'))
        plt.show()

    plot_metrics(train_losses,val_losses,train_accs,val_accs)
    res = pd.DataFrame(data={'train_loss': train_losses, 'val_loss': val_losses, 'train_accs': train_accs, 'val_accs': val_accs})
    res.to_csv(os.path.join(dir, my_params_name+'-loss_accs.csv'))
