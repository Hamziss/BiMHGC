from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import sys
import pickle as pkl
import scipy.sparse as sp
from evaluation import calculate_fmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import networkx as nx
import os
import random
import pandas as pd
import numpy as np
import torch
import pickle
from pandas.core.frame import DataFrame
from sklearn.manifold import TSNE
from datetime import datetime
from pathlib import Path

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1 :
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')
# filter by ambiguous amino acid
def find_amino_acid(x):
    return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)
# encode amino acid sequence using CT
def CT(sequence):
    classMap = {'G':'1','A':'1','V':'1','L':'2','I':'2','F':'2','P':'2',
            'Y':'3','M':'3','T':'3','S':'3','H':'4','N':'4','Q':'4','W':'4',
            'R':'5','K':'5','D':'6','E':'6','C':'7'}

    seq = ''.join([classMap[x] for x in sequence])
    length = len(seq)
    coding = np.zeros(343,dtype=np.int64)
    for i in range(length-2):
        index = int(seq[i]) + (int(seq[i+1])-1)*7 + (int(seq[i+2])-1)*49 - 1
        coding[index] = coding[index] + 1
    return coding
# 序列CT编码
def sequence_CT(gene_entry_seq):
    ambiguous_index = gene_entry_seq.loc[gene_entry_seq['Sequence'].apply(find_amino_acid)].index
    gene_entry_seq.drop(ambiguous_index, axis=0, inplace=True)
    gene_entry_seq.index = range(len(gene_entry_seq))
    print("after filtering:", gene_entry_seq.shape)
    print("encode amino acid sequence using CT...")
    CT_list = []
    for seq in gene_entry_seq['Sequence'].values:
        CT_list.append(CT(seq))
    gene_entry_seq['features_seq'] = CT_list

    return gene_entry_seq
# 过滤PPI
def preprocessing_PPI(PPI,Sequence):
    PPI_protein_list = list(set(PPI['protein1'].unique()).union(set(PPI['protein2'].unique())))
    PPI_protein = DataFrame(PPI_protein_list)
    PPI_protein['Entry'] = PPI_protein[0].apply(
        lambda x: Sequence[Sequence['Gene Names'].str.contains(x, case=False, na=False)]['Entry'].values[
            0] if Sequence['Gene Names'].str.contains(x, case=False).any() else 'NA')
    PPI_protein.columns = ['Gene_symbol', 'Entry']
    PPI_protein = PPI_protein[PPI_protein['Entry'] != 'NA']
    PPI_protein = PPI_protein[PPI_protein['Entry'].isin(set(Sequence['Entry']))]
    PPI_protein_list = list(set(PPI_protein['Gene_symbol'].unique()))
    #PPI = PPI.drop(['index'], axis=1)
    PPI = PPI[PPI['protein1'].isin(PPI_protein_list)]
    PPI = PPI[PPI['protein2'].isin(PPI_protein_list)]
    PPI_protein_list = list(set(PPI['protein1'].unique()).union(set(PPI['protein2'].unique())))
    PPI_protein = PPI_protein[PPI_protein['Gene_symbol'].isin(PPI_protein_list)]
    PPI_protein = PPI_protein.sort_values(by=['Gene_symbol'])
    PPI_protein_list = list(PPI_protein['Gene_symbol'].unique())
    protein_dict = dict(zip(PPI_protein_list, list(range(0, len(PPI_protein_list)))))
    PPI['protein1'] = PPI['protein1'].apply(lambda x: protein_dict[x])
    PPI['protein2'] = PPI['protein2'].apply(lambda x: protein_dict[x])
    PPI_protein['ID'] = PPI_protein['Gene_symbol'].apply(lambda x: protein_dict[x])
    PPI_protein = PPI_protein.sort_values(by=['ID'])

    return PPI,PPI_protein
#判断一个多元素的列表是否存在于另一个嵌套列表中
def is_sublist(sub_list, main_list):
    for item in main_list:
        if isinstance(item, list) and sub_list == item:
            return True
    return False

#嵌套列表去重
def Nested_list_dup(Nested_list):
    Nested_list = [sorted(sublist) for sublist in Nested_list]
    Nested_list_dup = []
    for sublist in Nested_list:
        if sublist not in Nested_list_dup:
            Nested_list_dup.append(sublist)
    return Nested_list_dup

#统计嵌套列表分布情况
def count_sublist_lengths(lst):
    length_counts = {}
    for sub_lst in lst:
        length = len(sub_lst)
        if length not in length_counts:
            length_counts[length] = 1
        else:
            length_counts[length] += 1
    return length_counts

#统计嵌套列表包含的唯一元素
def count_unique_elements(lst):
    unique_elements = set()
    for sub_lst in lst:
        unique_elements |= set(sub_lst)
    return unique_elements

def convert_ppi(ppi_list):
    ppi_dict = {}
    for ppi in ppi_list:
        for protein in ppi:
            if protein not in ppi_dict:
                ppi_dict[protein] = []
            other_proteins = [p for p in ppi if p != protein]
            ppi_dict[protein].extend(other_proteins)
    return ppi_dict

###随机采样--阴性数据集
def select_subunits(df, protein_list, fold):
        complex_list = []
        for index, row in df.iterrows():
            subunit_count = row['subunits_counts']
            complex_count = row['counts_subunits_counts']
            for _ in range(complex_count * fold):
                subunits = random.sample(protein_list, subunit_count)
                complex_list.append(subunits)
        return complex_list
 
###根据阳性数据集的分布取阴性数据集
def negative_on_distribution(PC, protein_list, fold):
        PC_subunit_count = {}
        for pc in PC:
            length = len(pc)
            if length in PC_subunit_count:
                PC_subunit_count[length] += 1
            else:
                PC_subunit_count[length] = 1

        PC_subunit_count = pd.DataFrame(PC_subunit_count, index=[0]).T.reset_index()
        PC_subunit_count.columns = ['subunits_counts', 'counts_subunits_counts']

        PCs_N = select_subunits(PC_subunit_count, protein_list, fold)
        PCs_N = [sorted(i) for i in PCs_N]

        return PCs_N

def list_intersection(l1, l2):
    return list(set(l1).intersection(set(l2)))

def list_difference(l1, l2):
    return list(set(l1).difference(set(l2)))

def list_union(l1, l2):
    return list(set(l1).union(set(l2)))

def load_pickle(path, file_name):
    # print(f'loading:{path}{file_name}')
    with open(f'{path}{file_name}', 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(path, file_name, data_pd):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}{file_name}', 'wb') as f:
        pickle.dump(data_pd, f)

def load_txt_list(path, file_name, display_flag=True):
    if display_flag:
        print(f'Loading {path}{file_name}')

    list = []
    with open(f'{path}{file_name}', 'r') as f:

        lines = f.readlines()

        for line in lines:
            node_list = line.strip('\n').strip(' ').split(' ')
            list.append(node_list)

    return list

def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    a_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        p0 = (preds < threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        tn = np.sum(p0) - fn
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        acc = (tp + tn) / (tp + fp + tn + fn)
        if p_max < precision:
            f_max = f
            a_max = acc
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, a_max, p_max, r_max, t_max

def calculate_f1_score(preds, labels):
    preds = preds.round()
    preds = preds.ravel()
    labels = labels.astype(np.int32)
    labels = labels.ravel()
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f = f1_score(labels, preds)
    return f, acc, precision, recall

def evaluate_performance(y_test, y_score):
    n_classes = y_test.shape[1]
    perf = dict()

    perf["M-aupr"] = 0.0
    n = 0
    aupr_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        num_pos = num_pos.astype(float)
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            aupr_list.append(ap)
            num_pos_list.append(num_pos)
    perf["M-aupr"] /= n
    perf['aupr_list'] = aupr_list
    perf['num_pos_list'] = num_pos_list

    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    perf['roc_auc'] = roc_auc_score(y_test.ravel(), y_score.ravel())
    perf['F-max'], perf['acc_max'], perf['pre_max_max'], perf['rec_max'], perf['thr_max'] = calculate_fmax(y_score, y_test)  #precision_max
    perf['F1-score'], perf['accuracy'], perf['precision'], perf['recall'] = calculate_f1_score(y_score, y_test)
    return perf

def plot_roc(Y_label, y_pred,str):
    fpr, tpr, threshold = roc_curve(Y_label, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, color='red',lw=2, label='sequence (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(str+' ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def read_ppi_file(file_path):
    with open(file_path, "r") as file:
        ppi_pairs = [line.strip().split() for line in file.readlines()]
    return ppi_pairs

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))

def SparseTensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_ppi_network(filename, gene_num):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num, gene_num))
    for x in data:
        temp = x.strip().split("\t")
        adj[int(temp[0]), int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T
    return adj

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    adj_sparse = sp.coo_matrix(adj_normalized)
    return adj_sparse

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def create_diagonal_1_array(arr):
        new_array = np.copy(arr)
        np.fill_diagonal(new_array, 1)
        return new_array

# PPI Extend
def subgraph_expansion(subgraph,PPI_dict,model_score,threshold_alpha):
    expanded_subgraph = subgraph.copy()
    adjacent_points = [PPI_dict[i] for i in expanded_subgraph]
    adjacent_points = list(set([item for sublist in adjacent_points for item in sublist]))

    while True:
        adjacent = list(set(adjacent_points).difference(set(expanded_subgraph)))

        if len(adjacent) == 0:
            break

        scores = model_score([expanded_subgraph + [v] for v in adjacent])
        max_index = np.argmax(scores)

        if scores[max_index] > threshold_alpha:
            expanded_subgraph.append(adjacent[max_index])
        else:
            break

    return expanded_subgraph

def nested_list_unique(nested_list):   #嵌套列表去重
    nested_list = [sorted(sublist) for sublist in nested_list]
    nested_list_dup = []
    for sublist in nested_list:
        if sublist not in nested_list_dup:
            nested_list_dup.append(sublist)
    return nested_list_dup

def calculate_overlap_ratio(subgraph_i, subgraph_k):

    intersection = len(set(subgraph_i).intersection(set(subgraph_k)))
    union = len(set(subgraph_i).union(set(subgraph_k)))
    #union = len(set(subgraph_k))

    overlap_ratio = intersection / union

    return overlap_ratio

def subgraph_filtration(candidate_subgraphs,scores,threshold_beta):
    # 将候选复合物和分数按照分数降序排列
    sorted_complexes = sorted(zip(candidate_subgraphs, scores), key=lambda x: x[1], reverse=True)
    # 挑选候选子图中，重叠率>threshold_beta,合并之后score> 原始得分的复合物
    filtered_PC = []

    for i in range(len(sorted_complexes)):
        candidate_subgraph_i, score_i = sorted_complexes[i]
        if len(candidate_subgraph_i) != 0:
            filtered_PC_i = []
            for k in range(i+1,len(sorted_complexes)):
                candidate_subgraph_k, score_j = sorted_complexes[k]
                overlapping_ratio = calculate_overlap_ratio(candidate_subgraph_i, candidate_subgraph_k)
                if overlapping_ratio >= threshold_beta:
                    candidate_subgraph = list(set(candidate_subgraph_i).union(set(candidate_subgraph_k)))
                    if model_score([candidate_subgraph])[0] > score_i:
                        filtered_PC_i.append(candidate_subgraph)
                    else:
                        sorted_complexes[k] = [],[]
            if len(filtered_PC_i) == 0:
                filtered_PC.append(candidate_subgraph_i)
            else:
                filtered_PC.extend(filtered_PC_i)

    return filtered_PC

# Additional imports for t-SNE visualization
import torch
import numpy as np
import os
from datetime import datetime

def generate_tsne_visualization(embedding, args, embeddings_dir="embeddings", 
                               perplexity_values=[30, 50, 100], max_samples=2000):
    """
    Generate and save t-SNE visualization of embeddings.
    
    Args:
        embedding (torch.Tensor): The embedding tensor to visualize
        args: Command line arguments containing model parameters
        embeddings_dir (str): Directory to save the visualizations
        perplexity_values (list): List of perplexity values to try
        max_samples (int): Maximum number of samples for t-SNE (to avoid performance issues)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\nGenerating t-SNE visualization of embeddings...")
    
    try:
        # Create embeddings directory if it doesn't exist
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)
        
        # Convert embeddings to numpy for t-SNE
        embeddings_numpy = embedding.detach().cpu().numpy()
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Apply t-SNE with different perplexity values
        for perplexity in perplexity_values:
            try:
                print(f"  Computing t-SNE with perplexity={perplexity}...")
                
                # Limit the number of samples for t-SNE if too large (t-SNE can be slow)
                if embeddings_numpy.shape[0] > max_samples:
                    print(f"  Sampling {max_samples} proteins for t-SNE visualization...")
                    indices = np.random.choice(embeddings_numpy.shape[0], max_samples, replace=False)
                    embeddings_sample = embeddings_numpy[indices]
                else:
                    embeddings_sample = embeddings_numpy
                    indices = np.arange(embeddings_numpy.shape[0])
                
                # Apply t-SNE
                tsne = TSNE(
                    n_components=2, 
                    perplexity=min(perplexity, len(embeddings_sample) - 1),
                    random_state=42,
                    n_iter=1000,
                    verbose=1
                )
                embeddings_2d = tsne.fit_transform(embeddings_sample)
                
                # Create the plot
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=range(len(embeddings_2d)), cmap='viridis', 
                                    alpha=0.6, s=20)
                plt.colorbar(scatter, label='Protein Index')
                plt.title(f't-SNE Visualization of Protein Embeddings\n'
                         f'Perplexity: {perplexity}, Model: {args.model}, '
                         f'Embedding Dim: {embedding.shape[1]}')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                
                # Add metadata text
                metadata_text = f'Species: {args.species}\nTotal Proteins: {embedding.shape[0]}\n'
                metadata_text += f'PPI Networks: {len(args.ppi_paths)}\nEpochs: {args.epochs}'
                plt.text(0.02, 0.98, metadata_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Save the plot
                filename = f"tsne_embeddings_{args.model}_{args.species}_{timestamp}_perp{perplexity}.png"
                filepath = os.path.join(embeddings_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  t-SNE plot saved to: {filepath}")
                
                # Note: .npz coordinate saving has been disabled
                print(f"  t-SNE coordinates (.npz format) generation skipped")
                
            except Exception as e:
                print(f"  Error generating t-SNE with perplexity {perplexity}: {str(e)}")
                continue
        
        # Save embedding statistics
        stats_filename = f"embedding_stats_{args.model}_{args.species}_{timestamp}.txt"
        stats_filepath = os.path.join(embeddings_dir, stats_filename)
        with open(stats_filepath, 'w') as f:
            f.write(f"Embedding Statistics\n")
            f.write(f"====================\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Species: {args.species}\n")
            f.write(f"Embedding Shape: {embedding.shape}\n")
            f.write(f"Embedding Mean: {embeddings_numpy.mean():.6f}\n")
            f.write(f"Embedding Std: {embeddings_numpy.std():.6f}\n")
            f.write(f"Embedding Min: {embeddings_numpy.min():.6f}\n")
            f.write(f"Embedding Max: {embeddings_numpy.max():.6f}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Hidden1: {args.hidden1}\n")
            f.write(f"Hidden2: {args.hidden2}\n")
            f.write(f"Dropout Rate: {args.droprate}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"PPI Networks: {len(args.ppi_paths)}\n")
            f.write(f"PPI Network Names: {', '.join(args.ppi_paths[:5])}{'...' if len(args.ppi_paths) > 5 else ''}\n")
        
        print(f"Embedding statistics saved to: {stats_filepath}")
        print("t-SNE visualization completed successfully!")
        
        return True
        
    except ImportError:
        print("Warning: scikit-learn not available. Skipping t-SNE visualization.")
        print("To enable t-SNE visualization, install scikit-learn: pip install scikit-learn")
        return False
    except Exception as e:
        print(f"Error generating t-SNE visualization: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False






def load_ppi_data_imad(ppi_file_path: str) -> list[tuple]:
    """
    Load protein-protein interaction network data from a TSV file.

    Reads a tab-separated file containing protein interaction pairs and converts
    them into a list of tuples. Each line in the file should contain two protein
    names separated by a tab character.

    Args:
        ppi_file_path (str): Path to the PPI network TSV file

    Returns:
        list[tuple]: List of tuples where each tuple contains two protein names
                     representing an interaction (protein1, protein2). Returns
                     empty list if file cannot be loaded.
    """
    try:
        ppi_df = pd.read_csv(
            ppi_file_path, sep="\t", header=None, names=["protein1", "protein2"]
        )
        interactions = [
            (row["protein1"], row["protein2"]) for _, row in ppi_df.iterrows()
        ]
        return interactions
    except Exception as e:
        print(f"Error loading PPI network: {str(e)}")
        return []


def get_proteins_list(ppi_network: list[tuple]):
    """
    Extract all unique proteins from a protein-protein interaction network.

    Args:
        ppi_network (list[tuple]): List of tuples where each tuple contains two protein names
                                   (protein1, protein2) representing an interaction

    Returns:
        set: Set of unique protein names found in the network.
    """
    proteins = set()
    for protein1, protein2 in ppi_network:
        proteins.add(protein1)
        proteins.add(protein2)
    return proteins


def file_exists(path: str):
    """
    Check if a file or directory exists at the specified path.

    Args:
        path (str): File or directory path

    Returns:
        bool: True if the path exists, False otherwise
    """
    return Path(path).exists()


def generate_dynamic_ppi_data(
    biclustering_results: list[tuple],
    discretized_expression: pd.DataFrame,
    static_ppi_network: list[tuple],
    output_path: str,
    dataset_name: str,
    print_results=False,
):
    """
    Save biclustering results as Dynamic PPI sub-networks and generate summary statistics.

    Processes the output of biclustering algorithms to create dynamic PPI sub-networks
    by filtering the static PPI network based on proteins selected in each bicluster.
    Generates individual TSV files for each bicluster's PPI sub-network and a summary
    file with statistics.

    Args:
        biclustering_results (list): List of tuples where each tuple contains:
                                     - bicluster_array (np.ndarray): Binary array indicating selected genes/timepoints
                                     - fitness (float): Fitness score of the bicluster
        discretized_expression (pd.DataFrame): Discretized gene expression dataframe
        static_ppi_network (list[tuple]): Static PPI network as list of protein interaction tuples
        output_path (str): Directory path where output files will be saved
        dataset_name (str): Name identifier for the dataset, used as prefix for output filenames
        print_results (bool): If True, prints summary statistics to console (default: False)

    Returns:
        None

    Side Effects:
        - Creates output directory if it doesn't exist
        - Generates individual TSV files for all dynamic PPI sub-networks
        - Creates "_results_info.tsv" with summary statistics
        - Prints confirmation message and optionally results summary

    Output Files:
        - {dataset_name}_{i}.tsv: Dynamic PPI sub-network for bicluster i (tab-separated protein pairs)
        - _results_info.tsv: Summary table with columns:
            - File Name: Name of the sub-network file
            - Fitness score: Quality score of the bicluster
            - Number of proteins: Count of unique proteins in the sub-network
            - Number of interactions: Count of interactions in the sub-network
    """
    m = discretized_expression.shape[0]

    # Create the folder if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    summary_data = {
        "File Name": [],
        "Fitness Score": [],
        "Protein Count": [],
        "Interaction Count": [],
    }

    print("Iterating on biclusters")
    dynamic_ppi_networks = []
    for i, (bicluster_array, fitness) in enumerate(biclustering_results):

        protein_set = set(discretized_expression.index[bicluster_array[:m] == 1])

        # Filter interactions in a single pass with optimized lookup
        filtered_interactions = [
            (protein1, protein2)
            for protein1, protein2 in static_ppi_network
            if protein1 in protein_set and protein2 in protein_set
        ]
        dynamic_ppi_networks.append(filtered_interactions)

        # Create output filename
        output_file_path = Path(output_path) / f"{dataset_name}_{i + 1}.tsv"
        
        # Write all interactions at once using bulk I/O
        if filtered_interactions:
            interactions_text = "\n".join(
                f"{p1}\t{p2}" for p1, p2 in filtered_interactions
            )
            with open(output_file_path, "w") as f:
                f.write(interactions_text + "\n")
        else:
            # Create empty file if no interactions
            with open(output_file_path, "w") as f:
                pass

        summary_data["File Name"].append(f"{dataset_name}_{i + 1}")
        summary_data["Fitness Score"].append(fitness)
        summary_data["Protein Count"].append(
            len(get_proteins_list(filtered_interactions))
        )
        summary_data["Interaction Count"].append(len(filtered_interactions))

    df = pd.DataFrame(data=summary_data)
    df.to_csv((Path(output_path) / "_data_summary.tsv"), sep="\t", index=False)

    print("Dynamic PPI network saved in ", output_path)

    if print_results:
        print(f"\nDynamic PPI network for {dataset_name} dataset")
        print(df)
    return dynamic_ppi_networks


def print_bicluster(
    DGE_df: pd.DataFrame,
    dataset_name: str,
    metaheuristic: str,
    biclustering_results: dict,
):

    output_path = dataset_name + "_meta_benchmark"
    m = DGE_df.shape[0]

    sorted_results = sorted(biclustering_results, key=lambda x: x[1], reverse=True)

    data = {
        "Metaheuristic": [],
        "Fitness Score": [],
        "Protein Count": [],
        "Time Points": [],
    }

    # Create the folder if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for i, (bicluster_array, fitness) in enumerate(sorted_results):
        data["Metaheuristic"].append(f"{metaheuristic}_{i + 1}")
        data["Fitness Score"].append(fitness)
        data["Protein Count"].append(sum(bicluster_array[:m]))
        data["Time Points"].append(sum(bicluster_array[m:]))

    df = pd.DataFrame(data=data)
    df.to_csv((Path(output_path) / f"{metaheuristic}.tsv"), sep="\t", index=False)
