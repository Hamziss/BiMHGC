import torch.optim as optim
from dhg import Hypergraph
from models import ParallelHGVAE
from Train_PC import HGC_DNN
import argparse
from torch.optim import Adam
import time
from utils import *
import scipy.sparse as sp
from layers import loss_function, multi_network_loss_function
from utils import load_txt_list, count_unique_elements
import networkx as nx
import pandas as pd
import numpy as np
import torch
from pandas.core.frame import DataFrame
import traceback
import os
import pickle

def main(args):
    try:
        print("Starting main function execution...")
        # Loading Protein sequence feature 
        Sequence_path = os.path.join(args.data_path, args.species, args.feature_path,
                                    "uniprot-sequences-2023.05.10-01.31.31.11.tsv")
        Sequence = pd.read_csv(Sequence_path, sep='\t')
        Sequence_feature = sequence_CT(Sequence)
        
        # Create lists to store multiple PPI networks data
        PPI_networks = []
        PPI_dicts = []
        hypergraphs = []
        edge_list_data_list = []
        norm_list = []
        pos_weight_list = []
          # First, collect all unique proteins from all PPI networks
        print(f"Collecting proteins from {len(args.ppi_paths)} PPI networks...")
        all_proteins_set = set()
        raw_ppi_networks = []
        
        for i, ppi_path in enumerate(args.ppi_paths):
            print(f"Collecting proteins from network {i+1}/{len(args.ppi_paths)} from {ppi_path}...")
            # Load each PPI network
            PPI_file = os.path.join(args.data_path, args.species, "dynamic_ppi", args.dataset_name, ppi_path, "network.txt")
            PPI = pd.read_csv(PPI_file, sep="\t", header=None, names=['protein1', 'protein2', 'score'])
            PPI = PPI[['protein1', 'protein2']].query('protein1 != protein2')
            
            # Create the reverse direction for undirected graph
            PPI_trans = PPI[['protein2', 'protein1']].copy()
            PPI_trans.columns = ['protein1', 'protein2']
            PPI = pd.concat([PPI, PPI_trans], axis=0).reset_index(drop=True)
            
            # Collect all unique proteins from this network
            network_proteins = set(PPI['protein1'].unique()).union(set(PPI['protein2'].unique()))
            all_proteins_set.update(network_proteins)
            raw_ppi_networks.append(PPI)
        
        print(f"Total unique proteins across all networks: {len(all_proteins_set)}")
        
        # Create a unified protein dictionary using all proteins
        print("Creating unified protein dictionary...")
        all_proteins_df = DataFrame(list(all_proteins_set), columns=['protein'])
        all_proteins_df['Entry'] = all_proteins_df['protein'].apply(
            lambda x: Sequence_feature[Sequence_feature['Gene Names'].str.contains(x, case=False, na=False)]['Entry'].values[
                0] if Sequence_feature['Gene Names'].str.contains(x, case=False).any() else 'NA')
        all_proteins_df.columns = ['Gene_symbol', 'Entry']
        all_proteins_df = all_proteins_df[all_proteins_df['Entry'] != 'NA']
        all_proteins_df = all_proteins_df[all_proteins_df['Entry'].isin(set(Sequence_feature['Entry']))]
        
        # Filter to only include proteins that have sequence features
        valid_proteins_list = list(set(all_proteins_df['Gene_symbol'].unique()))
        all_proteins_df = all_proteins_df.sort_values(by=['Gene_symbol'])
        valid_proteins_list = list(all_proteins_df['Gene_symbol'].unique())
        
        # Create unified protein ID mapping
        unified_protein_dict = dict(zip(valid_proteins_list, list(range(0, len(valid_proteins_list)))))
        all_proteins_df['ID'] = all_proteins_df['Gene_symbol'].apply(lambda x: unified_protein_dict[x])
        all_proteins_df = all_proteins_df.sort_values(by=['ID'])
        Protein_dict = all_proteins_df
        
        print("unified_protein_dict length:", len(unified_protein_dict), unified_protein_dict)
        # Save the unified protein dictionary
        Protein_dict.to_csv(os.path.join(args.data_path, args.species,
                            "Gene_Entry_ID_list/Protein_list.csv"),
                index=False, header=False, sep="\t")
        
        # Process multiple PPI networks using the unified protein dictionary
        print(f"Processing {len(args.ppi_paths)} PPI networks with unified protein mapping...")
        for i, ppi_path in enumerate(args.ppi_paths):
            print(f"Processing PPI network {i+1}/{len(args.ppi_paths)} from {ppi_path}...")
            
            PPI = raw_ppi_networks[i]
            
            # Map proteins using the unified dictionary
            PPI = PPI[PPI['protein1'].isin(valid_proteins_list)]
            PPI = PPI[PPI['protein2'].isin(valid_proteins_list)]
            PPI['protein1'] = PPI['protein1'].map(unified_protein_dict)
            PPI['protein2'] = PPI['protein2'].map(unified_protein_dict)
            # Remove any NaN values that might have been introduced
            PPI = PPI.dropna()

            save_path = os.path.join(args.data_path, args.species, "dynamic_ppi", args.dataset_name, ppi_path, f"ID_Change_PPI_{len(PPI_networks)}.txt")
            PPI.to_csv(save_path, index=False, header=False, sep="\t")
            
            PPI_list = PPI.values.tolist()
            PPI_list = Nested_list_dup(PPI_list)
            
            # Constructing PPI hypergraph
            G = nx.Graph()
            G.add_edges_from(PPI_list)
            print(f"  Finding cliques for network {i+1}...")
            PPI_hyperedge_dup = list(nx.find_cliques(G))
            
            unique_elements = count_unique_elements(PPI_hyperedge_dup)
            
            edge_list_data = {}
            edge_list_data["num_vertices"] = len(unique_elements)
            edge_list_data["PPI_edge_list"] = PPI_list
            edge_list_data["PPI_cliques_list"] = PPI_hyperedge_dup

            pkl_path = os.path.join(args.data_path, args.species, "dynamic_ppi", args.dataset_name, ppi_path, f"PPI_cliques_Hyperedge_{len(PPI_networks)}.pkl")
            f_save = open(pkl_path, 'wb')
            pickle.dump(edge_list_data, f_save)
            f_save.close()
            
            # Add data to our lists
            PPI_networks.append(PPI_list)
            PPI_dicts.append(convert_ppi(PPI_list))
            edge_list_data_list.append(edge_list_data)
            
            print(f"Network {i+1}/{len(args.ppi_paths)} processed successfully")
        
        # Create unified vertex set for all networks
        print("Creating unified vertex set for all networks...")
        # Get all unique vertices across all networks
        all_vertices = set()
        for edge_list_data in edge_list_data_list:
            all_vertices.update(count_unique_elements(edge_list_data["PPI_cliques_list"]))
        
        print(f"Total unique vertices across all networks: {len(all_vertices)}")
        vertex_mapping = {vertex: idx for idx, vertex in enumerate(sorted(all_vertices))}
        
        # Recreate hypergraphs with the unified vertex set
        unified_hypergraphs = []
        unified_norm_list = []
        unified_pos_weight_list = []
        
        for i, edge_list_data in enumerate(edge_list_data_list):
            print(f"Recreating hypergraph {i+1} with unified vertex set...")
            # Remap cliques to use unified vertex IDs
            unified_cliques = []
            for clique in edge_list_data["PPI_cliques_list"]:
                # Only include vertices that exist in the unified mapping
                unified_clique = [vertex for vertex in clique if vertex in vertex_mapping]
                if len(unified_clique) > 1:  # Only include non-empty cliques with at least 2 vertices
                    unified_cliques.append(unified_clique)
            
            try:
                hg = Hypergraph(len(all_vertices), unified_cliques)
                hg = hg.to(device=try_gpu())
                
                # Calculate norm and pos_weight for this network
                H_incidence = hg.H.to_dense()
                
                # Make sure there's at least one non-zero element to avoid division by zero
                sum_val = H_incidence.sum()
                if sum_val == 0:
                    pos_weight = 1.0
                    norm = 0.5
                else:
                    pos_weight = float(H_incidence.shape[0] * H_incidence.shape[0] - sum_val) / sum_val
                    norm = H_incidence.shape[0] * H_incidence.shape[0] / float(
                        (H_incidence.shape[0] * H_incidence.shape[0] - sum_val) * 2)
                
                unified_hypergraphs.append(hg)
                unified_norm_list.append(norm)
                unified_pos_weight_list.append(pos_weight)
                
            except Exception as e:
                print(f"  Error creating unified hypergraph for network {i+1}: {str(e)}")
                # Create a dummy hypergraph with one empty hyperedge
                hg = Hypergraph(len(all_vertices), [[0]])
                hg = hg.to(device=try_gpu())
                unified_hypergraphs.append(hg)
                unified_norm_list.append(0.5)  # Default values
                unified_pos_weight_list.append(1.0)
        
        # Update hypergraphs and lists
        hypergraphs = unified_hypergraphs
        norm_list = unified_norm_list
        pos_weight_list = unified_pos_weight_list
        
        # Get the number of hyperedges for each network
        num_hyperedges_list = [len(hg.state_dict['raw_groups']['main']) for hg in hypergraphs]
        print(f"Number of hyperedges in each network: {num_hyperedges_list}")
        
        # Prepare feature embedding
        Sequence_feature = pd.merge(Protein_dict, Sequence_feature, how='inner')
        Sequence_feature = Sequence_feature.sort_values(by=['ID'])
        Sequence_feature = DataFrame(Sequence_feature['features_seq'].to_list())
        
        # Ensure the feature matrix has the same number of rows as the unified vertex set
        X_full = np.zeros((len(all_vertices), Sequence_feature.shape[1]))
        
        # Map the features we have to the correct positions in the full feature matrix
        for idx, row in Protein_dict.iterrows():
            if row['ID'] in vertex_mapping:
                unified_idx = vertex_mapping[row['ID']]
                if idx < len(Sequence_feature):
                    X_full[unified_idx] = Sequence_feature.iloc[idx].values
        
        X = torch.FloatTensor(X_full)
        X = X.to(device=try_gpu())
        
        CT_Embedding_path = os.path.join(args.data_path, args.species, args.feature_path, "protein_feature_CT.pt")
        torch.save(X, CT_Embedding_path)
        
        # Feature embedding - Parallel HGVAE
        if args.model == 'ParallelHGVAE':
            print("\nTraining Parallel HGVAE model...")
            
            # Check if we actually have valid hypergraphs to train on
            valid_networks = [i for i, num_edges in enumerate(num_hyperedges_list) if num_edges > 1]
            if not valid_networks:
                print("Error: No valid hypergraphs to train on!")
                return
            
            try:
                net = ParallelHGVAE(
                    X.shape[1], 
                    args.hidden1, 
                    args.hidden2, 
                    num_hyperedges_list,
                    use_bn=True, 
                    drop_rate=args.droprate,
                    num_networks=len(PPI_networks)
                )
                
                net = net.to(device=try_gpu())
                optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
                # optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)  # 10x smaller
                list_loss = []
                list_epoch = []

                total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print(f"Total number of parameters: {total_params}")

                print("\nStarting training loop...")
                for epoch in range(args.epochs):
                    try:
                        list_epoch.append(epoch)
                        net.train()
                        st = time.time()                       
                        optimizer.zero_grad()                    
                        shared_bottleneck, Z_list, H_list, mu, logvar, edge_w_list = net(X, hypergraphs)

                        # Check that we have valid outputs before calculating loss
                        if len(Z_list) == 0 or len(H_list) == 0:
                            print(f"Epoch {epoch}: No valid outputs from model, skipping")
                            continue

                        # Prepare labels list for the loss function
                        labels_list = []
                        for i, hg in enumerate(hypergraphs):
                            try:
                                label = hg.H.to_dense()                            
                                labels_list.append(label)
                            except Exception as e:
                                # Create a dummy label matrix
                                size = H_list[i].shape if i < len(H_list) else (X.shape[0], X.shape[0])
                                labels_list.append(torch.zeros(size, device=X.device))
                        
                        # Filter out invalid predictions or labels
                        valid_indices = []
                        for i in range(min(len(H_list), len(labels_list))):
                            if H_list[i].shape == labels_list[i].shape and H_list[i].numel() > 0:
                                valid_indices.append(i)
                                
                        if not valid_indices:
                            print(f"Epoch {epoch}: No valid prediction-label pairs, skipping")
                            continue
                            
                        valid_H_list = [H_list[i] for i in valid_indices]
                        valid_labels_list = [labels_list[i] for i in valid_indices]
                        valid_norm_list = [norm_list[i] for i in valid_indices]
                        valid_pos_weight_list = [pos_weight_list[i] for i in valid_indices]
                        
                        try:
                            loss = multi_network_loss_function(
                                preds_list=valid_H_list, 
                                labels_list=valid_labels_list,
                                mu=mu, 
                                logvar=logvar, 
                                n_nodes=len(all_vertices),
                                norm_list=valid_norm_list, 
                                pos_weight_list=valid_pos_weight_list
                            )

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                            optimizer.step()
                            list_loss.append(loss.item())
                            print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
                            
                        except Exception as e:
                            print(f"Error calculating loss in epoch {epoch}: {str(e)}")
                            continue
                        
                        # Save model every 50 epochs for safety
                        if epoch % 50 == 0 and epoch > 0:
                            checkpoint_path = os.path.join(args.data_path, args.species, args.feature_path, f"ParallelHGVAE_checkpoint_epoch_{epoch}.pt")
                            torch.save(net, checkpoint_path)
                            
                    except Exception as e:
                        print(f"Error during epoch {epoch}: {str(e)}")
                        # Continue to next epoch rather than terminating
                        continue
                
                print("\nTraining completed. Running final evaluation...")
                net.eval()
                shared_bottleneck, Z_list, H_list, mu, logvar, edge_w_list = net(X, hypergraphs)
                Embedding = shared_bottleneck  # Use the shared bottleneck as the final embedding
                model_path = os.path.join(args.data_path, args.species, args.feature_path, "ParallelHGVAE_model.pt")
                torch.save(net, model_path)
                    
            except Exception as e:
                print(f"Error in model training: {str(e)}")
                # Fall back to using raw features
                print("Using raw features due to model training failure")
                Embedding = X
        elif args.model == 'None':
            print("Using raw features without model")
            Embedding = X
        embedding_filename = f"protein_feature_Embeddings_{args.ppi_paths[0]}.pt"
        Embedding_path = os.path.join(args.data_path, args.species, args.feature_path, embedding_filename)
        torch.save(Embedding, Embedding_path)
        print("Embeddings :", Embedding.shape)
        print("Embeddings saved successfully")
          # Generate and save t-SNE visualization of embeddings
        generate_tsne_visualization(Embedding, args)
        
        # Use the first PPI network for protein complex prediction
        print("\nRunning protein complex prediction using the first PPI network...")
        PPI_dict = PPI_dicts[0]
        PC_path = os.path.join(args.data_path, args.species, args.PC_path, "AdaPPI_golden_standard.txt")
        PC = load_txt_list(os.path.join(args.data_path, args.species, args.PC_path), '/AdaPPI_golden_standard.txt')
        
        protein_dict = unified_protein_dict
        print("Starting HGC_DNN training and evaluation...")
        # print("Protein complex data path:", PC_path)
        # print("Protein complex data:", PC)
        print("number of PCs:", len(PC))
        print("number of proteins:", len(protein_dict))
        
        # Prepare dataset information
        dataset_info = {
            'species': args.species,
            'data_path': args.data_path,
            'feature_path': args.feature_path,
            'PC_path': args.PC_path,
            'ppi_networks': args.ppi_paths,
            'total_ppi_networks': len(args.ppi_paths),
            'primary_ppi_network': args.ppi_paths[0],
            'num_protein_complexes': len(PC),
            'num_proteins': len(protein_dict),
            'sequence_feature_file': "uniprot-sequences-2023.05.10-01.31.31.11.tsv"
        }
        
        # Prepare model parameters
        model_params = {
            'model_type': args.model,
            'learning_rate': args.lr,
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'dropout_rate': args.droprate,
            'epochs': args.epochs,
            'dnn_epochs': 1,  # Fixed value used in HGC_DNN
            'dnn_learning_rate': 0.001,  # Fixed value used in HGC_DNN
            'dnn_dropout_rate': 0.3,  # Fixed value used in HGC_DNN
            'cross_validation_folds': 5,
            'negative_sampling_ratio': 5
        }
        unified_PPI_dict = {}
        for pdt in PPI_dicts:
            for prot, nbrs in pdt.items():
                unified_PPI_dict.setdefault(prot, []).extend(nbrs)
        # remove duplicates in neighbor lists
        for prot, nbrs in unified_PPI_dict.items():
            unified_PPI_dict[prot] = list(set(nbrs))  

        print("PPI_dicts[0] length:", len(PPI_dicts[0]))

        model_score = HGC_DNN(PC, protein_dict, PPI_dicts[0], Embedding, 
                             save_predictions=True, 
                             dataset_info=dataset_info, 
                             model_params=model_params)
        print("Model evaluation score:", model_score)
        print("Main function completed successfully")
        
    except Exception as e:
        print("Error in main function:")
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument('--species', type=str, default="Saccharomyces_cerevisiae", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../data", help="path storing data.")
    parser.add_argument('--feature_path', type=str, default="protein_feature", help="feature path data")
    parser.add_argument('--PPI_path', type=str, default="PPI", help="Default PPI data path")
    parser.add_argument('--ppi_paths', type=str, nargs='+', default=[
        "collins_csa2_1", "collins_csa2_2", "collins_csa2_3", "collins_csa2_4", "collins_csa2_5",
        "collins_csa2_6", "collins_csa2_7", "collins_csa2_8", "collins_csa2_9", "collins_csa2_10",
        "collins_csa2_11", "collins_csa2_12", "collins_csa2_13", "collins_csa2_14", "collins_csa2_15",
        "collins_csa2_16", "collins_csa2_17", "collins_csa2_18", "collins_csa2_19", "collins_csa2_20",
        "collins_csa2_21", "collins_csa2_22", "collins_csa2_23", "collins_csa2_24", "collins_csa2_25",
        "collins_csa2_26", "collins_csa2_27", "collins_csa2_28", "collins_csa2_29", "collins_csa2_30"
    ], help="Multiple PPI network paths")
    parser.add_argument('--PC_path', type=str, default="Protein_complex", help="Protein complex data path")
    parser.add_argument('--model', type=str, default="ParallelHGVAE", help="Feature coding")

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--hidden1', type=int, default=100, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=100, help="Number of units in hidden layer 2.")
    parser.add_argument('--droprate', type=float, default=0.5, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs to HGVAE.")
    parser.add_argument('--dataset_name', type=str, default="collins_csa2", help="Dataset name.")

    args = parser.parse_args()
    print("Arguments:", args)
    main(args)
