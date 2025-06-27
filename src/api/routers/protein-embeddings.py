import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import networkx as nx
from io import BytesIO
import os
import shutil
import torch
import torch.optim as optim
from dhg import Hypergraph
import numpy as np
import pickle
import time
from datetime import datetime
import tempfile

# Import necessary modules from the existing codebase
from src.models.wrappers import ParallelHGVAE
from src.models.layers import multi_network_loss_function
from src.utils.helpers import (
    try_gpu, sequence_CT, Nested_list_dup, count_unique_elements,
    convert_ppi, generate_tsne_visualization
)

router = APIRouter(prefix="/generate-embeddings", tags=["generate-embeddings"])

@router.post("", summary="Generate embeddings from dynamic PPI networks")
async def generate_embeddings(
    networks: List[UploadFile] = File(..., description="One or more PPI network files (CSV edge lists)"),
    dataset_name: str = "collins_GA"
):
    """
    Generate protein embeddings using ParallelHGVAE model.
    
    Handles two cases:
    1. When networks are provided: Uses uploaded network files
    2. When no networks provided: Uses default biclusters from dataset_name
    """
    
    try:
        print(f"Starting embedding generation for dataset: {dataset_name}")
        
        # Define paths based on the existing project structure
        base_data_path = "./data"
        species = "Saccharomyces_cerevisiae"
        feature_path = "protein_feature"
        
        # Load protein sequence features
        sequence_path = os.path.join(base_data_path, species, feature_path, 
                                   "uniprot-sequences-2023.05.10-01.31.31.11.tsv")
        
        if not os.path.exists(sequence_path):
            raise HTTPException(status_code=404, 
                              message=f"Sequence file not found: {sequence_path}")
        
        print("Loading protein sequence features...")
        Sequence = pd.read_csv(sequence_path, sep='\t')
        Sequence_feature = sequence_CT(Sequence)
        
        # Determine if we have provided networks or need to use default
        use_provided_networks = networks and len(networks) > 0
        ppi_networks = []
        ppi_paths = []
        
        if use_provided_networks:
            print(f"Processing {len(networks)} provided networks...")
            
            # Create temporary directory for uploaded networks
            temp_dir = tempfile.mkdtemp()
            dynamic_ppi_dir = os.path.join(temp_dir, "dynamic_ppi", dataset_name)
            
            try:
                # Process uploaded network files
                for i, network_file in enumerate(networks):
                    network_name = f"{dataset_name}_{i+1}"
                    network_dir = os.path.join(dynamic_ppi_dir, network_name)
                    os.makedirs(network_dir, exist_ok=True)
                    
                    # Read and save network file
                    content = await network_file.read()
                    network_path = os.path.join(network_dir, "network.txt")
                    
                    with open(network_path, 'wb') as f:
                        f.write(content)
                    
                    ppi_paths.append(network_name)
                
                # Update paths to use temporary directory
                base_data_path = temp_dir
                
            except Exception as e:
                # Cleanup temp directory on error
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=400, 
                                  message=f"Error processing uploaded networks: {str(e)}")
        
        else:
            print("No networks provided, using default biclusters...")
            
            # Use biclusters from the specified dataset
            biclusters_path = os.path.join(base_data_path, species, "biclusters")
            dynamic_ppi_path = os.path.join(base_data_path, species, "dynamic_ppi", dataset_name)
            
            if not os.path.exists(biclusters_path):
                raise HTTPException(status_code=404, 
                                  message=f"Biclusters directory not found: {biclusters_path}")
            
            # Copy biclusters to dynamic_ppi format (similar to copy_bicluster_to_collins.py)
            os.makedirs(dynamic_ppi_path, exist_ok=True)
            
            # Find available bicluster files
            bicluster_files = [f for f in os.listdir(biclusters_path) 
                             if f.startswith(dataset_name) and f.endswith('.tsv')]
            
            if not bicluster_files:
                raise HTTPException(status_code=404, 
                                  message=f"No bicluster files found for dataset {dataset_name}")
            
            # Copy first 10 biclusters or all available (whichever is smaller)
            max_networks = min(10, len(bicluster_files))
            
            for i in range(max_networks):
                network_name = f"{dataset_name}_{i+1}"
                network_dir = os.path.join(dynamic_ppi_path, network_name)
                os.makedirs(network_dir, exist_ok=True)
                
                # Copy bicluster file as network.txt
                src_file = os.path.join(biclusters_path, f"{dataset_name}_{i+1}.tsv")
                dest_file = os.path.join(network_dir, "network.txt")
                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dest_file)
                    ppi_paths.append(network_name)
                    print(f"Copied {src_file} to {dest_file}")
        
        if not ppi_paths:
            raise HTTPException(status_code=400, 
                              message="No PPI networks available for processing")
        
        print(f"Processing {len(ppi_paths)} PPI networks...")
        
        # Collect all unique proteins from all PPI networks
        all_proteins_set = set()
        raw_ppi_networks = []
        
        for i, ppi_path in enumerate(ppi_paths):
            print(f"Loading network {i+1}/{len(ppi_paths)} from {ppi_path}...")
            
            # Load PPI network
            ppi_file = os.path.join(base_data_path, species, "dynamic_ppi", 
                                  dataset_name, ppi_path, "network.txt")
            
            try:
                PPI = pd.read_csv(ppi_file, sep="\t", header=None, names=['protein1', 'protein2', 'score'])
                PPI = PPI[['protein1', 'protein2']].query('protein1 != protein2')
                
                # Create undirected graph
                PPI_trans = PPI[['protein2', 'protein1']].copy()
                PPI_trans.columns = ['protein1', 'protein2']
                PPI = pd.concat([PPI, PPI_trans], axis=0).reset_index(drop=True)
                
                # Collect unique proteins
                network_proteins = set(PPI['protein1'].unique()).union(set(PPI['protein2'].unique()))
                all_proteins_set.update(network_proteins)
                raw_ppi_networks.append(PPI)
                
            except Exception as e:
                print(f"Error loading network {ppi_path}: {str(e)}")
                continue
        
        if not raw_ppi_networks:
            raise HTTPException(status_code=400, 
                              message="No valid PPI networks could be loaded")
        
        print(f"Total unique proteins across all networks: {len(all_proteins_set)}")
        
        # Create unified protein dictionary
        print("Creating unified protein dictionary...")
        all_proteins_df = pd.DataFrame(list(all_proteins_set), columns=['protein'])
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
        
        print(f"Unified protein dictionary length: {len(unified_protein_dict)}")
        
        # Process networks with unified protein mapping
        PPI_networks = []
        PPI_dicts = []
        hypergraphs = []
        edge_list_data_list = []
        
        for i, PPI in enumerate(raw_ppi_networks):
            print(f"Processing network {i+1}/{len(raw_ppi_networks)}...")
            
            # Map proteins using unified dictionary
            PPI = PPI[PPI['protein1'].isin(valid_proteins_list)]
            PPI = PPI[PPI['protein2'].isin(valid_proteins_list)]
            PPI['protein1'] = PPI['protein1'].map(unified_protein_dict)
            PPI['protein2'] = PPI['protein2'].map(unified_protein_dict)
            PPI = PPI.dropna()
            
            PPI_list = PPI.values.tolist()
            PPI_list = Nested_list_dup(PPI_list)
            
            # Construct PPI hypergraph
            G = nx.Graph()
            G.add_edges_from(PPI_list)
            print(f"  Finding cliques for network {i+1}...")
            PPI_hyperedge_dup = list(nx.find_cliques(G))
            
            unique_elements = count_unique_elements(PPI_hyperedge_dup)
            
            edge_list_data = {
                "num_vertices": len(unique_elements),
                "PPI_edge_list": PPI_list,
                "PPI_cliques_list": PPI_hyperedge_dup
            }
            
            PPI_networks.append(PPI_list)
            PPI_dicts.append(convert_ppi(PPI_list))
            edge_list_data_list.append(edge_list_data)
            
            print(f"Network {i+1} processed successfully")
        
        # Create unified vertex set for all networks
        print("Creating unified vertex set for all networks...")
        all_vertices = set()
        for edge_list_data in edge_list_data_list:
            all_vertices.update(count_unique_elements(edge_list_data["PPI_cliques_list"]))
        
        print(f"Total unique vertices: {len(all_vertices)}")
        vertex_mapping = {vertex: idx for idx, vertex in enumerate(sorted(all_vertices))}
        
        # Create hypergraphs with unified vertex set
        unified_hypergraphs = []
        unified_norm_list = []
        unified_pos_weight_list = []
        
        for i, edge_list_data in enumerate(edge_list_data_list):
            print(f"Creating unified hypergraph {i+1}...")
            
            # Remap cliques to use unified vertex IDs
            unified_cliques = []
            for clique in edge_list_data["PPI_cliques_list"]:
                unified_clique = [vertex for vertex in clique if vertex in vertex_mapping]
                if len(unified_clique) > 1:
                    unified_cliques.append(unified_clique)
            
            try:
                hg = Hypergraph(len(all_vertices), unified_cliques)
                hg = hg.to(device=try_gpu())
                
                # Calculate norm and pos_weight
                H_incidence = hg.H.to_dense()
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
                print(f"Error creating hypergraph {i+1}: {str(e)}")
                # Create dummy hypergraph
                hg = Hypergraph(len(all_vertices), [[0]])
                hg = hg.to(device=try_gpu())
                unified_hypergraphs.append(hg)
                unified_norm_list.append(0.5)
                unified_pos_weight_list.append(1.0)
        
        hypergraphs = unified_hypergraphs
        norm_list = unified_norm_list
        pos_weight_list = unified_pos_weight_list
        
        # Prepare feature embedding
        Sequence_feature = pd.merge(Protein_dict, Sequence_feature, how='inner')
        Sequence_feature = Sequence_feature.sort_values(by=['ID'])
        Sequence_feature = pd.DataFrame(Sequence_feature['features_seq'].to_list())
        
        # Create feature matrix
        X_full = np.zeros((len(all_vertices), Sequence_feature.shape[1]))
        
        for idx, row in Protein_dict.iterrows():
            if row['ID'] in vertex_mapping:
                unified_idx = vertex_mapping[row['ID']]
                if idx < len(Sequence_feature):
                    X_full[unified_idx] = Sequence_feature.iloc[idx].values
        
        X = torch.FloatTensor(X_full)
        X = X.to(device=try_gpu())
        
        # Train ParallelHGVAE model
        print("\nTraining ParallelHGVAE model...")
        
        # Model parameters (based on main.py defaults)
        hidden1 = 200
        hidden2 = 100
        droprate = 0.5
        lr = 0.001
        epochs = 10
        
        num_hyperedges_list = [len(hg.state_dict['raw_groups']['main']) for hg in hypergraphs]
        print(f"Number of hyperedges in each network: {num_hyperedges_list}")
        
        # Check for valid hypergraphs
        valid_networks = [i for i, num_edges in enumerate(num_hyperedges_list) if num_edges > 1]
        if not valid_networks:
            raise HTTPException(status_code=400, 
                              message="No valid hypergraphs available for training!")
        
        try:
            net = ParallelHGVAE(
                X.shape[1], 
                hidden1, 
                hidden2, 
                num_hyperedges_list,
                use_bn=True, 
                drop_rate=droprate,
                num_networks=len(PPI_networks)
            )
            
            net = net.to(device=try_gpu())
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
            
            total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params}")
            
            print("Starting training loop...")
            for epoch in range(epochs):
                try:
                    net.train()
                    st = time.time()
                    optimizer.zero_grad()
                    
                    shared_bottleneck, Z_list, H_list, mu, logvar, edge_w_list = net(X, hypergraphs)
                    
                    if len(Z_list) == 0 or len(H_list) == 0:
                        print(f"Epoch {epoch}: No valid outputs, skipping")
                        continue
                    
                    # Prepare labels
                    labels_list = []
                    for i, hg in enumerate(hypergraphs):
                        try:
                            label = hg.H.to_dense()
                            labels_list.append(label)
                        except Exception as e:
                            size = H_list[i].shape if i < len(H_list) else (X.shape[0], X.shape[0])
                            labels_list.append(torch.zeros(size, device=X.device))
                    
                    # Filter valid predictions
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
                    
                    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
                    
                except Exception as e:
                    print(f"Error in epoch {epoch}: {str(e)}")
                    continue
            
            # Get final embeddings
            print("Generating final embeddings...")
            net.eval()
            with torch.no_grad():
                shared_bottleneck, Z_list, H_list, mu, logvar, edge_w_list = net(X, hypergraphs)
                Embedding = shared_bottleneck
            
            print(f"Embeddings shape: {Embedding.shape}")
            
            # Generate t-SNE visualization
            class Args:
                def __init__(self):
                    self.model = "ParallelHGVAE"
                    self.species = species
                    self.ppi_paths = ppi_paths
                    self.epochs = epochs
                    self.dataset_name = dataset_name
                    self.lr = lr
                    self.hidden1 = hidden1
                    self.hidden2 = hidden2
                    self.droprate = droprate
            
            args = Args()
            generate_tsne_visualization(Embedding, args)
            
            # Convert embeddings to JSON-serializable format
            embeddings_numpy = Embedding.detach().cpu().numpy()
            embeddings_list = embeddings_numpy.tolist()
            
            # Clean up temporary directory if used
            if use_provided_networks and 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Prepare response
            response = {
                "message": "Embeddings generation completed successfully",
                "dataset_name": dataset_name,
                "model": "ParallelHGVAE",
                "num_networks": len(ppi_paths),
                "num_proteins": len(valid_proteins_list),
                "embedding_dimension": Embedding.shape[1],
                "training_epochs": epochs,
                "protein_names": valid_proteins_list,
                "embeddings": embeddings_list,
                "model_parameters": {
                    "hidden1": hidden1,
                    "hidden2": hidden2,
                    "dropout_rate": droprate,
                    "learning_rate": lr,
                    "total_parameters": total_params
                },
                "training_info": {
                    "num_hyperedges_per_network": num_hyperedges_list,
                    "total_vertices": len(all_vertices),
                    "valid_networks": len(valid_networks)
                }
            }
            
            return response
            
        except Exception as e:
            # Clean up on error
            if use_provided_networks and 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, 
                              message=f"Error training model: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on any error
        if use_provided_networks and 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        import traceback
        error_trace = traceback.format_exc()
        print(f"Unexpected error: {error_trace}")
        
        raise HTTPException(status_code=500, 
                          message=f"Unexpected error during embedding generation: {str(e)}")
