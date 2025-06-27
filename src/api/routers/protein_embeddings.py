import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional
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
import json

# Import necessary modules from the existing codebase
from src.models.wrappers import ParallelHGVAE
from src.models.layers import multi_network_loss_function
from src.utils.helpers import (
    try_gpu, sequence_CT, Nested_list_dup, count_unique_elements,
    convert_ppi, generate_tsne_visualization, load_txt_list
)
from src.training.train_dnn import HGC_DNN

router = APIRouter(prefix="/generate-embeddings", tags=["generate-embeddings"])

@router.post("", summary="Generate embeddings from dynamic PPI networks")
async def generate_embeddings(
    dataset_name: str = Form("collins_GA"),
    model_params: str = Form(None, description="JSON string containing ParallelHGVAE model parameters"),
    networks: Optional[List[UploadFile]] = File(
        None, description="One or more PPI network files (CSV edge lists)"
    ),
):
    """
    Generate protein embeddings using ParallelHGVAE model.
    
    Handles two cases:
    1. When networks are provided: Uses uploaded network files
    2. When no networks provided: Uses default biclusters from dataset_name
    
    Model parameters should be provided as JSON string with the following structure:
    {
        "hidden1": 200,
        "hidden2": 100,
        "droprate": 0.5,
        "lr": 0.001,
        "epochs": 10,
        "dnn_epochs": 500,
        "dnn_learning_rate": 0.001,
        "dnn_dropout_rate": 0.3
    }
    """
    
    try:
        print(f"Starting embedding generation for dataset: {dataset_name}")
        
        # Parse model parameters from form data
        if model_params:
            try:
                parsed_params = json.loads(model_params)
                print(f"Received model parameters: {parsed_params}")
                
                # Extract ParallelHGVAE parameters with defaults
                hidden1 = parsed_params.get('hidden1', 200)
                hidden2 = parsed_params.get('hidden2', 100)
                droprate = parsed_params.get('droprate', 0.5)
                lr = parsed_params.get('lr', 0.001)
                epochs = parsed_params.get('epochs', 10)
                
                # Extract DNN parameters with defaults
                dnn_epochs = parsed_params.get('dnn_epochs', 500)
                dnn_learning_rate = parsed_params.get('dnn_learning_rate', 0.001)
                dnn_dropout_rate = parsed_params.get('dnn_dropout_rate', 0.3)
                
                # Validate ParallelHGVAE parameter types and ranges
                if not isinstance(hidden1, int) or hidden1 <= 0:
                    raise ValueError("hidden1 must be a positive integer")
                if not isinstance(hidden2, int) or hidden2 <= 0:
                    raise ValueError("hidden2 must be a positive integer")
                if not isinstance(droprate, (int, float)) or not (0 <= droprate <= 1):
                    raise ValueError("droprate must be a number between 0 and 1")
                if not isinstance(lr, (int, float)) or lr <= 0:
                    raise ValueError("lr must be a positive number")
                if not isinstance(epochs, int) or epochs <= 0:
                    raise ValueError("epochs must be a positive integer")
                
                # Validate DNN parameter types and ranges
                if not isinstance(dnn_epochs, int) or dnn_epochs <= 0:
                    raise ValueError("dnn_epochs must be a positive integer")
                if not isinstance(dnn_learning_rate, (int, float)) or dnn_learning_rate <= 0:
                    raise ValueError("dnn_learning_rate must be a positive number")
                if not isinstance(dnn_dropout_rate, (int, float)) or not (0 <= dnn_dropout_rate <= 1):
                    raise ValueError("dnn_dropout_rate must be a number between 0 and 1")
                    
                print(f"Using ParallelHGVAE parameters - hidden1: {hidden1}, hidden2: {hidden2}, droprate: {droprate}, lr: {lr}, epochs: {epochs}")
                print(f"Using DNN parameters - dnn_epochs: {dnn_epochs}, dnn_learning_rate: {dnn_learning_rate}, dnn_dropout_rate: {dnn_dropout_rate}")
                
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, 
                                  detail=f"Invalid JSON format in model_params: {str(e)}")
            except ValueError as e:
                raise HTTPException(status_code=400, 
                                  detail=f"Invalid parameter value: {str(e)}")
        else:
            # Use default parameters
            hidden1 = 200
            hidden2 = 100
            droprate = 0.5
            lr = 0.001
            epochs = 10
            dnn_epochs = 500
            dnn_learning_rate = 0.001
            dnn_dropout_rate = 0.3
            
        
        # Define paths based on the existing project structure
        base_data_path = "./data"
        species = "Saccharomyces_cerevisiae"
        feature_path = "protein_feature"
        
        # Load protein sequence features
        sequence_path = os.path.join(base_data_path, species, feature_path, 
                                   "uniprot-sequences-2023.05.10-01.31.31.11.tsv")
        
        if not os.path.exists(sequence_path):
            raise HTTPException(status_code=404, 
                              detail=f"Sequence file not found: {sequence_path}")
        
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
                                  detail=f"Error processing uploaded networks: {str(e)}")
        
        else:
            print("No networks provided, using default biclusters...")
            
            # Use biclusters from the specified dataset
            biclusters_path = os.path.join(base_data_path, species, "biclusters")
            dynamic_ppi_path = os.path.join(base_data_path, species, "dynamic_ppi", dataset_name)
            
            if not os.path.exists(biclusters_path):
                raise HTTPException(status_code=404, 
                                  detail=f"Biclusters directory not found: {biclusters_path}")
            
            # Copy biclusters to dynamic_ppi format (similar to copy_bicluster_to_collins.py)
            os.makedirs(dynamic_ppi_path, exist_ok=True)
            
            # print("fjdkslqfjqldmfjdklqsmfjdlskjf", dataset_name)
            # Find available bicluster files
            bicluster_files = [f for f in os.listdir(biclusters_path + "/" + dataset_name) 
                             if f.startswith(dataset_name) and f.endswith('.tsv')]
            print(f"Found bicluster files: {bicluster_files}")
            
            if not bicluster_files:
                raise HTTPException(status_code=404, 
                                  detail=f"No bicluster files found for dataset {dataset_name}")
            
            
            for i in range(len(bicluster_files)):
                network_name = f"{dataset_name}_{i+1}"
                network_dir = os.path.join(dynamic_ppi_path, network_name)
                os.makedirs(network_dir, exist_ok=True)
                
                # Copy bicluster file as network.txt
                src_file = os.path.join(biclusters_path, dataset_name, f"{dataset_name}_{i+1}.tsv")
                dest_file = os.path.join(network_dir, "network.txt")
                print(f"Copying {src_file} to {dest_file}...")

                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dest_file)
                    ppi_paths.append(network_name)
                    print(f"Copied {src_file} to {dest_file}")
        
        if not ppi_paths:
            raise HTTPException(status_code=400, 
                              detail="No PPI networks available for processing")
        
        print(f"Processing {len(ppi_paths)} PPI networks...")
        
        # Collect all unique proteins from all PPI networks
        all_proteins_set = set()
        raw_ppi_networks = []
        
        for i, ppi_path in enumerate(ppi_paths):
            print(f"Loading network {i+1}/{len(ppi_paths)} from {ppi_path}...")
            # for name in os.listdir(base_data_path + "/dynamic_ppi"):
                # print("fdsklfjlkqsfjmldksjflkdsqfmjskldfjls",name)
            
            ppi_file = None
            # Load PPI network
            if use_provided_networks:
                ppi_file = os.path.join(base_data_path, "dynamic_ppi", 
                                  dataset_name.lower(), ppi_path.lower(), "network.txt")
            else:
                ppi_file = os.path.join(base_data_path, species, "dynamic_ppi", 
                                  dataset_name.lower(), ppi_path.lower(), "network.txt")

            
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
                              detail="No valid PPI networks could be loaded")
        
        print(f"Total unique proteins across all networks: {len(all_proteins_set)}")
        
        # Create unified protein dictionary using the same method as extract_complexes
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
        
        # Save protein list file in the same format as expected by extract_complexes
        protein_list_dir = os.path.join("./data", species, "Gene_Entry_ID_list", dataset_name)
        os.makedirs(protein_list_dir, exist_ok=True)
        protein_list_path = os.path.join(protein_list_dir, "Protein_list.csv")
        
        # Save in the format expected by extract_complexes: Gene_symbol, Entry, ID (tab-separated, no header)
        protein_list_df = all_proteins_df[['Gene_symbol', 'Entry', 'ID']].copy()
        protein_list_df.to_csv(protein_list_path, sep='\t', header=False, index=False)
        print(f"Saved protein list to {protein_list_path}")
        
        # Now load the protein dictionary using the same method as extract_complexes
        if os.path.exists(protein_list_path):
            protein_df = pd.read_csv(protein_list_path, sep='\t', header=None, 
                                    names=['Gene_symbol', 'Entry', 'ID'])
            protein_dict = dict(zip(protein_df['Gene_symbol'], protein_df['ID']))
            print(f"Loaded protein dictionary with {len(protein_dict)} proteins using extract_complexes method")
        else:
            raise HTTPException(status_code=500, detail="Failed to create protein list file")
        
        print(f"Unified protein dictionary length: {len(protein_dict)}")
        
        # Use protein_dict instead of unified_protein_dict for consistency
        unified_protein_dict = protein_dict
        
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

            save_path = os.path.join("./data", species, "dynamic_ppi", dataset_name, dataset_name + "_" + str(i+1), f"ID_Change_PPI_{len(PPI_networks)}.txt")
            PPI.to_csv(save_path, index=False, header=False, sep='\t')
            
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
                    optimizer.step();
                    
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
            embedding_filename = f"Embeddings_{dataset_name}.pt"
            print(f"Saving embeddings to {embedding_filename}...")
            Embedding_path = os.path.join("./data", "results", "embeddings", embedding_filename)
            if not os.path.exists(os.path.dirname(Embedding_path)):
                os.makedirs(os.path.dirname(Embedding_path))
            torch.save(Embedding, Embedding_path)
            
            # Train DNN for protein complex prediction
            print("\nRunning protein complex prediction using DNN...")
            
            # Load protein complexes
            PC_path = os.path.join("./data", species, "protein_complex", "AdaPPI_golden_standard.txt")
            if not os.path.exists(PC_path):
                raise HTTPException(status_code=404, 
                                  detail=f"Protein complex file not found: {PC_path}")
            
            protein_complexes = load_txt_list("./data/" + species + "/protein_complex", '/AdaPPI_golden_standard.txt')
            
            # Use the first PPI network for protein complex prediction
            PPI_dict = PPI_dicts[0]
            
            print(f"Number of protein complexes: {len(protein_complexes)}")
            print(f"Number of proteins: {len(unified_protein_dict)}")
            
            # Prepare dataset information
            dataset_info = {
                'species': species,
                'data_path': "./data",
                'feature_path': feature_path,
                'PC_path': "protein_complex",
                'ppi_networks': ppi_paths,
                'total_ppi_networks': len(ppi_paths),
                'primary_ppi_network': ppi_paths[0],
                'num_protein_complexes': len(protein_complexes),
                'num_proteins': len(unified_protein_dict),
                'sequence_feature_file': "uniprot-sequences-2023.05.10-01.31.31.11.tsv"
            }
            
            # Prepare model parameters
            model_params = {
                'model_type': "ParallelHGVAE",
                'learning_rate': lr,
                'hidden1': hidden1,
                'hidden2': hidden2,
                'dropout_rate': droprate,
                'epochs': epochs,
                'dnn_epochs': dnn_epochs,  # Use parsed parameter instead of fixed value
                'dnn_learning_rate': dnn_learning_rate,  # Use parsed parameter instead of fixed value
                'dnn_dropout_rate': dnn_dropout_rate,  # Use parsed parameter instead of fixed value
                'cross_validation_folds': 5,
                'negative_sampling_ratio': 5
            }
            
            # Run DNN training and evaluation
            try:
                score, precision, recall, f1, acc, sn, predict_pc, predict_pc_names, performance, threshold = HGC_DNN(protein_complexes, unified_protein_dict, PPI_dict, Embedding, 
                                     save_predictions=True, 
                                     dataset_info=dataset_info, 
                                     model_params=model_params)
                
                dnn_training_successful = True
                
            except Exception as e:
                print(f"Error during DNN training: {str(e)}")
                score = None
                precision = None
                recall = None
                f1 = None
                acc = None
                predict_pc_names = []
                dnn_training_successful = False
            
            # Calculate overlap scores for predicted complexes
            overlap_scores = []
            # if predict_pc_names and dnn_training_successful:
            from src.evaluation.metrics import calculate_overlap_scores
            PCs = []
            for pc in protein_complexes:
                if len(pc) > 2:
                    if set(pc).issubset(set(list(protein_dict.keys()))):
                        pc_map = [protein_dict[sub] for sub in pc]
                        PCs.append(pc_map)

            PC = [sorted(i) for i in PCs]
                
            
            overlap_scores = calculate_overlap_scores(predict_pc, PC)
                    # print(f"Calculated overlap scores for {len(overlap_scores)} predicted complexes")
            
            # Save predicted complexes to local file similar to extract_complexes
            output_filename = f"{dataset_name}_predicted_complexes.txt"
            output_path = os.path.join("./data", "results", "predicted_complexes", output_filename)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                with open(output_path, 'w') as f:
                    for complex_genes in predict_pc_names:
                        f.write(' '.join(complex_genes) + '\n')
                print(f"Saved {len(predict_pc_names)} predicted complexes to {output_path}")
            except Exception as e:
                print(f"Warning: Could not save predicted complexes to file: {e}")
            
            # Clean up temporary directory if used
            if use_provided_networks and 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Prepare response matching extract_complexes structure
            response_data = {
                "dataset_name": dataset_name,
                "predicted_complexes": predict_pc_names,
                "threshold": threshold,
                "embeddings_file": dataset_name + ".pt", 
                "overlap_scores": overlap_scores,
                "output_file": output_filename,
                "model_source": "trained_from_scratch",
                "metrics": {
                    "Precision": precision,
                    "Recall": recall,
                    "F1_score": f1,
                    "Sensitivity": sn,
                    "Accuracy": acc,
                },
                "model_parameters": {
                    "model_type": "ParallelHGVAE",
                    "hidden1": hidden1,
                    "hidden2": hidden2,
                    "dropout_rate": droprate,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "total_parameters": total_params,
                    "dnn_epochs": dnn_epochs,  # Use parsed parameter instead of fixed value
                    "dnn_learning_rate": dnn_learning_rate,  # Use parsed parameter instead of fixed value
                    "dnn_dropout_rate": dnn_dropout_rate  # Use parsed parameter instead of fixed value
                },
                "training_info": {
                    "num_networks": len(ppi_paths),
                    "num_proteins": len(valid_proteins_list),
                    "embedding_dimension": Embedding.shape[1],
                    "num_hyperedges_per_network": num_hyperedges_list,
                    "total_vertices": len(all_vertices),
                    "valid_networks": len(valid_networks),
                    "num_protein_complexes": len(PC),
                    "training_successful": dnn_training_successful
                }
            }
            
            return response_data
        except Exception as e:
            # Clean up on error
            if use_provided_networks and 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, 
                              detail=f"Error training model: {str(e)}")
    
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
                          detail=f"Unexpected error during embedding generation: {str(e)}")
