from src.models.wrappers import PCpredict
from src.utils.helpers import Nested_list_dup, convert_ppi
import os
import pandas as pd

def load_custom_dnn_model(weights_data, embeddings):
    """
    Load a custom DNN model from weights data.
    
    Args:
        weights_data: Loaded torch checkpoint data
        embeddings: Input embeddings tensor to infer dimensions if needed
    
    Returns:
        Loaded and configured PCpredict model
    """
    # Check if it's a complete model checkpoint or just weights
    if 'model_config' in weights_data and 'model_state_dict' in weights_data:
        # Complete checkpoint with config
        model_config = weights_data['model_config']
        model = PCpredict(
            in_channels=model_config['in_channels'],
            num_class=model_config['num_class'],
            drop_rate=model_config.get('drop_rate', 0.3)
        )
        model.load_state_dict(weights_data['model_state_dict'])
        print(f"Loaded custom DNN model with config: {model_config}")
    elif 'state_dict' in weights_data:
        # Just state dict, need to infer config from embeddings
        embedding_dim = embeddings.shape[-1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        model = PCpredict(
            in_channels=embedding_dim,
            num_class=1,  # Binary classification
            drop_rate=0.3  # Default dropout rate
        )
        model.load_state_dict(weights_data['state_dict'])
        print(f"Loaded custom DNN weights, inferred embedding dim: {embedding_dim}")
    else:
        # Assume it's a direct state dict
        # Try to infer dimensions from the weights themselves
        first_layer_key = next(iter(weights_data.keys()))
        if 'weight' in first_layer_key and len(weights_data[first_layer_key].shape) > 1:
            embedding_dim = weights_data[first_layer_key].shape[1]
        else:
            embedding_dim = embeddings.shape[-1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        
        model = PCpredict(
            in_channels=embedding_dim,
            num_class=1,  # Binary classification
            drop_rate=0.3  # Default dropout rate
        )
        model.load_state_dict(weights_data)
        print(f"Loaded custom DNN weights as direct state dict, embedding dim: {embedding_dim}")
    
    model.eval()
    return model

def load_ppi_data(data_path, species, ppi_path="None", dataset_name="None"):
    """Load PPI network data from dynamic_ppi directory."""
    print(f"Loading PPI network data from {ppi_path}...")
    
    # Try to load from the specified PPI directory (krogan1, collins1, etc.)
    ppi_file_path = os.path.join(data_path, species, "dynamic_ppi", dataset_name, ppi_path)
    
    # Look for common PPI file patterns in order of preference
    possible_files = [
        "ID_Change_PPI_0.txt",  # Processed file with IDs
        "network.txt",          # Raw network file
        "ID_Change_PPI_1.txt",
        "PPI_network.txt", 
        "STRING.txt",
        "ppi.txt"
    ]
    
    ppi_data = None
    used_file = None
    for filename in possible_files:
        full_path = os.path.join(ppi_file_path, filename)
        if os.path.exists(full_path):
            print(f"Loading PPI data from: {full_path}")
            try:
                if filename == "network.txt":
                    # Raw network file format: protein1, protein2, score
                    ppi_data = pd.read_csv(full_path, sep="\t", header=None, names=['protein1', 'protein2', 'score'])
                    ppi_data = ppi_data[['protein1', 'protein2']].query('protein1 != protein2')
                    # Add reverse direction for undirected graph
                    ppi_trans = ppi_data[['protein2', 'protein1']].copy()
                    ppi_trans.columns = ['protein1', 'protein2'] 
                    ppi_data = pd.concat([ppi_data, ppi_trans], axis=0).reset_index(drop=True)
                else:
                    # ID-based file format: already processed
                    ppi_data = pd.read_csv(full_path, sep="\t", header=None)
                used_file = filename
                break
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
                continue
    
    if ppi_data is None:
        print(f"No valid PPI file found in {ppi_file_path}")
        return None, None
    
    # Convert to list format
    ppi_list = ppi_data.values.tolist()
    ppi_list = Nested_list_dup(ppi_list)
    ppi_dict = convert_ppi(ppi_list)
    
    print(f"Loaded PPI network from {used_file} with {len(ppi_dict)} proteins and {sum(len(interactions) for interactions in ppi_dict.values())} interactions")
    
    return ppi_list, ppi_dict
