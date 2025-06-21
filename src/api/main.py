"""
FastAPI app for protein complex extraction using HGC-DNN model.

The main /extract-complexes endpoint now supports:
1. Required: embeddings file (.pt format)
2. Optional: custom DNN weights file (.pt format)

If no DNN weights are provided, the default pre-trained model will be used.
The custom weights file can be in one of these formats:
- Complete checkpoint with 'model_config' and 'model_state_dict' keys
- Checkpoint with 'state_dict' key (dimensions inferred from embeddings)
- Direct state dictionary (dimensions inferred from weights or embeddings)
"""

# main.py
from fastapi import Body, FastAPI, UploadFile, File, HTTPException, Form
from typing import Optional
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
import torch
import io
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import local modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from Train_PC import HGC_DNN, load_pretrained_dnn, predict_with_pretrained_dnn  # Import the HGC_DNN function and pretrained model utilities
from utils import calculate_fmax, generate_dynamic_ppi_data, try_gpu, load_txt_list, Nested_list_dup, convert_ppi  # Import utility functions
from models import PCpredict  # Import the model class for creating models with custom weights
from evaluation import get_score  # Import evaluation function
from fastapi.middleware.cors import CORSMiddleware
from utils import negative_on_distribution, print_bicluster, load_ppi_data_imad
import biclustering  


# Import load_ppi_data from local benchmarks module
import importlib.util
benchmarks_path = os.path.join(parent_dir, 'benchmarks.py')
spec = importlib.util.spec_from_file_location("local_benchmarks", benchmarks_path)
local_benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_benchmarks)
load_ppi_data = local_benchmarks.load_ppi_data

app = FastAPI()

origins = ["*"]  # Allow all origins for CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/extract-complexes")
async def extract_complexes(
    embeddings_weights: UploadFile = File(..., description="Embeddings file (.pt format)"),
    dnn_weights: Optional[UploadFile] = File(None, description="Optional DNN weights file (.pt format)")
):
    """
    Accepts a .pt file with embeddings, optionally accepts DNN weights, 
    runs the DNN model on the embeddings, and returns the results.
    """
    if not embeddings_weights.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Please upload a .pt file for embeddings")
    
    if dnn_weights is not None and not dnn_weights.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="DNN weights file must be a .pt file")

    # Read embeddings file bytes into a BytesIO buffer
    data = await embeddings_weights.read()
    buffer = io.BytesIO(data)

    try:
        embeddings = torch.load(buffer, map_location="cpu")
        print(f"Loaded embeddings from {embeddings_weights.filename} with shape {embeddings.shape}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load embeddings file: {e}")
    
    # Ensure the embeddings are in the correct format
    if not isinstance(embeddings, torch.Tensor):
        raise HTTPException(status_code=400, detail="The embeddings file does not contain valid embeddings")
    
    data_path = "d:/code/ia/codesandbox/HGC-final-clean/data"
    species = "Saccharomyces_cerevisiae"
    
    # Load protein complexes from the golden standard file
    protein_complexes = load_txt_list(os.path.join(data_path, species, "protein_complex"), '/AdaPPI_golden_standard.txt')
    
    # Load protein dictionary from the protein list file
    protein_list_path = os.path.join(data_path, species, "Gene_Entry_ID_list", "Protein_list.csv")
    if os.path.exists(protein_list_path):
        protein_df = pd.read_csv(protein_list_path, sep='\t', header=None, 
                                names=['Gene_symbol', 'Entry', 'ID'])
        protein_dict = dict(zip(protein_df['Gene_symbol'], protein_df['ID']))       
    else:
        raise HTTPException(status_code=500, detail="Protein list file not found")
    
    # Load PPI data
    ppi_name = embeddings_weights.filename.rsplit('.', 1)[0]
    ppi_list, ppi_dict = load_ppi_data(data_path, species, ppi_name + "_1", ppi_name)
    if ppi_list is None or ppi_dict is None:
        raise HTTPException(status_code=500, detail="Could not load PPI data") 
  
    try:       
        # Load custom DNN weights
        dnn_data = await dnn_weights.read()
        dnn_buffer = io.BytesIO(dnn_data)        
        try:
            weights_checkpoint = torch.load(dnn_buffer, map_location='cpu')
            model = load_custom_dnn_model(weights_checkpoint, embeddings)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not load custom DNN weights: {e}")
     
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load model: {e}")
    
    # Prepare test complexes from golden standard list
    PCs = []
    for pc in protein_complexes:
        if len(pc) > 2:
            if set(pc).issubset(set(list(protein_dict.keys()))):
                pc_map = [protein_dict[sub] for sub in pc]
                PCs.append(pc_map)

    PC = [sorted(i) for i in PCs]
    print(f"Converted {len(PC)} protein complexes to ID format")

    # Generate negative examples
    print("Generating negative protein complexes...")
    PC_negative = negative_on_distribution(PC, list(ppi_dict.keys()), 5)

    # create labels
    postive_labels = torch.ones(len(PC), 1, dtype=torch.float)
    negative_labels = torch.zeros(len(PC_negative), 1, dtype=torch.float) 
    all_labels = torch.cat((postive_labels, negative_labels), dim=0)

    # Combine positive and negative complexes
    all_complexes = PC + PC_negative
    

    all_idx = list(range(len(all_complexes)))
    np.random.shuffle(all_idx)  # Shuffle indices for randomness
    all_complexes = [all_complexes[i] for i in all_idx]
    all_labels = all_labels[all_idx]  # Shuffle labels accordingly
    print("all_labels shape:", all_labels.shape)

    print(f"Total evaluation set: {len(all_complexes)} complexes ({len(PC)} positive, {len(PC_negative)} negative)")

    # Predict using pretrained model
    try:
        predictions = predict_with_pretrained_dnn(model, embeddings, all_complexes)
        print("Predictions completed successfully.")

        # convert predictions to numpy for evaluation
        y_true = all_labels.detach().cpu().numpy()
        y_pred = predictions.detach().cpu().numpy()
        print("y_true shape:", y_true.shape)
        print("y_pred shape:", y_pred.shape) 
        F1_score, threshold, Precision, Recall, Sensitivity = calculate_fmax(y_pred, y_true) 
        print("F1 Score:", F1_score)
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred)
        print("precision_curve", precision_curve)
        # auprc = auc(recall_curve, precision_curve)
        # auroc = roc_auc_score(y_true, y_pred)
        

        predicted_complexes = [all_complexes[i] for i in range(len(y_pred)) if y_pred[i] >= threshold]
        print(f"Predicted complexes: {len(predicted_complexes)} complexes above threshold {threshold}")
          # Calculate evaluation scores
        precision, recall, f1, acc, sn, PPV, score = get_score(PC, predicted_complexes)
        print("evaluation completed successfully.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
    # Return the results and predicted complexes

    # create reverse protein dictionary for complex conversion
    id_to_gene = {v: k for k, v in protein_dict.items()}

    predicted_complexes_genes = []
    for complex_ids in predicted_complexes:
        complex_genes = []
        for protein_id in complex_ids:
            if protein_id in id_to_gene:
                complex_genes.append(id_to_gene[protein_id])
            else:
                complex_genes.append(str(protein_id))  # Keep ID if no gene symbol found
        predicted_complexes_genes.append(complex_genes)

    # Calculate overlap scores for each predicted complex
    overlap_scores = calculate_overlap_scores(predicted_complexes, PC)
    
    print("Precision:", Precision, precision)
    print("Recall:", Recall, recall)
    print("F1 Score:", F1_score, f1)
    print(f"Calculated overlap scores for {len(overlap_scores)} predicted complexes")

    response_data = {
        "embeddings_file": embeddings_weights.filename,
        # "results": results,
        "predicted_complexes": predicted_complexes_genes,
        "overlap_scores": overlap_scores,
        "threshold": threshold,
        "model_source": "custom_weights" if dnn_weights is not None else "pretrained",
        "metrics": {
            "F1_score": F1_score,
            "Precision": precision,
            "Recall": Recall,
            "Sensitivity": sn,
            "Accuracy": acc,
        }
    }
    
    if dnn_weights is not None:
        response_data["dnn_weights_filename"] = dnn_weights.filename
    
    return response_data

def calculate_overlap_scores(predicted_complexes, reference_complexes):
    """
    Calculate overlap scores for each predicted complex.
    
    Args:
        predicted_complexes: List of predicted protein complexes
        reference_complexes: List of reference (golden standard) protein complexes
    
    Returns:
        List of overlap scores, one for each predicted complex
    """
    overlap_scores = []
    
    for pred in predicted_complexes:
        max_overlap_score = 0.0
        for ref in reference_complexes:
            set1 = set(ref)
            set2 = set(pred)
            overlap = set1 & set2
            score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
            if score > max_overlap_score:
                max_overlap_score = score
        overlap_scores.append(round(max_overlap_score, 4))
    
    return overlap_scores

EPSILON = 0.6
MIN_ROW = 200
MIN_COL = 4
DYNAMIC_SUBNETS_COUNT = 30
DGE_FOLDER_PATH = "./discretized_gene_expression_data"
DYNAMIC_PPI_FOLDER_PATH = "./dataset/dynamic_PPINs"
STATIC_PPI_FOLDER_PATH = "./static_PPINs"
# Metaheuristics parameters
METAHEURISTIC_PARAMS = {
    "GA": {  # Genetic Algorithm
        "DGE_df": pd.DataFrame,
        "min_row": MIN_ROW,
        "min_col": MIN_COL,
        "population_size": 400,  # 400
        "result_size": DYNAMIC_SUBNETS_COUNT,
        "max_generations": 200,  # 100
        "crossover_rate": 0.7,
        "mutation_rate": 0.05,
        "elitism_ratio": 0.1,  # 4
    },
    "SA": {  # Simulated Anealing
        "DGE_df": pd.DataFrame,
        "min_row": MIN_ROW,
        "min_col": MIN_COL,
        "initial_temperature": 0.01,  # 0.01
        "final_temperature": 0.00001,  # 0.000000001
        "cooling_rate": 0.9975,  # 0.99
        "max_iterations": 10000,  # 10000
        "neighborhood_size": 4,  # 10
    },
    "CS": {  # Cuckoo Search
        "DGE_df": pd.DataFrame,
        "min_row": MIN_ROW,
        "min_col": MIN_COL,
        "population_size": 400,
        "result_size": DYNAMIC_SUBNETS_COUNT,
        "max_generations": 100,
        "discovery_rate": 0.3,
        "levy_alpha": 2.0,
        "levy_beta": 1.5,
    },
}

@app.post("/get-dynamic-ppi")
async def get_dynamic_ppi(
    dataset_name: str = Body(...),
    metaheuristic_name: str = Body(...)
):    
    # get dataset path
    static_ppi_path = STATIC_PPI_FOLDER_PATH + "/" + dataset_name + ".tsv"

    # get static PPI network
    static_ppi_network = load_ppi_data_imad(static_ppi_path)
    print(f"Loaded protein-protein interactions from {dataset_name} dataset")
    print(f"{dataset_name} static PPI network size:")
    
   # load discretized gene expression data
    DGE_DF = pd.read_csv(
        DGE_FOLDER_PATH + "/" + dataset_name + "_DGE.tsv", sep="\t", index_col=0
    )
    print("Loaded discretized gene expression data")
    print("Discretized gene expression data size:")
    print(f"    - {DGE_DF.shape[0]} Proteins")
    print(f"    - {DGE_DF.shape[1]} Time points")


    METAHEURISTIC_PARAMS["SA"]["DGE_df"] = DGE_DF
    METAHEURISTIC_PARAMS["GA"]["DGE_df"] = DGE_DF
    METAHEURISTIC_PARAMS["CS"]["DGE_df"] = DGE_DF


    biclustering_mean_fitness = {}
    match metaheuristic_name:
        case "GA":
                obj = biclustering.GeneticAlgorithm(METAHEURISTIC_PARAMS["GA"])
                obj.optim(debug=True)
                best_biclusters = obj.final_biclusters
                del obj            

        case "CS":
                obj = biclustering.CuckooSearch(METAHEURISTIC_PARAMS["CS"])
                obj.optim(debug=True)
                best_biclusters = obj.final_biclusters
                del obj

    print(
        f"generating dynamic PPI data from biclustering results for {dataset_name} dataset"
    )
    dynamic_ppi_networks = generate_dynamic_ppi_data(
        best_biclusters,
        DGE_DF,
        static_ppi_network,
        (DYNAMIC_PPI_FOLDER_PATH + "/" + dataset_name + "_dynamic"),
        dataset_name,
        print_results=True,
    )

    return dynamic_ppi_networks





    