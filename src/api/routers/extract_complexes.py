# app/routers/extract_complexes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.utils.helpers import negative_on_distribution
from typing import Optional
from src.models.inference import predict_with_pretrained_dnn
from src.utils.loaders import load_custom_dnn_model, load_ppi_data
from src.evaluation.metrics import calculate_fmax, get_score, calculate_overlap_scores
from sklearn.metrics import precision_recall_curve
from ...utils.helpers import load_txt_list 
import numpy as np
import io, torch
import pandas as pd

import os
# from ..services.dnn_loader import load_custom_dnn_model, load_pretrained_model
# from ..services.prediction import run_prediction
# from ..utils.io_utils import save_complexes, load_txt_list, load_protein_list
# from ..utils.data_loading import load_ppi_data
# from ..utils.metrics import calculate_overlap_scores

router = APIRouter(prefix="/extract-complexes", tags=["extract"])

@router.post("", summary="Run DNN on embeddings to extract complexes")
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
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "..","..","data")
    species = "Saccharomyces_cerevisiae"
    
    # Load protein complexes from the golden standard file
    protein_complexes = load_txt_list(os.path.join(data_path, species, "protein_complex"), '/AdaPPI_golden_standard.txt')
    
    # Load protein dictionary from the protein list file
    embeddings_filename = embeddings_weights.filename.rsplit('.', 1)[0]

    DATASETS = ["biogrid", "krogan14k", "dip", "collins"]

    
    dataset_base = next((d for d in DATASETS if embeddings_filename.startswith(d)), None)
    if dataset_base is None:
        raise ValueError(f"Unknown dataset prefix in '{dataset_base}'")

    print ("embeddings_filename", embeddings_filename)
        
    protein_list_path = os.path.join(data_path, species, "Gene_Entry_ID_list", embeddings_filename, "Protein_list.csv")
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
        F1_score, threshold, Precision, Recall, Sensitivity, *rest = calculate_fmax(y_pred, y_true) 
        print("F1 Score:", F1_score)
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred)
        print("precision_curve", precision_curve)
        # auprc = auc(recall_curve, precision_curve)
        # auroc = roc_auc_score(y_true, y_pred)
        
        all_predicted_complexes = [all_complexes[i] for i in range(len(y_pred))] 
        print(f"Total predicted complexes: {len(all_predicted_complexes)}")
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
    all_predicted_complexes_genes = []
    for complex_ids in all_predicted_complexes:
        complex_genes = []
        for protein_id in complex_ids:
            if protein_id in id_to_gene:
                complex_genes.append(id_to_gene[protein_id])
            else:
                complex_genes.append(str(protein_id))
        all_predicted_complexes_genes.append(complex_genes)

    predicted_complexes_genes = []
    for complex_ids in predicted_complexes:
        complex_genes = []
        for protein_id in complex_ids:
            if protein_id in id_to_gene:
                complex_genes.append(id_to_gene[protein_id])
            else:
                complex_genes.append(str(protein_id))  # Keep ID if no gene symbol found
        predicted_complexes_genes.append(complex_genes)    # Calculate overlap scores for each predicted complex
    print("fjkdlqmfjdklfjqlkjfqlkd",predicted_complexes[:5])  # Show first 5 predicted complexes for debugging
    print("rsjeklflm", PC[:5])  # Show first 5 ground truth complexes for debugging
    overlap_scores = calculate_overlap_scores(predicted_complexes, PC)
    
    print("Precision:", Precision, precision)
    print("Recall:", Recall, recall)
    print("F1 Score:", F1_score, f1)
    print(f"Calculated overlap scores for {len(overlap_scores)} predicted complexes")

    # Save predicted complexes to local file
    embeddings_filename = embeddings_weights.filename.rsplit('.', 1)[0]  # Remove .pt extension
    output_filename = f"{embeddings_filename}_predicted_complexes.txt"
    output_all_filename = f"{embeddings_filename}_all_predicted_complexes.txt"
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "results", "predicted_complexes", output_filename)
    output_all_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "results", "predicted_complexes", output_all_filename)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_all_path), exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            for complex_genes in predicted_complexes_genes:
                f.write(' '.join(complex_genes) + '\n')
        print(f"Saved {len(predicted_complexes_genes)} predicted complexes to {output_path}")
        with open(output_all_path, 'w') as f:
            for complex_genes in all_predicted_complexes_genes:
                f.write(' '.join(complex_genes) + '\n')
        print(f"Saved all predicted complexes to {output_all_path}")
    except Exception as e:
        print(f"Warning: Could not save predicted complexes to file: {e}")

    response_data = {
        "embeddings_file": embeddings_weights.filename,
        "predicted_complexes": predicted_complexes_genes,
        "overlap_scores": overlap_scores,
        "threshold": threshold,
        "model_source": "custom_weights" if dnn_weights is not None else "pretrained",
        "output_file": output_filename,
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
