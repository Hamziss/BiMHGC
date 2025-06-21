#!/usr/bin/env python3
"""
Benchmark script for testing pretrained DNN model with pretrained embeddings.

This script loads a pretrained DNN protein complex prediction model and evaluates it
on protein complexes using pretrained embeddings from the protein_feature directory.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import json
import pickle

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Train_PC import load_pretrained_dnn, predict_with_pretrained_dnn, HGC_DNN
from utils import (try_gpu, load_txt_list, convert_ppi, negative_on_distribution,
                   Nested_list_dup, preprocessing_PPI, sequence_CT)
from evaluation import calculate_fmax, get_score
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc


def load_protein_data(data_path, species):
    """Load protein dictionary and mappings."""
    print("Loading protein data...")
    
    # Load protein list mapping
    protein_list_path = os.path.join(data_path, species, "Gene_Entry_ID_list", "Protein_list.csv")
    if os.path.exists(protein_list_path):
        protein_df = pd.read_csv(protein_list_path, sep='\t', header=None, 
                                names=['Gene_symbol', 'Entry', 'ID'])
        protein_dict = dict(zip(protein_df['Gene_symbol'], protein_df['ID']))
        id_to_name = dict(zip(protein_df['ID'], protein_df['Gene_symbol']))
        print(f"Loaded {len(protein_dict)} proteins from protein list")
        return protein_dict, id_to_name, protein_df
    else:
        print(f"Protein list file not found at {protein_list_path}")
        return None, None, None


def load_pretrained_embeddings(data_path, species, embedding_type="protein_feature_Embeddings"):
    """Load pretrained protein embeddings."""
    print(f"Loading pretrained embeddings: {embedding_type}")
    
    embeddings_path = os.path.join(data_path, species, "protein_feature", f"{embedding_type}.pt")
    
    if os.path.exists(embeddings_path):
        embeddings = torch.load(embeddings_path, map_location='cpu')
        print(f"Loaded embeddings shape: {embeddings.shape}")
        return embeddings
    else:
        print(f"Embeddings file not found at {embeddings_path}")
        return None


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


def load_protein_complexes(data_path, species, pc_path="protein_complex"):
    """Load protein complex ground truth data."""
    print("Loading protein complexes...")
    
    pc_file_path = os.path.join(data_path, species, pc_path, "AdaPPI_golden_standard.txt")
    print(f"Looking for protein complex file at: {pc_file_path}")
    
    if os.path.exists(pc_file_path):
        # Load the file directly using the complete path
        list_data = []
        with open(pc_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                node_list = line.strip('\n').strip(' ').split(' ')
                list_data.append(node_list)
        
        print(f"Loaded {len(list_data)} protein complexes")
        return list_data
    else:
        print(f"Protein complex file not found at {pc_file_path}")
        return None


def benchmark_pretrained_model(model_path, embeddings, protein_complexes, protein_dict, ppi_dict, device, species="Saccharomyces_cerevisiae"):
    """Benchmark a pretrained DNN model."""
    print(f"\nBenchmarking pretrained model: {model_path}")
    
    # Load the pretrained model
    try:
        model, checkpoint = load_pretrained_dnn(model_path)
        model = model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Convert protein complexes to ID format
    PCs = []
    for pc in protein_complexes:
        if len(pc) > 2:  # At least 3 proteins in complex
            if set(pc).issubset(set(list(protein_dict.keys()))):
                pc_map = [protein_dict[sub] for sub in pc]
                PCs.append(pc_map)
    
    PC = [sorted(i) for i in PCs]
    print(f"Converted {len(PC)} protein complexes to ID format")
    
    if len(PC) == 0:
        print("No valid protein complexes found for evaluation")
        return None
    
    # Generate negative examples
    print("Generating negative protein complexes...")
    PC_negative = negative_on_distribution(PC, list(ppi_dict.keys()), 5)
    
    # Create labels
    positive_labels = torch.ones(len(PC), 1, dtype=torch.float)
    negative_labels = torch.zeros(len(PC_negative), 1, dtype=torch.float)
    all_labels = torch.cat((positive_labels, negative_labels), dim=0)
   
    
    # Combine positive and negative complexes
    all_complexes = PC + PC_negative
    
    # Shuffle the data
    all_idx = list(range(len(all_complexes)))
    np.random.shuffle(all_idx)
    all_complexes = [all_complexes[i] for i in all_idx]
    all_labels = all_labels[all_idx]
    print("all_labels shape:", all_labels.shape)
    
    print(f"Total evaluation set: {len(all_complexes)} complexes ({len(PC)} positive, {len(PC_negative)} negative)")
    
    # Make predictions
    embeddings = embeddings.to(device)
    predictions = predict_with_pretrained_dnn(model, embeddings, all_complexes)
    
    # Convert to numpy for evaluation
    y_true = all_labels.detach().cpu().numpy()
    y_pred = predictions.detach().cpu().numpy()
    print(f"Predictions made for {len(y_pred)} complexes")
    print(f"Predictions shape: {y_pred.shape}, Labels shape: {y_true.shape}")
    print("y_true shape:", y_true.shape)
    
    # Calculate metrics
    F1_score, threshold, Precision, Recall, Sensitivity, Specificity, ACC = calculate_fmax(y_pred, y_true)
    print("F1 Score:", F1_score)
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall_curve, precision_curve)
    auroc = roc_auc_score(y_true, y_pred)
      # Get predicted complexes above threshold
    predicted_complexes = [all_complexes[i] for i in range(len(y_pred)) if y_pred[i] > threshold]
      # Save predicted complexes
    dataset_info = {
        'species': species,
        'num_test_complexes': len(PC),
        'num_negative_complexes': len(PC_negative),
        'threshold_used': float(threshold),
        'embedding_shape': list(embeddings.shape)    }
    
    try:
        complexes_file = save_predicted_complexes(
            predicted_complexes, model_path, protein_dict, 
            os.path.join(os.path.dirname(__file__), "predicted_complexes"), 
            dataset_info
        )
    except Exception as e:
        print(f"Warning: Failed to save predicted complexes: {e}")
        complexes_file = None
    
    # Calculate complex-level metrics
    precision_complex, recall_complex, f1_complex, acc_complex, sn_complex, PPV_complex, score_complex = get_score(PC, predicted_complexes)
    results = {
        'model_path': model_path,
        'threshold': float(threshold),
        'binary_classification_metrics': {
            'F1_score': float(F1_score),
            'AUPRC': float(auprc),
            'AUROC': float(auroc),
            'Precision': float(Precision),
            'Recall': float(Recall),
            'Sensitivity': float(Sensitivity),
            'Specificity': float(Specificity),
            'ACC': float(ACC)
        },
        'complex_level_metrics': {
            'precision': float(precision_complex),
            'recall': float(recall_complex),
            'f1': float(f1_complex),
            'accuracy': float(acc_complex),
            'sensitivity': float(sn_complex),
            'PPV': float(PPV_complex)
        },
        'data_info': {
            'num_test_complexes': len(PC),
            'num_negative_complexes': len(PC_negative),
            'num_predicted_complexes': len(predicted_complexes),
            'embedding_shape': list(embeddings.shape)
        },        'saved_files': {
            'complexes_file': complexes_file
        }
    }
    
    print(f"\nResults Summary:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score: {F1_score:.4f}")
    print(f"Complex-level F1: {f1_complex:.4f}")
    print(f"Predicted {len(predicted_complexes)} complexes (threshold: {threshold:.4f})")
    
    return results


def benchmark_multiple_models(trained_models_dir, embeddings, protein_complexes, protein_dict, ppi_dict, device, species="Saccharomyces_cerevisiae"):
    """Benchmark multiple pretrained models."""
    print(f"Searching for models in: {trained_models_dir}")
    
    all_results = []
    
    # Find all model files
    model_files = []
    for root, dirs, files in os.walk(trained_models_dir):
        for file in files:
            if file.endswith('.pt') and ('dnn' in file.lower() or 'model' in file.lower()):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        print("No model files found")
        return []
    
    print(f"Found {len(model_files)} model files")
    
    for model_path in model_files:
        try:
            results = benchmark_pretrained_model(model_path, embeddings, protein_complexes, 
                                               protein_dict, ppi_dict, device)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking {model_path}: {e}")
            continue
    
    return all_results


def save_benchmark_results(results, output_dir, show_binary_metrics=False):
    """Save benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    results_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Create summary table
    if len(results) >= 1:
        summary_data = []
        for result in results:
            row_data = {
                'Model': os.path.basename(result['model_path']),
                'AUROC': result['binary_classification_metrics']['AUROC'],
                'AUPRC': result['binary_classification_metrics']['AUPRC'],
                'F1_Complex': result['complex_level_metrics']['f1'],
                'Precision_Complex': result['complex_level_metrics']['precision'],
                'Recall_Complex': result['complex_level_metrics']['recall'],
                'Accuracy_Complex': result['complex_level_metrics']['accuracy'],
                'Predicted_Complexes': result['data_info']['num_predicted_complexes'],
                'Threshold': result['threshold']
            }
            
            # Add binary classification metrics if requested
            if show_binary_metrics:
                row_data.update({
                    'F1_Binary': result['binary_classification_metrics']['F1_score'],
                    'Precision_Binary': result['binary_classification_metrics']['Precision'],
                    'Recall_Binary': result['binary_classification_metrics']['Recall'],
                    'Accuracy_Binary': result['binary_classification_metrics']['ACC']
                })
            
            summary_data.append(row_data)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary table saved to: {summary_file}")
        
        # Print summary
        print(f"\nBenchmark Summary:")
        print(summary_df.to_string(index=False))


def run_new_training_benchmark(data_path, species, embeddings, protein_complexes, protein_dict, ppi_dict):
    """Run a new training session for comparison."""
    print("\nRunning new training for comparison...")
    
    # Prepare dataset information
    dataset_info = {
        'species': species,
        'data_path': data_path,
        'benchmark_mode': True,
        'num_protein_complexes': len(protein_complexes),
        'num_proteins': len(protein_dict)
    }
    
    # Prepare model parameters
    model_params = {
        'model_type': 'HGC_DNN',
        'dnn_epochs': 500,  # Increased for better training
        'dnn_learning_rate': 0.001,
        'dnn_dropout_rate': 0.3,
        'cross_validation_folds': 5,
        'negative_sampling_ratio': 5
    }
    
    try:
        score = HGC_DNN(protein_complexes, protein_dict, ppi_dict, embeddings, 
                       save_predictions=True, dataset_info=dataset_info, model_params=model_params)
        print(f"New training completed with score: {score}")
        return score
    except Exception as e:
        print(f"Error in new training: {e}")
        return None


def save_predicted_complexes(predicted_complexes, model_path, protein_dict, output_dir, dataset_info=None):
    """Save predicted protein complexes to file with metadata."""
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_dir = datetime.now().strftime("%Y-%m-%d")
    complexes_output_dir = os.path.join(output_dir, date_dir)
    os.makedirs(complexes_output_dir, exist_ok=True)
    
    # Create reverse protein dictionary (ID to gene symbol)
    id_to_gene = {v: k for k, v in protein_dict.items()}
    
    # Convert predicted complexes back to gene symbols
    predicted_complexes_genes = []
    for complex_ids in predicted_complexes:
        complex_genes = []
        for protein_id in complex_ids:
            if protein_id in id_to_gene:
                complex_genes.append(id_to_gene[protein_id])
            else:
                complex_genes.append(str(protein_id))  # Keep ID if no gene symbol found
        predicted_complexes_genes.append(complex_genes)
      # Get model name from path
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Save complexes in text format (one complex per line)
    complexes_file = os.path.join(complexes_output_dir, f"predicted_complexes_{model_name}_{timestamp}.txt")
    with open(complexes_file, 'w') as f:
        for complex_genes in predicted_complexes_genes:
            f.write(' '.join(complex_genes) + '\n')
    
    print(f"Predicted complexes saved to: {complexes_file}")
    
    return complexes_file


def main():
    parser = argparse.ArgumentParser(description='Benchmark pretrained DNN models for protein complex prediction')
    parser.add_argument('--data_path', type=str, default="../data", 
                       help='Path to data directory')
    parser.add_argument('--species', type=str, default="Saccharomyces_cerevisiae", 
                       help='Species to use')
    parser.add_argument('--trained_models_dir', type=str, default="trained_models", 
                       help='Directory containing trained models')
    parser.add_argument('--embedding_type', type=str, default="protein_feature_Embeddings_collins_csa2_1",
                       help='Type of embeddings to use')
    parser.add_argument('--output_dir', type=str, default="benchmark_results",
                       help='Directory to save benchmark results')
    parser.add_argument('--run_new_training', action='store_true',
                       help='Also run new training for comparison')
    parser.add_argument('--ppi_path', type=str, default="collins_csa2_1",
                       help='PPI network directory')
    parser.add_argument('--show_binary_metrics', action='store_true',
                       help='Include binary classification metrics in summary table')
    parser.add_argument('--dataset_name', type=str, default="collins_csa2",
                       help='Name of the dataset to use')

    args = parser.parse_args()

    print("=" * 60)
    print("Protein Complex Prediction - Model Benchmarking")
    print("=" * 60)
    
    device = try_gpu()
    print(f"Using device: {device}")
    
    # Load all required data
    protein_dict, id_to_name, protein_df = load_protein_data(args.data_path, args.species)
    if protein_dict is None:
        print("Failed to load protein data")
        return
    
    embeddings = load_pretrained_embeddings(args.data_path, args.species, args.embedding_type)
    if embeddings is None:
        print("Failed to load embeddings")
        return
    ppi_list, ppi_dict = load_ppi_data(args.data_path, args.species, args.ppi_path, args.dataset_name)
    if ppi_list is None or ppi_dict is None:
        print("Failed to load PPI data")
        return
    
    protein_complexes = load_protein_complexes(args.data_path, args.species)
    if protein_complexes is None:
        print("Failed to load protein complexes")
        return
    
    # Benchmark pretrained models
    print("\n" + "=" * 40)
    print("BENCHMARKING PRETRAINED MODELS")
    print("=" * 40)
    
    trained_models_dir = args.trained_models_dir
    if not os.path.isabs(trained_models_dir):
        trained_models_dir = os.path.join(os.path.dirname(__file__), trained_models_dir)
    
    results = benchmark_multiple_models(trained_models_dir, embeddings, protein_complexes, 
                                      protein_dict, ppi_dict, device)
    
    # Run new training if requested
    if args.run_new_training:
        print("\n" + "=" * 40)
        print("RUNNING NEW TRAINING FOR COMPARISON")
        print("=" * 40)
        
        new_score = run_new_training_benchmark(args.data_path, args.species, embeddings, 
                                             protein_complexes, protein_dict, ppi_dict)
        if new_score:
            results.append({
                'model_path': 'NEW_TRAINING',
                'score': new_score,
                'type': 'fresh_training'
            })
      # Save results
    if results:
        save_benchmark_results(results, args.output_dir, args.show_binary_metrics)
    else:
        print("No results to save")
    
    print("\nBenchmarking completed!")


if __name__ == "__main__":
    main()
