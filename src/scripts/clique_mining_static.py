import torch
import networkx as nx
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.loaders import load_custom_dnn_model, load_ppi_data
from utils.helpers import subgraph_expansion, subgraph_filtration, convert_ppi, Nested_list_dup
from evaluation.metrics import calculate_overlap_scores

# Global variables for model and embeddings
model = None
X = None

def model_score(subgraphs):
    """Score subgraphs using the loaded DNN model."""
    probabilities = model(X, subgraphs)
    probabilities = probabilities.cpu().tolist()
    probabilities = [i[0] for i in probabilities]
    return probabilities

def load_reference_complexes(dataset_name="collins_GASA"):
    """Load reference complexes from golden standard."""
    reference_path = "./data/Saccharomyces_cerevisiae/protein_complex/AdaPPI_golden_standard.txt"
    print(f"Loading reference complexes from {reference_path}...")
    
    # Load protein mapping to convert gene names to IDs
    id_to_name = load_protein_mapping(dataset_name)
    name_to_id = {name: id for id, name in id_to_name.items()}
    
    reference_complexes = []
    with open(reference_path, 'r') as f:
        for line in f:
            proteins = line.strip().split()
            if len(proteins) >= 3:  # Only consider complexes with at least 3 proteins
                # Convert protein names to IDs
                complex_ids = []
                for protein_name in proteins:
                    if protein_name in name_to_id:
                        complex_ids.append(name_to_id[protein_name])
                
                if len(complex_ids) >= 3:  # Only keep if we have enough mapped proteins
                    reference_complexes.append(complex_ids)
    
    print(f"Loaded {len(reference_complexes)} reference complexes")
    return reference_complexes

def load_protein_mapping(dataset_name="collins_GASA"):
    """Load protein ID to name mapping."""
    mapping_path = f"./data/Saccharomyces_cerevisiae/Gene_Entry_ID_list/{dataset_name}/Protein_list.csv"
    print(f"Loading protein mapping from {mapping_path}...")
    
    # Load the mapping file (tab-separated)
    df = pd.read_csv(mapping_path, sep='\t', header=None, names=['gene_name', 'protein_id', 'index'])
    
    # Create mapping from index to gene name
    id_to_name = dict(zip(df['index'], df['gene_name']))
    print(f"Loaded mapping for {len(id_to_name)} proteins")
    
    return id_to_name

def load_model_and_embeddings():
    """Load the DNN model and embeddings."""
    global model, X
    
    # Load embeddings - now using collins_CSSA to match the static network
    embeddings_path = "./data/results/embeddings/collins_CSSA.pt"
    print(f"Loading embeddings from {embeddings_path}...")
    X = torch.load(embeddings_path)
    print(f"Loaded embeddings with shape: {X.shape}")
    
    # Load model
    model_path = "./trained_models/2025-07-17/run_07-30-33/dnn_ensemble_model.pt"
    print(f"Loading model from {model_path}...")
    weights_data = torch.load(model_path, map_location='cpu')
    model = load_custom_dnn_model(weights_data, X)
    model.eval()
    print("Model loaded successfully!")

def load_static_ppi_network(dataset_name="collins_CSSA"):
    """Load static PPI network from collins.tsv and map to numeric IDs."""
    print("Loading static PPI network from collins.tsv...")
    
    # Load the static PPI network
    static_ppi_path = "./data/Saccharomyces_cerevisiae/static_PPINs/collins.tsv"
    
    try:
        ppi_df = pd.read_csv(static_ppi_path, sep="\t", header=None, names=["protein1", "protein2"])
        print(f"Loaded {len(ppi_df)} interactions from static PPI network")
    except Exception as e:
        print(f"Error loading static PPI network: {e}")
        return None, None
    
    # Load protein mapping to convert gene names to numeric IDs
    mapping_path = f"./data/Saccharomyces_cerevisiae/Gene_Entry_ID_list/{dataset_name}/Protein_list.csv"
    
    try:
        mapping_df = pd.read_csv(mapping_path, sep='\t', header=None, names=['gene_name', 'protein_id', 'index'])
        gene_to_id = dict(zip(mapping_df['gene_name'], mapping_df['index']))
        print(f"Loaded protein mapping for {len(gene_to_id)} proteins")
    except Exception as e:
        print(f"Error loading protein mapping: {e}")
        return None, None
    
    # Map gene names to numeric IDs
    mapped_interactions = []
    unmapped_count = 0
    
    for _, row in ppi_df.iterrows():
        protein1 = row['protein1']
        protein2 = row['protein2']
        
        if protein1 in gene_to_id and protein2 in gene_to_id:
            id1 = gene_to_id[protein1]
            id2 = gene_to_id[protein2]
            # Add both directions for undirected graph
            mapped_interactions.append([id1, id2])
            mapped_interactions.append([id2, id1])
        else:
            unmapped_count += 1
    
    print(f"Successfully mapped {len(mapped_interactions)//2} interactions")
    print(f"Failed to map {unmapped_count} interactions (proteins not in mapping file)")
    
    if not mapped_interactions:
        print("No interactions could be mapped!")
        return None, None
    
    # Convert to the expected format
    ppi_list = Nested_list_dup(mapped_interactions)
    ppi_dict = convert_ppi(ppi_list)
    
    print(f"Final PPI network: {len(ppi_dict)} proteins with {sum(len(interactions) for interactions in ppi_dict.values())} total interactions")
    
    return ppi_list, ppi_dict

def load_ppi_network():
    """Load PPI network for Collins dataset - now using static network."""
    return load_static_ppi_network("collins_CSSA")

def find_maximal_cliques(ppi_list):
    """Find maximal cliques in the PPI network using NetworkX."""
    print("Finding maximal cliques in PPI network...")
    
    # Create NetworkX graph
    G = nx.Graph()
    for edge in ppi_list:
        if len(edge) >= 2:
            G.add_edge(edge[0], edge[1])
    
    # Find maximal cliques
    cliques = list(nx.find_cliques(G))
    print(f"Found {len(cliques)} maximal cliques")
    
    # Filter cliques by minimum size (e.g., at least 3 proteins for a complex)
    min_clique_size = 3
    filtered_cliques = [clique for clique in cliques if len(clique) >= min_clique_size]
    print(f"Filtered to {len(filtered_cliques)} cliques with size >= {min_clique_size}")
    
    return filtered_cliques

def clique_mining_algorithm(threshold_alpha=0.5, threshold_beta=0.8, score_threshold=0.8, output_dir="./data/results/predicted_complexes"):
    """
    Main clique mining algorithm for protein complex prediction.
    
    Args:
        threshold_alpha (float): Threshold for subgraph expansion
        threshold_beta (float): Threshold for subgraph filtration overlap
        score_threshold (float): Minimum score threshold for final complexes
        output_dir (str): Directory to save predicted complexes
    """
    print("Starting Clique Mining Algorithm for Protein Complex Prediction")
    print("=" * 60)
    
    # Step 1: Load model and embeddings
    load_model_and_embeddings()
    
    # Step 2: Load PPI network
    ppi_list, ppi_dict = load_ppi_network()
    
    # Step 3: Find initial maximal cliques as seed subgraphs
    seed_cliques = find_maximal_cliques(ppi_list)
    
    # Step 4: Subgraph Expansion
    print(f"\nStep 4: Expanding {len(seed_cliques)} seed cliques...")
    expanded_subgraphs = []
    
    for i, seed_clique in enumerate(seed_cliques):
        if i % 100 == 0:
            print(f"  Processing clique {i+1}/{len(seed_cliques)}")
        
        # Convert clique to list format expected by subgraph_expansion
        seed_subgraph = list(seed_clique)
        
        # Expand the subgraph
        expanded_subgraph = subgraph_expansion(
            seed_subgraph, ppi_dict, model_score, threshold_alpha
        )
        
        if len(expanded_subgraph) >= 3:  # Only keep complexes with at least 3 proteins
            expanded_subgraphs.append(expanded_subgraph)
    
    print(f"After expansion: {len(expanded_subgraphs)} candidate complexes")
    
    # Step 5: Score all expanded subgraphs
    print("\nStep 5: Scoring expanded subgraphs...")
    scores = model_score(expanded_subgraphs)
    print(f"Computed scores for {len(scores)} complexes")
    
    # Step 6: Subgraph Filtration
    print(f"\nStep 6: Filtering overlapping complexes (threshold_beta={threshold_beta})...")
    filtered_complexes = subgraph_filtration(
        expanded_subgraphs, scores, threshold_beta, model_score
    )
    
    print(f"After filtration: {len(filtered_complexes)} final complexes")
    
    # Step 7: Apply score threshold filter
    print(f"\nStep 7: Applying score threshold filter (>= {score_threshold})...")
    final_scores = model_score(filtered_complexes)
    
    # Filter complexes by score threshold
    high_score_complexes = []
    high_scores = []
    
    for complex_proteins, score in zip(filtered_complexes, final_scores):
        if score >= score_threshold:
            high_score_complexes.append(complex_proteins)
            high_scores.append(score)
    
    print(f"After score filtering: {len(high_score_complexes)} complexes with score >= {score_threshold}")
    
    # Step 8: Load reference complexes and calculate overlap scores
    print(f"\nStep 8: Loading reference complexes and calculating overlap scores...")
    reference_complexes = load_reference_complexes("collins_CSSA")

    if len(high_score_complexes) > 0 and len(reference_complexes) > 0:
        print(f"Calculating overlap scores for {len(high_score_complexes)} predicted complexes against {len(reference_complexes)} reference complexes...")
        
        # Calculate overlap scores for each predicted complex
        overlap_scores = calculate_overlap_scores(high_score_complexes, reference_complexes)
        
        print(f"\nOVERLAP SCORES FOR HIGH-SCORING COMPLEXES (>= {score_threshold}):")
        print("=" * 60)
        print(f"Average Overlap Score: {np.mean(overlap_scores):.4f}")
        print(f"Max Overlap Score: {max(overlap_scores):.4f}")
        print(f"Min Overlap Score: {min(overlap_scores):.4f}")
    else:
        print("No complexes or reference data available for overlap score calculation.")
        overlap_scores = []
    
    # Step 9: Load protein mapping for names
    print(f"\nStep 9: Loading protein name mapping...")
    id_to_name = load_protein_mapping("collins_CSSA")
    
    # Step 10: Save results in CSV format
    print(f"\nStep 10: Saving results to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV (only high-scoring complexes)
    csv_output_file = os.path.join(output_dir, f"clique_mined_complexes_filtered_{timestamp}.csv")
    
    # Prepare data for CSV
    csv_data = []
    for i, (complex_proteins, score) in enumerate(zip(high_score_complexes, high_scores)):
        # Convert protein IDs to names
        protein_names = []
        for protein_id in complex_proteins:
            if protein_id in id_to_name:
                protein_names.append(id_to_name[protein_id])
            else:
                protein_names.append(f"UNKNOWN_{protein_id}")
        
        # Add overlap score if available
        overlap_score = overlap_scores[i] if i < len(overlap_scores) else 0.0
        
        csv_data.append({
            'Complex_ID': f"Complex_{i+1}",
            'Size': len(complex_proteins),
            'Score': round(score, 4),
            'Overlap_Score': overlap_score,
            'Protein_Names': ';'.join(protein_names),
            'Protein_IDs': ';'.join(map(str, complex_proteins))
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_output_file, index=False)
    
    print(f"Filtered CSV results saved to: {csv_output_file}")
    
    # Save overlap scores summary
    if overlap_scores:
        overlap_file = os.path.join(output_dir, f"clique_mining_overlap_scores_{timestamp}.txt")
        with open(overlap_file, 'w') as f:
            f.write(f"Clique Mining Overlap Scores Summary\n")
            f.write(f"===================================\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Score Threshold: {score_threshold}\n")
            f.write(f"Number of Predicted Complexes: {len(high_score_complexes)}\n")
            f.write(f"Number of Reference Complexes: {len(reference_complexes)}\n")
            f.write(f"\nOverlap Score Statistics:\n")
            f.write(f"Average Overlap Score: {np.mean(overlap_scores):.4f}\n")
            f.write(f"Max Overlap Score: {max(overlap_scores):.4f}\n")
            f.write(f"Min Overlap Score: {min(overlap_scores):.4f}\n")
        
        print(f"Overlap scores summary saved to: {overlap_file}")
    
    # Also save a detailed version with one protein per row for easier analysis
    detailed_csv_file = os.path.join(output_dir, f"clique_mined_complexes_detailed_filtered_{timestamp}.csv")
    detailed_data = []
    
    for i, (complex_proteins, score) in enumerate(zip(high_score_complexes, high_scores)):
        complex_id = f"Complex_{i+1}"
        overlap_score = overlap_scores[i] if i < len(overlap_scores) else 0.0
        
        for protein_id in complex_proteins:
            protein_name = id_to_name.get(protein_id, f"UNKNOWN_{protein_id}")
            detailed_data.append({
                'Complex_ID': complex_id,
                'Complex_Size': len(complex_proteins),
                'Complex_Score': round(score, 4),
                'Complex_Overlap_Score': overlap_score,
                'Protein_ID': protein_id,
                'Protein_Name': protein_name
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(detailed_csv_file, index=False)
    
    print(f"Detailed filtered CSV results saved to: {detailed_csv_file}")
    
    # Step 11: Print summary statistics
    print("\n" + "=" * 60)
    print("CLIQUE MINING SUMMARY")
    print("=" * 60)
    print(f"Initial seed cliques: {len(seed_cliques)}")
    print(f"Expanded complexes: {len(expanded_subgraphs)}")
    print(f"Filtered complexes (before score filter): {len(filtered_complexes)}")
    print(f"Final high-scoring complexes (>= {score_threshold}): {len(high_score_complexes)}")
    
    if high_score_complexes:
        print(f"Average complex size: {np.mean([len(c) for c in high_score_complexes]):.2f}")
        print(f"Min complex size: {min([len(c) for c in high_score_complexes])}")
        print(f"Max complex size: {max([len(c) for c in high_score_complexes])}")
        print(f"Average score: {np.mean(high_scores):.4f}")
        print(f"Score range: {min(high_scores):.4f} - {max(high_scores):.4f}")
        
        if overlap_scores:
            print(f"\nOverlap Score Statistics:")
            print(f"Average Overlap Score: {np.mean(overlap_scores):.4f}")
            print(f"Max Overlap Score: {max(overlap_scores):.4f}")
            print(f"Min Overlap Score: {min(overlap_scores):.4f}")
    else:
        print("No complexes passed the score threshold!")
    
    return high_score_complexes, high_scores, overlap_scores

if __name__ == "__main__":
    # Run the clique mining algorithm with default parameters
    complexes, scores, overlap_scores = clique_mining_algorithm(
        threshold_alpha=0.9,      # Expansion threshold
        threshold_beta=0.8,       # Filtration overlap threshold
        score_threshold=0.9,      # Minimum score threshold for final output
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Generated {len(complexes)} high-quality protein complexes")
    if overlap_scores:
        print(f"Average Overlap Score: {np.mean(overlap_scores):.3f}")
    print("Results saved to CSV files in ./data/results/predicted_complexes/")
