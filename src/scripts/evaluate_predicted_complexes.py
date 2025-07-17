#!/usr/bin/env python3
"""
Script to evaluate predicted protein complexes against golden standard.

This script takes predicted protein complexes (CSV format) and evaluates them
using precision, recall, F1-score, and accuracy metrics against the AdaPPI
golden standard complexes.

Usage:
    python evaluate_predicted_complexes.py <predicted_complexes_csv> [options]

Example:
    python evaluate_predicted_complexes.py ../data/results/predicted_complexes/Predict_PC_collins_GA.csv
"""

import os
import sys
import argparse
import pandas as pd
from typing import List, Tuple, Dict

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import get_score, precision_score, recall_score, acc_score, calculate_overlap_scores
from utils.helpers import load_txt_list


def load_predicted_complexes_from_csv(csv_path: str) -> List[List[str]]:
    """
    Load predicted complexes from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing predicted complexes
        
    Returns:
        List[List[str]]: List of predicted complexes, each as a list of proteins
    """
    print(f"Loading predicted complexes from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    predicted_complexes = []
    
    for _, row in df.iterrows():
        complex_str = row['predict_pc']
        proteins = complex_str.split(';')
        predicted_complexes.append(proteins)
    
    print(f"Loaded {len(predicted_complexes)} predicted complexes")
    return predicted_complexes


def load_golden_standard(data_path: str, species: str = "Saccharomyces_cerevisiae") -> List[List[str]]:
    """
    Load golden standard protein complexes.
    
    Args:
        data_path (str): Path to the data directory
        species (str): Species name (default: Saccharomyces_cerevisiae)
        
    Returns:
        List[List[str]]: List of golden standard complexes
    """
    golden_path = os.path.join(data_path, species, "protein_complex")
    golden_complexes = load_txt_list(golden_path, "/AdaPPI_golden_standard.txt")
    
    print(f"Loaded {len(golden_complexes)} golden standard complexes")
    return golden_complexes


def print_detailed_results(precision: float, recall: float, f1: float, acc: float, 
                          sn: float, ppv: float, predicted_complexes: List[List[str]], 
                          golden_complexes: List[List[str]]) -> None:
    """
    Print detailed evaluation results.
    
    Args:
        precision, recall, f1, acc, sn, ppv: Evaluation metrics
        predicted_complexes: List of predicted complexes
        golden_complexes: List of golden standard complexes
    """
    print("\n" + "="*80)
    print("PROTEIN COMPLEX PREDICTION EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Number of predicted complexes: {len(predicted_complexes)}")
    print(f"  Number of golden standard complexes: {len(golden_complexes)}")
    
    # Calculate average complex sizes
    avg_pred_size = sum(len(complex) for complex in predicted_complexes) / len(predicted_complexes)
    avg_golden_size = sum(len(complex) for complex in golden_complexes) / len(golden_complexes)
    
    print(f"  Average predicted complex size: {avg_pred_size:.2f}")
    print(f"  Average golden standard complex size: {avg_golden_size:.2f}")
    
    print(f"\nEvaluation Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Sensitivity (Sn): {sn:.4f}")
    print(f"  PPV:       {ppv:.4f}")
    
    print("\n" + "="*80)


def analyze_complex_sizes(predicted_complexes: List[List[str]], 
                         golden_complexes: List[List[str]]) -> None:
    """
    Analyze and print statistics about complex sizes.
    
    Args:
        predicted_complexes: List of predicted complexes
        golden_complexes: List of golden standard complexes
    """
    pred_sizes = [len(complex) for complex in predicted_complexes]
    golden_sizes = [len(complex) for complex in golden_complexes]
    
    print(f"\nComplex Size Analysis:")
    print(f"  Predicted complexes:")
    print(f"    Min size: {min(pred_sizes)}")
    print(f"    Max size: {max(pred_sizes)}")
    print(f"    Average size: {sum(pred_sizes)/len(pred_sizes):.2f}")
    
    print(f"  Golden standard complexes:")
    print(f"    Min size: {min(golden_sizes)}")
    print(f"    Max size: {max(golden_sizes)}")
    print(f"    Average size: {sum(golden_sizes)/len(golden_sizes):.2f}")


def save_evaluation_report(output_path: str, predicted_csv_path: str, 
                          precision: float, recall: float, f1: float, acc: float,
                          sn: float, ppv: float, predicted_complexes: List[List[str]], 
                          golden_complexes: List[List[str]]) -> None:
    """
    Save evaluation results to a report file.
    
    Args:
        output_path: Path to save the evaluation report
        predicted_csv_path: Path to the predicted complexes CSV
        precision, recall, f1, acc, sn, ppv: Evaluation metrics
        predicted_complexes: List of predicted complexes
        golden_complexes: List of golden standard complexes
    """
    with open(output_path, 'w') as f:
        f.write("Protein Complex Prediction Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Input file: {predicted_csv_path}\n")
        f.write(f"Golden standard: AdaPPI_golden_standard.txt\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write(f"  Predicted complexes: {len(predicted_complexes)}\n")
        f.write(f"  Golden standard complexes: {len(golden_complexes)}\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1-Score: {f1:.4f}\n")
        f.write(f"  Accuracy: {acc:.4f}\n")
        f.write(f"  Sensitivity (Sn): {sn:.4f}\n")
        f.write(f"  PPV: {ppv:.4f}\n")
    
    print(f"\nEvaluation report saved to: {output_path}")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate predicted protein complexes against golden standard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python evaluate_predicted_complexes.py data/results/predicted_complexes/Predict_PC_collins_GA.csv
    
    python evaluate_predicted_complexes.py data/results/predicted_complexes/Predict_PC_collins_GA.csv \\
        --data-path ./data --species Saccharomyces_cerevisiae --output-report results.txt
        """
    )
    
    parser.add_argument(
        'predicted_complexes_csv',
        help='Path to CSV file containing predicted complexes'
    )
    
    parser.add_argument(
        '--data-path',
        default='./data',
        help='Path to data directory (default: ./data)'
    )
    
    parser.add_argument(
        '--species',
        default='Saccharomyces_cerevisiae',
        help='Species name (default: Saccharomyces_cerevisiae)'
    )
    
    parser.add_argument(
        '--output-report',
        help='Path to save evaluation report (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed analysis including complex size statistics'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.predicted_complexes_csv):
        print(f"Error: Predicted complexes file not found: {args.predicted_complexes_csv}")
        sys.exit(1)
    
    # Check if running from the correct directory
    if not os.path.exists(args.data_path):
        print(f"Error: Data path not found: {args.data_path}")
        print("Make sure you're running the script from the project root directory")
        sys.exit(1)
    
    try:
        # Load data
        predicted_complexes = load_predicted_complexes_from_csv(args.predicted_complexes_csv)
        golden_complexes = load_golden_standard(args.data_path, args.species)
        
        # Calculate evaluation metrics
        print("\nCalculating evaluation metrics...")
        precision, recall, f1, acc, sn, ppv, msg = get_score(golden_complexes, predicted_complexes)
        
        # Print results
        print_detailed_results(precision, recall, f1, acc, sn, ppv, 
                             predicted_complexes, golden_complexes)
        
        if args.verbose:
            analyze_complex_sizes(predicted_complexes, golden_complexes)
            
            # Calculate and show overlap score distribution
            overlap_scores = calculate_overlap_scores(predicted_complexes, golden_complexes)
            print(f"\nOverlap Score Statistics:")
            print(f"  Min overlap score: {min(overlap_scores):.4f}")
            print(f"  Max overlap score: {max(overlap_scores):.4f}")
            print(f"  Average overlap score: {sum(overlap_scores)/len(overlap_scores):.4f}")
            
            # Count complexes with significant overlap (>0.25)
            significant_overlaps = [score for score in overlap_scores if score > 0.25]
            print(f"  Complexes with overlap > 0.25: {len(significant_overlaps)}/{len(overlap_scores)} ({len(significant_overlaps)/len(overlap_scores)*100:.1f}%)")
        
        # Save report if requested
        if args.output_report:
            save_evaluation_report(args.output_report, args.predicted_complexes_csv,
                                 precision, recall, f1, acc, sn, ppv,
                                 predicted_complexes, golden_complexes)
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
