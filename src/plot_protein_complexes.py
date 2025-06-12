#!/usr/bin/env python3
"""
Protein Complex Visualization Script

This script creates various visualizations for protein complexes from a CSV file
containing complex predictions with proteins, sizes, and prediction scores.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_complex_data(file_path):
    """Load protein complex data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} protein complexes from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def parse_proteins(protein_string):
    """Parse semicolon-separated protein string into list."""
    return protein_string.split(';')

def create_summary_plots(df, output_dir):
    """Create summary plots for protein complexes."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Protein Complex Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. Complex size distribution
    axes[0, 0].hist(df['Size'], bins=range(int(df['Size'].min()), int(df['Size'].max()) + 2), 
                    alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Complex Size (Number of Proteins)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Complex Sizes')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Prediction score distribution
    axes[0, 1].hist(df['Prediction_Score'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Prediction Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Prediction Scores')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Size vs Score scatter plot
    scatter = axes[1, 0].scatter(df['Size'], df['Prediction_Score'], 
                                alpha=0.6, c=df['Prediction_Score'], cmap='viridis', s=50)
    axes[1, 0].set_xlabel('Complex Size')
    axes[1, 0].set_ylabel('Prediction Score')
    axes[1, 0].set_title('Complex Size vs Prediction Score')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # 4. Top complexes by score
    top_complexes = df.nlargest(15, 'Prediction_Score')
    bars = axes[1, 1].barh(range(len(top_complexes)), top_complexes['Prediction_Score'], 
                          color='lightgreen', alpha=0.7)
    axes[1, 1].set_yticks(range(len(top_complexes)))
    axes[1, 1].set_yticklabels(top_complexes['Complex_ID'], fontsize=8)
    axes[1, 1].set_xlabel('Prediction Score')
    axes[1, 1].set_title('Top 15 Complexes by Prediction Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add score labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1, 1].text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complex_summary_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_protein_frequency_plot(df, output_dir, top_n=20):
    """Create plot showing most frequent proteins across complexes."""
    # Count protein frequencies
    all_proteins = []
    for proteins_str in df['Proteins']:
        all_proteins.extend(parse_proteins(proteins_str))
    
    protein_counts = Counter(all_proteins)
    top_proteins = protein_counts.most_common(top_n)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of top proteins
    proteins, counts = zip(*top_proteins)
    bars = ax1.bar(range(len(proteins)), counts, color='lightblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Proteins')
    ax1.set_ylabel('Frequency in Complexes')
    ax1.set_title(f'Top {top_n} Most Frequent Proteins')
    ax1.set_xticks(range(len(proteins)))
    ax1.set_xticklabels(proteins, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart of protein frequency distribution
    freq_distribution = Counter(protein_counts.values())
    frequencies = list(freq_distribution.keys())
    freq_counts = list(freq_distribution.values())
    
    ax2.pie(freq_counts, labels=[f'{f} complex(es)' for f in frequencies], autopct='%1.1f%%')
    ax2.set_title('Distribution of Protein Frequencies\n(How many complexes each protein appears in)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'protein_frequency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return protein_counts

def create_network_visualization(df, output_dir, min_score=0.8, max_complexes=20):
    """Create network visualization of protein interactions."""
    # Filter high-confidence complexes
    high_conf_df = df[df['Prediction_Score'] >= min_score].head(max_complexes)
    
    if len(high_conf_df) == 0:
        print(f"No complexes found with score >= {min_score}")
        return
    
    # Create network graph
    G = nx.Graph()
    complex_colors = {}
    color_map = plt.cm.Set3(np.linspace(0, 1, len(high_conf_df)))
    
    for idx, (_, row) in enumerate(high_conf_df.iterrows()):
        proteins = parse_proteins(row['Proteins'])
        complex_id = row['Complex_ID']
        score = row['Prediction_Score']
        
        # Add nodes and edges for this complex
        for i, protein1 in enumerate(proteins):
            G.add_node(protein1)
            for j, protein2 in enumerate(proteins[i+1:], i+1):
                G.add_edge(protein1, protein2, complex=complex_id, score=score)
                
        # Assign colors to proteins in this complex
        for protein in proteins:
            if protein not in complex_colors:
                complex_colors[protein] = color_map[idx]
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    node_colors = [complex_colors.get(node, 'lightgray') for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=300, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(f'Protein Interaction Network\n(Top {len(high_conf_df)} complexes with score ≥ {min_score})', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Add legend for complexes
    legend_elements = []
    for idx, (_, row) in enumerate(high_conf_df.iterrows()):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color_map[idx], markersize=10,
                                        label=f"{row['Complex_ID']} (score: {row['Prediction_Score']:.3f})"))
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'protein_network_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_heatmap_visualization(df, output_dir, top_proteins=30):
    """Create heatmap showing protein co-occurrence in complexes."""
    # Get most frequent proteins
    all_proteins = []
    for proteins_str in df['Proteins']:
        all_proteins.extend(parse_proteins(proteins_str))
    
    protein_counts = Counter(all_proteins)
    top_protein_names = [p[0] for p in protein_counts.most_common(top_proteins)]
    
    # Create co-occurrence matrix
    cooccurrence_matrix = np.zeros((len(top_protein_names), len(top_protein_names)))
    
    for _, row in df.iterrows():
        proteins = parse_proteins(row['Proteins'])
        proteins_in_top = [p for p in proteins if p in top_protein_names]
        
        for i, protein1 in enumerate(proteins_in_top):
            idx1 = top_protein_names.index(protein1)
            for protein2 in proteins_in_top[i:]:
                idx2 = top_protein_names.index(protein2)
                cooccurrence_matrix[idx1, idx2] += 1
                if idx1 != idx2:
                    cooccurrence_matrix[idx2, idx1] += 1
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence_matrix, 
                xticklabels=top_protein_names, 
                yticklabels=top_protein_names,
                annot=False, 
                cmap='YlOrRd', 
                square=True,
                cbar_kws={'label': 'Co-occurrence Count'})
    
    plt.title(f'Protein Co-occurrence Heatmap\n(Top {top_proteins} most frequent proteins)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Proteins')
    plt.ylabel('Proteins')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'protein_cooccurrence_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_complex_plot(df, output_dir, complex_ids=None):
    """Create detailed visualization for specific complexes."""
    if complex_ids is None:
        # Select top 5 complexes by score
        complex_ids = df.nlargest(5, 'Prediction_Score')['Complex_ID'].tolist()
    
    selected_complexes = df[df['Complex_ID'].isin(complex_ids)]
    
    fig, axes = plt.subplots(len(selected_complexes), 1, 
                            figsize=(12, 3 * len(selected_complexes)))
    if len(selected_complexes) == 1:
        axes = [axes]
    
    for idx, (_, row) in enumerate(selected_complexes.iterrows()):
        proteins = parse_proteins(row['Proteins'])
        
        # Create a small network for this complex
        G = nx.complete_graph(len(proteins))
        pos = nx.circular_layout(G)
        
        # Map node indices to protein names
        node_labels = {i: proteins[i] for i in range(len(proteins))}
        
        # Draw the complex
        nx.draw_networkx_nodes(G, pos, ax=axes[idx], node_color='lightblue', 
                              node_size=1000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=axes[idx], alpha=0.5, width=2)
        nx.draw_networkx_labels(G, pos, node_labels, ax=axes[idx], 
                               font_size=10, font_weight='bold')
        
        axes[idx].set_title(f"{row['Complex_ID']}: {len(proteins)} proteins, "
                           f"Score: {row['Prediction_Score']:.4f}", 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_complex_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_statistics_report(df, output_dir):
    """Generate and save a statistical report."""
    report = []
    report.append("PROTEIN COMPLEX ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Basic statistics
    report.append("BASIC STATISTICS:")
    report.append(f"Total number of complexes: {len(df)}")
    report.append(f"Average complex size: {df['Size'].mean():.2f}")
    report.append(f"Median complex size: {df['Size'].median():.2f}")
    report.append(f"Size range: {df['Size'].min()} - {df['Size'].max()}")
    report.append("")
    
    report.append(f"Average prediction score: {df['Prediction_Score'].mean():.4f}")
    report.append(f"Median prediction score: {df['Prediction_Score'].median():.4f}")
    report.append(f"Score range: {df['Prediction_Score'].min():.4f} - {df['Prediction_Score'].max():.4f}")
    report.append("")
    
    # Size distribution
    size_dist = df['Size'].value_counts().sort_index()
    report.append("SIZE DISTRIBUTION:")
    for size, count in size_dist.items():
        percentage = (count / len(df)) * 100
        report.append(f"Size {size}: {count} complexes ({percentage:.1f}%)")
    report.append("")
    
    # Top complexes
    report.append("TOP 10 COMPLEXES BY PREDICTION SCORE:")
    top_complexes = df.nlargest(10, 'Prediction_Score')
    for _, row in top_complexes.iterrows():
        report.append(f"{row['Complex_ID']}: {row['Prediction_Score']:.4f} "
                     f"(size: {row['Size']})")
    report.append("")
    
    # Protein statistics
    all_proteins = []
    for proteins_str in df['Proteins']:
        all_proteins.extend(parse_proteins(proteins_str))
    
    protein_counts = Counter(all_proteins)
    report.append(f"PROTEIN STATISTICS:")
    report.append(f"Total unique proteins: {len(protein_counts)}")
    report.append(f"Most frequent protein: {protein_counts.most_common(1)[0][0]} "
                 f"(appears in {protein_counts.most_common(1)[0][1]} complexes)")
    
    # Save report
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='Visualize protein complexes from CSV file')
    parser.add_argument('input_file', help='Path to input CSV file with protein complexes')
    parser.add_argument('--output_dir', '-o', default='complex_plots', 
                       help='Output directory for plots (default: complex_plots)')
    parser.add_argument('--min_score', type=float, default=0.8,
                       help='Minimum score for network visualization (default: 0.8)')
    parser.add_argument('--top_proteins', type=int, default=30,
                       help='Number of top proteins for heatmap (default: 30)')
    parser.add_argument('--complex_ids', nargs='+', 
                       help='Specific complex IDs to visualize in detail')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    df = load_complex_data(args.input_file)
    if df is None:
        return
    
    print(f"Creating visualizations in: {args.output_dir}")
    
    # Generate all plots
    print("1. Creating summary plots...")
    create_summary_plots(df, args.output_dir)
    
    print("2. Creating protein frequency analysis...")
    create_protein_frequency_plot(df, args.output_dir)
    
    print("3. Creating network visualization...")
    create_network_visualization(df, args.output_dir, min_score=args.min_score)
    
    print("4. Creating co-occurrence heatmap...")
    create_heatmap_visualization(df, args.output_dir, top_proteins=args.top_proteins)
    
    print("5. Creating detailed complex visualization...")
    create_detailed_complex_plot(df, args.output_dir, complex_ids=args.complex_ids)
    
    print("6. Generating statistics report...")
    generate_statistics_report(df, args.output_dir)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    # If no command line arguments, use the provided file as example
    if len(sys.argv) == 1:
        input_file = r"d:\code\ia\codesandbox\HGC-final-clean\src\predicted_complexes\2025-06-12\run_07-00-13\predicted_complexes_by_names.csv"
        output_dir = "complex_visualization_output"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df = load_complex_data(input_file)
        if df is not None:
            print(f"Creating visualizations in: {output_dir}")
            create_summary_plots(df, output_dir)
            create_protein_frequency_plot(df, output_dir)
            create_network_visualization(df, output_dir, min_score=0.8)
            create_heatmap_visualization(df, output_dir, top_proteins=25)
            create_detailed_complex_plot(df, output_dir)
            generate_statistics_report(df, output_dir)
            print(f"\nAll visualizations saved to: {output_dir}")
    else:
        main()
