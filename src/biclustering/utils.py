import pandas as pd
from pathlib import Path


def load_ppi_data(ppi_file_path: str) -> list[tuple]:
    """
    Load protein-protein interaction network data from a TSV file.

    Reads a tab-separated file containing protein interaction pairs and converts
    them into a list of tuples. Each line in the file should contain two protein
    names separated by a tab character.

    Args:
        ppi_file_path (str): Path to the PPI network TSV file

    Returns:
        list[tuple]: List of tuples where each tuple contains two protein names
                     representing an interaction (protein1, protein2). Returns
                     empty list if file cannot be loaded.
    """
    try:
        ppi_df = pd.read_csv(
            ppi_file_path, sep="\t", header=None, names=["protein1", "protein2"]
        )
        interactions = [
            (row["protein1"], row["protein2"]) for _, row in ppi_df.iterrows()
        ]
        return interactions
    except Exception as e:
        print(f"Error loading PPI network: {str(e)}")
        return []


def get_proteins_list(ppi_network: list[tuple]):
    """
    Extract all unique proteins from a protein-protein interaction network.

    Args:
        ppi_network (list[tuple]): List of tuples where each tuple contains two protein names
                                   (protein1, protein2) representing an interaction

    Returns:
        set: Set of unique protein names found in the network.
    """
    proteins = set()
    for protein1, protein2 in ppi_network:
        proteins.add(protein1)
        proteins.add(protein2)
    return proteins


def file_exists(path: str):
    """
    Check if a file or directory exists at the specified path.

    Args:
        path (str): File or directory path

    Returns:
        bool: True if the path exists, False otherwise
    """
    return Path(path).exists()


def generate_dynamic_ppi_data(
    biclustering_results: list[tuple],
    discretized_expression: pd.DataFrame,
    static_ppi_network: list[tuple],
    output_path: str,
    dataset_name: str,
    print_results=False,
    metaheuristic_name: str = "Unknown"  # Optional parameter for metaheuristic name
):
    """
    Save biclustering results as Dynamic PPI sub-networks and generate summary statistics.

    Processes the output of biclustering algorithms to create dynamic PPI sub-networks
    by filtering the static PPI network based on proteins selected in each bicluster.
    Generates individual TSV files for each bicluster's PPI sub-network and a summary
    file with statistics.

    Args:
        biclustering_results (list): List of tuples where each tuple contains:
                                     - bicluster_array (np.ndarray): Binary array indicating selected genes/timepoints
                                     - fitness (float): Fitness score of the bicluster
        discretized_expression (pd.DataFrame): Discretized gene expression dataframe
        static_ppi_network (list[tuple]): Static PPI network as list of protein interaction tuples
        output_path (str): Directory path where output files will be saved
        dataset_name (str): Name identifier for the dataset, used as prefix for output filenames
        print_results (bool): If True, prints summary statistics to console (default: False)

    Returns:
        None

    Side Effects:
        - Creates output directory if it doesn't exist
        - Generates individual TSV files for all dynamic PPI sub-networks
        - Creates "_results_info.tsv" with summary statistics
        - Prints confirmation message and optionally results summary

    Output Files:
        - {dataset_name}_{i}.tsv: Dynamic PPI sub-network for bicluster i (tab-separated protein pairs)
        - _results_info.tsv: Summary table with columns:
            - File Name: Name of the sub-network file
            - Fitness score: Quality score of the bicluster
            - Number of proteins: Count of unique proteins in the sub-network
            - Number of interactions: Count of interactions in the sub-network
    """
    m = discretized_expression.shape[0]

    # Create the folder if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    summary_data = {
        "File Name": [],
        "Fitness Score": [],
        "Protein Count": [],
        "Interaction Count": [],
    }

    print("Iterating on biclusters")
    dynamic_ppi_networks = []
    for i, (bicluster_array, fitness) in enumerate(biclustering_results):

        protein_set = set(discretized_expression.index[bicluster_array[:m] == 1])

        # Filter interactions in a single pass with optimized lookup
        filtered_interactions = [
            (protein1, protein2)
            for protein1, protein2 in static_ppi_network
            if protein1 in protein_set and protein2 in protein_set
        ]
        dynamic_ppi_networks.append(filtered_interactions)

        # Create output filename
        output_file_path = Path(output_path) / f"{dataset_name}_{metaheuristic_name}_{i + 1}.tsv"
        
        # Write all interactions at once using bulk I/O
        if filtered_interactions:
            interactions_text = "\n".join(
                f"{p1}\t{p2}" for p1, p2 in filtered_interactions
            )
            with open(output_file_path, "w") as f:
                f.write(interactions_text + "\n")
        else:
            # Create empty file if no interactions
            with open(output_file_path, "w") as f:
                pass

        summary_data["File Name"].append(f"{dataset_name}_{i + 1}")
        summary_data["Fitness Score"].append(fitness)
        summary_data["Protein Count"].append(
            len(get_proteins_list(filtered_interactions))
        )
        summary_data["Interaction Count"].append(len(filtered_interactions))

    df = pd.DataFrame(data=summary_data)
    df.to_csv((Path(output_path) / "_data_summary.tsv"), sep="\t", index=False)

    print("Dynamic PPI network saved in ", output_path)

    if print_results:
        print(f"\nDynamic PPI network for {dataset_name} dataset")
        print(df)
    return dynamic_ppi_networks


def print_bicluster(
    DGE_df: pd.DataFrame,
    dataset_name: str,
    metaheuristic: str,
    biclustering_results: dict,
):

    output_path = dataset_name + "_meta_benchmark"
    m = DGE_df.shape[0]

    sorted_results = sorted(biclustering_results, key=lambda x: x[1], reverse=True)

    data = {
        "Metaheuristic": [],
        "Fitness Score": [],
        "Protein Count": [],
        "Time Points": [],
    }

    # Create the folder if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for i, (bicluster_array, fitness) in enumerate(sorted_results):
        data["Metaheuristic"].append(f"{metaheuristic}_{i + 1}")
        data["Fitness Score"].append(fitness)
        data["Protein Count"].append(sum(bicluster_array[:m]))
        data["Time Points"].append(sum(bicluster_array[m:]))

    df = pd.DataFrame(data=data)
    df.to_csv((Path(output_path) / f"{metaheuristic}.tsv"), sep="\t", index=False)
