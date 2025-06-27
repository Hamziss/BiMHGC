import utils

import pandas as pd
from pathlib import Path


def _load_gene_expression_data(proteins: set, gene_expression_path: str):
    """
    Load and process gene expression data for specified proteins.

    Only proteins present in proteins are included in output.

    Args:
        proteins (set): Set of protein names
        gene_expression_path (str): Path to the gene expression data file

    Returns:
        pd.DataFrame: DataFrame with proteins as rows and time points (T1-T12) as columns.
                      Each cell contains the average expression value across three test cycles.
    """
    expression_data = {}
    f = open(gene_expression_path, "r")
    for line in f:
        val = []
        line = line.strip().split()
        if len(line) == 38:
            if line[1] in proteins:
                # expression_list.update({line[1]: [line[i] for i in range(2, 38)]})
                for i in range(2, 12 + 2):
                    val.append(
                        (float(line[i]) + float(line[i + 12]) + float(line[i + 24])) / 3
                    )

                expression_data.update({line[1]: val})
    f.close()
    dataframe = pd.DataFrame(data=expression_data).T
    dataframe.columns = [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
    ]
    return dataframe


def _discretize_gene_expression(expression_df: pd.DataFrame, epsilon: float):
    """
    Discretize (binarize) gene expression data using statistical thresholding.

    This function converts continuous gene expression values to binary values (0 or 1)
    based on a threshold calculated from each gene's mean and standard deviation.
    The threshold is defined as |μ - σ * ε| where μ is the mean, σ is the population
    standard deviation, and ε is the penalty factor.

    Parameters:
        expression_df (pd.DataFrame): DataFrame containing continuous gene expression values.
        epsilon (float): Penalty factor for normalization threshold. Controls the sensitivity
                         of the binarization. Higher values make the threshold more strict.

    Returns:
        pd.DataFrame: Binarized DataFrame with same shape as input.
    """
    mu = expression_df.mean(axis=1)
    sigma = expression_df.std(axis=1, ddof=0)  # population standard deviation
    threshold = (mu - sigma * epsilon).abs()

    # Broadcast threshold across columns
    binary_df = expression_df.ge(threshold, axis=0).astype(int)
    return binary_df


def _save_to_tsv(
    dataframe: pd.DataFrame, output_folder_name: str, output_file_name: str
):
    """
    Save a DataFrame to a TSV file in the specified folder.

    Creates the output folder if it doesn't exist and saves the DataFrame
    as a tab-separated values file with row indices included.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save
        output_folder_name (str): path of the folder where the file will be saved
        output_file_name (str): Name of the output TSV file

    Returns:
        None

    Side Effects:
        - Creates the output folder if it doesn't exist
        - Saves the DataFrame to the specified file path
        - Prints confirmation message upon successful save
    """
    # Create the folder if it doesn't exist
    folder_path = Path(output_folder_name)
    folder_path.mkdir(parents=True, exist_ok=True)

    file_path = folder_path / output_file_name
    dataframe.to_csv(file_path, sep="\t", index=True)

    print(f"Discretized gene expression data successfully saved to '{folder_path}'")


# main
def generate_discretized_gene_expression(
    static_ppi_network: list[tuple],
    gene_expression_path: str,
    epsilon: float,
    output_folder: str,
    dataset_name: str,
):
    """
    Generate Discretized Gene Expression (DGE) data from PPI network and gene expression data.

    This is the main function that orchestrates the entire workflow:
    1. Extract unique proteins from the PPI network
    2. Load and process gene expression data for those proteins
    3. Discretize (binarize) the expression data
    4. Save the resulting DGE data to a file

    Args:
        static_ppi_network (list[tuple]): List of tuples representing protein-protein interactions.
                                          Each tuple contains two protein names (protein1, protein2).
        gene_expression_path (str): Path to the gene expression data file.
        epsilon (float): Penalty factor for discretization threshold. Higher values
                         create more strict thresholds for binarization.
        output_folder (str): Path to the folder where the DGE data will be saved.
                             Folder will be created if it doesn't exist.
        dataset_name (str): Name identifier for the PPI dataset.

    Returns:
        None

    Side Effects:
        - Creates output folder if it doesn't exist
        - Saves discretized gene expression data to TSV file
        - Prints progress and confirmation messages
    """
    protein_list = utils.get_proteins_list(static_ppi_network)

    GE_df = _load_gene_expression_data(protein_list, gene_expression_path)

    DGE_df = _discretize_gene_expression(GE_df, epsilon)

    output_file_name = dataset_name + "_DGE.tsv"
    _save_to_tsv(DGE_df, output_folder, output_file_name)
