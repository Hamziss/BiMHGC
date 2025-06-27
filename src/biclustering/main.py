import pandas as pd
import numpy as np
import random

def generate_random_bicluster(
    discretized_expression_df: pd.DataFrame, min_row: int, min_col: int
):
    """
    Generate a random bicluster represented as a binary array.

    Creates a binary array of length m+n where the first m elements represent
    row selections (proteins) and the last n elements represent column selections (time points).
    Ensures minimum constraints are met and then randomly includes additional rows/columns.

    Args:
        discretized_expression_df (pd.DataFrame): Discretized gene expression dataframe with
                                                  m rows (proteins) and n columns (time points)
        min_row (int): Minimum number of rows that must be selected in the bicluster
        min_col (int): Minimum number of columns that must be selected in the bicluster

    Returns:
        np.ndarray: Binary array of length m+n

    """
    num_proteins, time_points = discretized_expression_df.shape
    bicluster_array = np.zeros(num_proteins + time_points, dtype=int)

    # Ensure the minimum number of rows and columns is selected
    row_indeces = random.sample(range(num_proteins), min_row)
    col_indeces = random.sample(range(time_points), min_col)

    for i in row_indeces:
        bicluster_array[i] = 1

    for j in col_indeces:
        bicluster_array[num_proteins + j] = 1

    # Randomly select other rows and columns
    for i in range(num_proteins + time_points):
        if np.random.random() < 0.5:  # chance to include each row/column
            bicluster_array[i] = 1

    return bicluster_array


def is_bicluster_valid(
    bicluster_array: np.ndarray,
    discretized_expression_df: pd.DataFrame,
    min_row: int,
    min_col: int,
):
    """
    Validate if a bicluster meets the minimum size requirements.

    Checks whether the bicluster contains at least the minimum required number
    of rows and columns as specified by the constraints.

    Args:
        bicluster_array (np.ndarray): Binary array of length m+n representing the bicluster.
        discretized_expression_df (pd.DataFrame): Discretized gene expression dataframe.
        min_row (int): Minimum required number of rows in the bicluster.
        min_col (int): Minimum required number of columns in the bicluster.

    Returns:
        bool: True if bicluster meets minimum size requirements, False otherwise
    """
    num_proteins = discretized_expression_df.shape[0]
    if (
        sum(bicluster_array[:num_proteins]) < min_row
        or sum(bicluster_array[num_proteins:]) < min_col
    ):
        return False
    return True


def calculate_fitness(
    bicluster_array: np.ndarray, discretized_expression_df: pd.DataFrame
):
    """
    Calculate the fitness score of a bicluster based on the density of 1s.

    The fitness function evaluates the quality of a bicluster by computing the
    ratio of 1s (high expression) to the total number of elements in the bicluster.

    - The fitness is calculated as: (sum of 1s in bicluster) / (bicluster size).
    - Returns 0 if bicluster is empty to avoid division by zero.

    Args:
        bicluster_array (np.ndarray): Binary array of length m+n representing the bicluster
        discretized_expression_df (pd.DataFrame): Discretized gene expression dataframe.

    Returns:
        float: Fitness score between 0 and 1.
    """
    num_proteins = discretized_expression_df.shape[0]
    row_indices = np.where(bicluster_array[:num_proteins] == 1)[0]
    col_indices = np.where(bicluster_array[num_proteins:] == 1)[0]

    bicluster_size = len(row_indices) * len(col_indices)

    if bicluster_size == 0:
        return 0

    # Extract the bicluster
    bicluster = discretized_expression_df.iloc[row_indices, col_indices]

    score = bicluster.values.sum()

    return float(score / bicluster_size)
