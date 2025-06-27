import pandas as pd
from fastapi import Body, APIRouter
from typing import Dict, Any, Optional
from src.biclustering.utils import generate_dynamic_ppi_data,load_ppi_data
import src.biclustering.metaheuristics as metaheuristics
import os

EPSILON = 0.6
MIN_ROW = 200
MIN_COL = 4
DYNAMIC_SUBNETS_COUNT = 30

# DGE_FOLDER_PATH = "../../../data/Saccharomyces_cerevisiae/discretized_gene_expression_data"
DGE_FOLDER_PATH =  os.path.join(os.path.dirname(__file__), "..", "..","..","data", "Saccharomyces_cerevisiae", "discretized_gene_expression_data")
DYNAMIC_PPI_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "Saccharomyces_cerevisiae", "biclusters")
# DYNAMIC_PPI_FOLDER_PATH = "../../../data/Saccharomyces_cerevisiae/biclusters"
STATIC_PPI_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "Saccharomyces_cerevisiae", "static_PPINs")
# STATIC_PPI_FOLDER_PATH = "../../../data/Saccharomyces_cerevisiae/static_PPINs"
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

router = APIRouter(prefix="/get-dynamic-ppi", tags=["dynamic-ppi"])
@router.post("", summary="Generate dynamic PPI via metaheuristics")
async def get_dynamic_ppi(
    dataset_name: str = Body(...),
    metaheuristic_name: str = Body(...),
    ga: Optional[Dict[str, Any]] = Body(None),
    sa: Optional[Dict[str, Any]] = Body(None),
    cs: Optional[Dict[str, Any]] = Body(None)
):    
    # get dataset path
    static_ppi_path = STATIC_PPI_FOLDER_PATH + "/" + dataset_name + ".tsv"

    # get static PPI network
    static_ppi_network = load_ppi_data(static_ppi_path)
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

    # Build metaheuristic parameters from request body or use defaults
    ga_params = {
        "DGE_df": DGE_DF,
        "min_row": MIN_ROW,
        "min_col": MIN_COL,
        "population_size": int(ga.get("population_size", "400")) if ga else 400,
        "result_size": DYNAMIC_SUBNETS_COUNT,
        "max_generations": int(ga.get("max_generations", "200")) if ga else 200,
        "crossover_rate": float(ga.get("crossover_rate", "0.7")) if ga else 0.7,
        "mutation_rate": float(ga.get("mutation_rate", "0.05")) if ga else 0.05,
        "elitism_ratio": float(ga.get("elitism_ratio", "0.1")) if ga else 0.1,
    }
    
    sa_params = {
        "DGE_df": DGE_DF,
        "min_row": MIN_ROW,
        "min_col": MIN_COL,
        "initial_temperature": float(sa.get("initial_temperature", "0.01")) if sa else 0.01,
        "final_temperature": float(sa.get("final_temperature", "0.00001")) if sa else 0.00001,
        "cooling_rate": float(sa.get("cooling_rate", "0.9975")) if sa else 0.9975,
        "max_iterations": int(sa.get("max_iterations", "10000")) if sa else 10000,
        "neighborhood_size": 4,  # Keep default as it's not in the example
    }
    
    cs_params = {
        "DGE_df": DGE_DF,
        "min_row": MIN_ROW,
        "min_col": MIN_COL,
        "population_size": int(cs.get("population_size", "400")) if cs else 400,
        "result_size": DYNAMIC_SUBNETS_COUNT,
        "max_generations": int(cs.get("max_generations", "100")) if cs else 100,
        "discovery_rate": float(cs.get("discovery_rate", "0.3")) if cs else 0.3,
        "levy_alpha": float(cs.get("levy_alpha", "2.0")) if cs else 2.0,
        "levy_beta": float(cs.get("levy_beta", "1.5")) if cs else 1.5,
    }

    biclustering_mean_fitness = {}
    match metaheuristic_name:
        case "GA":
                obj = metaheuristics.GeneticAlgorithm(ga_params)
                obj.optim(debug=True)
                best_biclusters = obj.final_biclusters
                del obj            
        case "CSSA":
                obj = metaheuristics.CuckooSearchSA(cs_params, sa_params)
                obj.optim(debug=True)
                best_biclusters = obj.final_biclusters
                del obj            
        case "GASA":
                obj = metaheuristics.GeneticAlgorithmSA(ga_params, sa_params)
                obj.optim(debug=True)
                best_biclusters = obj.final_biclusters
                del obj            
        case "CS":
                obj = metaheuristics.CuckooSearch(cs_params)
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
        (DYNAMIC_PPI_FOLDER_PATH + "/" + dataset_name + "_" + metaheuristic_name),
        dataset_name,
        print_results=True,
        metaheuristic_name=metaheuristic_name,
    )

    return dynamic_ppi_networks



