import pandas as pd
from fastapi import Body, APIRouter, HTTPException
from typing import Dict, Any, Optional, List
from goatools.obo_parser import GODag
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
import os
from pathlib import Path

# Define the response models based on the required interface
class GOEnrichmentResult:
    def __init__(self, goId: str, term: str, pValue: float):
        self.goId = goId
        self.term = term
        self.pValue = pValue

class GOEnrichmentData:
    def __init__(self, biologicalProcess: List[GOEnrichmentResult], 
                 cellularComponent: List[GOEnrichmentResult], 
                 molecularFunction: List[GOEnrichmentResult]):
        self.biologicalProcess = biologicalProcess
        self.cellularComponent = cellularComponent
        self.molecularFunction = molecularFunction

def parse_gaf(gaf_path):
    """Parse GAF to build associations keyed by gene symbol for CC, BP, and MF."""
    assoc_cc = {}
    assoc_bp = {}
    assoc_mf = {}
    syn_map = {}
    with open(gaf_path) as gf:
        for ln in gf:
            if ln.startswith('!'):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 13:
                continue
            symbol = parts[2]
            go_id  = parts[4]
            aspect = parts[8]  # C, P or F
            # collect per-ontology associations
            if aspect == 'C':
                assoc_cc.setdefault(symbol, set()).add(go_id)
            elif aspect == 'P':
                assoc_bp.setdefault(symbol, set()).add(go_id)
            elif aspect == 'F':
                assoc_mf.setdefault(symbol, set()).add(go_id)
            # build synonym map: include symbol and listed synonyms
            syn_names = [symbol]
            syn_field = parts[10] if len(parts) > 10 else ''
            if syn_field:
                syn_names.extend(syn_field.split('|'))
            for n in syn_names:
                syn_map[n] = symbol
    return assoc_cc, assoc_bp, assoc_mf, syn_map

def perform_go_enrichment(proteins: List[str]):
    """Perform GO enrichment analysis on a list of proteins."""
    # Paths to GO data files
    base_path = Path(__file__).parent.parent.parent.parent
    gaf_path = base_path / "data/Saccharomyces_cerevisiae/GO/sgd.gaf"
    obo_path = base_path / "data/Saccharomyces_cerevisiae/GO/go-basic.obo"
    
    if not gaf_path.exists() or not obo_path.exists():
        raise HTTPException(status_code=500, detail="GO data files not found")
    
    # Load GO ontology
    godag = GODag(str(obo_path))
    
    # Parse GAF to build associations
    assoc_cc, assoc_bp, assoc_mf, syn_map = parse_gaf(str(gaf_path))
    
    # Determine background universe
    bg = sorted(set(assoc_cc) | set(assoc_bp) | set(assoc_mf))
    
    # Initialize GO enrichment study
    ns2assoc = {"CC": assoc_cc, "BP": assoc_bp, "MF": assoc_mf}
    goea = GOEnrichmentStudyNS(
        pop=bg,
        ns2assoc=ns2assoc,
        godag=godag,
        propagate_counts=True,
        alpha=0.05,
        methods=["fdr_bh"]
    )
    
    # Map input proteins to symbols
    mapped = []
    missing = []
    for protein in proteins:
        sym = syn_map.get(protein)
        if sym:
            mapped.append(sym)
        elif protein in bg:
            # Try direct mapping if protein name doesn't need synonyms
            mapped.append(protein)
        else:
            missing.append(protein)
    
    if not mapped:
        raise HTTPException(
            status_code=400, 
            detail=f"No valid proteins found in GO annotations. Missing: {missing[:10]}..."
        )
    
    # Initialize result structure
    result = {
        "biologicalProcess": [],
        "cellularComponent": [],
        "molecularFunction": []
    }
    
    # Namespace mapping for categorizing results
    namespace_mapping = {
        "BP": "biologicalProcess",
        "CC": "cellularComponent", 
        "MF": "molecularFunction"
    }
    
    # Get the top 1 result from each namespace
    for ns in ["BP", "CC", "MF"]:
        try:
            enrichment_results = goea.ns2objgoea[ns].run_study(mapped)
            
            # Filter and sort by p-value (FDR corrected)
            valid_results = [r for r in enrichment_results if hasattr(r, 'p_fdr_bh')]
            
            if valid_results:
                # Sort by p-value and get the best one
                valid_results.sort(key=lambda x: x.p_fdr_bh)
                best_result = valid_results[0]
                
                # Add the best result from this namespace
                go_result = {
                    "goId": best_result.GO,
                    "term": best_result.name,
                    "pValue": best_result.p_fdr_bh,
                    "ratio_in_study": best_result.ratio_in_study

                }
                result[namespace_mapping[ns]].append(go_result)
                
        except Exception as e:
            # Continue with other namespaces if there's an error
            continue
    
    return result

router = APIRouter(prefix="/go-enrichment", tags=["go-enrichment"])

@router.post("", summary="Run GO enrichment analysis on proteins")
async def get_go_enrichment(
    proteins: List[str] = Body(..., description="List of protein names/IDs to analyze")
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform GO enrichment analysis on a list of proteins.
    
    Returns the top 3 GO terms for each namespace:
    - biologicalProcess: Top 3 biological process GO terms
    - cellularComponent: Top 3 cellular component GO terms  
    - molecularFunction: Top 3 molecular function GO terms
    
    Each GO term includes:
    - goId: GO identifier (e.g., "GO:0008150")
    - term: Human-readable GO term name
    - pValue: FDR-corrected p-value
    """
    try:
        result = perform_go_enrichment(proteins)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing GO enrichment: {str(e)}")



