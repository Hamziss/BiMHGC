#!/usr/bin/env python3
"""
compute_enrichment_scores.py

Calcule, pour chaque dataset, le nombre de complexes (PC) et
la proportion de complexes « significatifs » ou
« extrêmement significatifs » selon p_fdr_bh.

Usage simple (racine contenant les sous-dossiers Collins, Krogan14K, …) :
    python compute_enrichment_scores.py /path/to/results_root

Usage avec un seul dossier (pas de sous-dossiers) :
    python compute_enrichment_scores.py /path/to/one_dataset

Dépendances : pandas (pip install pandas)
"""

import sys
import pathlib
from collections import defaultdict
import pandas as pd

ALPHA_EXTREME = 0.001   # seuil « extremely significant »
ALPHA_SIGNIF  = 0.05   # seuil « significant »

def collect_files(dataset_dir: pathlib.Path):
    """Retourne une liste de fichiers TSV d’un dataset donné."""
    return list(dataset_dir.glob("complex_*_*_enrichment.tsv"))

def complex_id_from_filename(fname: str) -> str:
    """
    Extrait l’identifiant du complexe à partir d’un nom de fichier :
    complex_123_BP_enrichment.tsv  -->  complex_123
    """
    parts = fname.split("_")
    return "_".join(parts[:2])  # 'complex' + number

def analyse_dataset(dataset_dir: pathlib.Path):
    """
    Parcourt tous les fichiers TSV du dossier et retourne :
    total_pc, nb_extreme, nb_signif
    """
    # Map complexe -> plus petite p_fdr_bh observée (toutes ontologies confondues)
    min_pvalues = defaultdict(lambda: 1.0)

    for tsv_file in collect_files(dataset_dir):
        try:
            df = pd.read_csv(tsv_file, sep="\t", low_memory=False)
        except Exception as e:
            print(f"(!) Impossible de lire {tsv_file}: {e}", file=sys.stderr)
            continue
     
        if "p_fdr_bh" not in df.columns:
            print(f"(!) Colonne p_fdr_bh manquante dans {tsv_file}", file=sys.stderr)
            continue

        cid = complex_id_from_filename(tsv_file.name)
        cur_min = df["p_fdr_bh"].min(skipna=True)
        if pd.notna(cur_min):
            min_pvalues[cid] = min(min_pvalues[cid], cur_min)

    total_pc = len(min_pvalues)
    nb_extreme = sum(p <= ALPHA_EXTREME for p in min_pvalues.values())
    nb_signif  = sum(p <= ALPHA_SIGNIF  for p in min_pvalues.values())

    return total_pc, nb_extreme, nb_signif

def main(root_path: str):
    root = pathlib.Path(root_path).resolve()
    if not root.exists():
        sys.exit(f"Chemin {root} introuvable.")

    # Si la racine contient déjà des fichiers TSV, on le traite comme un dataset unique
    root_has_tsv = any(root.glob("complex_*_*_enrichment.tsv"))
    dataset_dirs = [root] if root_has_tsv else [
        d for d in root.iterdir() if d.is_dir()
    ]

    summary_rows = []
    print("Analyse des datasets dans ",  dataset_dirs)
    for d in sorted(dataset_dirs):
        print(f"\nTraitement du dataset : {d.name}", file=sys.stderr)
        total_pc, nb_extreme, nb_signif = analyse_dataset(d)
        print("total_pc:", total_pc, "nb_extreme:", nb_extreme, "nb_signif:", nb_signif)
        if total_pc == 0:
            print(f"(i) Aucun fichier TSV trouvé dans {d}", file=sys.stderr)
            continue

        pct_extreme = nb_extreme / total_pc * 100
        pct_signif  = nb_signif  / total_pc * 100

        summary_rows.append({
            "Dataset": d.name,
            "PC": total_pc,
            "PC_extreme": nb_extreme,
            "%_extreme": f"{pct_extreme:.1f}%",
            "PC_signif": nb_signif,
            "%_signif": f"{pct_signif:.1f}%"
        })

    if not summary_rows:
        sys.exit("Aucun résultat exploitable n’a été trouvé.")

    # Affichage joli
    df_sum = pd.DataFrame(summary_rows)
    print("\n=== Tableau récapitulatif ===")
    print(df_sum.to_string(index=False))

    # Option : enregistrer en CSV
    out_csv = root / "enrichment_summary.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"\nRésumé enregistré dans {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python compute_enrichment_scores.py <chemin_dossier_racine>")
    main(sys.argv[1])
