#!/usr/bin/env python3
# top_k_per_namespace.py
import sys, pathlib, pandas as pd

ALPHA_EXTREME = 0.001      # seuil FDR très strict (inutile ici mais conservé)
K = 4                     # combien de termes par namespace

def load_tsv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment=None, low_memory=False)
    df.columns = [c.lstrip("# ").strip() for c in df.columns]
    return df[["GO", "name", "NS", "p_fdr_bh"]]

def collect_terms(root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for f in root.rglob("complex_*_enrich*.tsv"):
        try:
            df = load_tsv(f)
        except Exception:
            continue
        if "p_fdr_bh" in df.columns:
            df["p_fdr_bh"] = pd.to_numeric(df["p_fdr_bh"], errors="coerce")
            rows.append(df.dropna(subset=["p_fdr_bh"]))
    if not rows:
        sys.exit("Aucun TSV lisible.")
    return pd.concat(rows, ignore_index=True)

def main(root_dir: str, k: int):
    root = pathlib.Path(root_dir).resolve()
    df = collect_terms(root)

    # Plus petite p_fdr_bh par (GO, name, NS)
    df_min = (df.groupby(["GO", "name", "NS"], as_index=False)["p_fdr_bh"]
                .min()
                .sort_values(["NS", "p_fdr_bh"]))

    # 🆕  Sélectionner les k meilleurs par namespace
    topk = (df_min.groupby("NS", as_index=False)
                   .head(k)
                   .reset_index(drop=True))

    # Affichage
    for ns in ["BP", "MF", "CC"]:
        subset = topk[topk["NS"] == ns]
        if subset.empty: continue
        print(f"\nTop {k} termes {ns}")
        print(subset.to_string(index=False,
                               formatters={"p_fdr_bh": "{:.2e}".format}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python top_k_per_namespace.py <dossier_racine> [k]")
    root_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else K
    main(root_path, k)
