#!/usr/bin/env python3
"""
go_enrichment.py

Usage:
    python go_enrichment.py \
        --complexes complexes.txt \
        --gaf annotations.gaf \
        --obo go-basic.obo \
        [--background background.txt] \
        --outdir results/
"""

import os
import sys
import argparse
from goatools.obo_parser import GODag
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS


def read_complexes(path):
    """Read complexes: one complex per line; proteins split on whitespace or commas."""
    complexes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            genes = [g.strip() for g in line.replace(",", " ").split()]
            complexes.append(genes)
    return complexes


def read_background(path):
    """Optional custom background file: one gene symbol per line."""
    if not path:
        return None
    bg = []
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l and not l.startswith("#"):
                bg.append(l)
    return bg


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


def main(args):
    # 1) Load GO ontology
    print(f"Loading GO ontology from {args.obo}", file=sys.stderr)
    godag = GODag(args.obo)

    # 2) Parse GAF manually to build CC/BP/MF associations
    print(f"Parsing GAF from {args.gaf}", file=sys.stderr)
    assoc_cc, assoc_bp, assoc_mf, syn_map = parse_gaf(args.gaf)

    # 3) Determine background universe (union of all annotated symbols if no custom file)
    bg = read_background(args.background)
    if bg is None:
        bg = sorted(set(assoc_cc) | set(assoc_bp) | set(assoc_mf))
        print(f"No background file; using {len(bg)} genes from GAF.", file=sys.stderr)
    else:
        print(f"Using custom background of {len(bg)} genes.", file=sys.stderr)

    # 4) Prepare output directory
    os.makedirs(args.outdir, exist_ok=True)

    # 5) Initialize GO enrichment study across three namespaces
    ns2assoc = {"CC": assoc_cc, "BP": assoc_bp, "MF": assoc_mf}
    goea = GOEnrichmentStudyNS(
        pop=bg,
        ns2assoc=ns2assoc,
        godag=godag,
        propagate_counts=True,
        alpha=0.05,
        methods=["fdr_bh"]
    )

    # 6) Process each complex
    complexes = read_complexes(args.complexes)
    for idx, original_genes in enumerate(complexes, start=1):
        name = f"complex_{idx}"
        mapped = []
        missing = []
        for g in original_genes:
            sym = syn_map.get(g)
            if sym:
                mapped.append(sym)
            else:
                missing.append(g)
        print(f"\n[{name}] {len(original_genes)} genes: {len(mapped)} mapped, {len(missing)} missing", file=sys.stderr)
        if missing:
            print(f"  Missing IDs: {missing}", file=sys.stderr)
        if not mapped:
            print(f"  No valid genes for {name}, skipping.", file=sys.stderr)
            continue

        # run enrichment for CC, BP, MF
        for ns in ("CC", "BP", "MF"):
            res = goea.ns2objgoea[ns].run_study(mapped)
            sig = [r for r in res if r.p_fdr_bh <= 0.05]
            if sig:
                out = os.path.join(args.outdir, f"{name}_{ns}_enrichment.tsv")
                # write with full precision for p-values
                goea.ns2objgoea[ns].wr_tsv(out, sig, pval_digits=58)
                print(f"  → {ns} results written to {out}", file=sys.stderr)
            else:
                print(f"  No significant {ns} enrichment for {name}", file=sys.stderr)

if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes",  required=True, help="File: one complex per line")
    p.add_argument("--gaf",        required=True, help="GO annotation GAF file")
    p.add_argument("--obo",        required=True, help="GO ontology OBO file")
    p.add_argument("--background", help="Optional: file of background gene symbols")
    p.add_argument("--outdir",     required=True, help="Directory for output TSVs")
    args = p.parse_args()
    main(args)
