"""
render_molecules.py
On-demand SVG rendering of molecules from failure CSV files.
Atom saliency scores are read from the 'atom_scores' JSON column
and mapped to atom colors.

Usage:
  # Render worst-10 molecules for GCN, homo target
  python render_molecules.py \\
      --csv results/01_QM9/raw_data/gcn_na_0_homo_failures.csv \\
      --out results/05_Interpretability/molecule_vis \\
      --mode worst --n 10

  # Render specific indices from the CSV
  python render_molecules.py \\
      --csv results/01_QM9/raw_data/gcn_na_0_homo_failures.csv \\
      --out results/05_Interpretability/molecule_vis \\
      --indices 0 3 7

  # Render all worst molecules across multiple CSVs
  python render_molecules.py \\
      --csv results/01_QM9/raw_data/gcn_na_0_homo_failures.csv \\
             results/01_QM9/raw_data/transformer_na_0_homo_failures.csv \\
      --out results/05_Interpretability/molecule_vis \\
      --mode worst --n 5

Requirements:
  pip install rdkit-pypi  (or rdkit in conda)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# SVG rendering helpers
# ---------------------------------------------------------------------------

def _scores_to_rgba(scores: list, cmap_pos=(0.2, 0.8, 0.2), cmap_neg=(0.9, 0.2, 0.2),
                    alpha: float = 0.6) -> list:
    """
    Map saliency scores to RGBA tuples for RDKit atom coloring.
    Positive scores → green, near-zero → white, (negative not expected but handled).
    Returns list of (R, G, B, A) tuples in [0, 1].
    """
    arr = np.array(scores, dtype=float)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())  # normalize to [0, 1]
    else:
        arr = np.zeros_like(arr)

    colors = []
    for v in arr:
        r = 1.0 - v * (1.0 - cmap_pos[0])
        g = 1.0 - v * (1.0 - cmap_pos[1])
        b = 1.0 - v * (1.0 - cmap_pos[2])
        colors.append((r, g, b, alpha))
    return colors


def render_molecule_svg(
    smiles: str,
    atom_scores: list = None,
    width: int = 400,
    height: int = 300,
    legend: str = "",
) -> str:
    """
    Render a molecule to SVG string using RDKit MolDraw2DSVG.
    If atom_scores is provided, colors atoms by saliency.
    Returns SVG string, or empty string on error.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        raise ImportError("rdkit not installed. Run: pip install rdkit-pypi")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True

    if atom_scores and len(atom_scores) == mol.GetNumAtoms():
        atom_colors = {}
        highlight_atoms = list(range(mol.GetNumAtoms()))
        rgba_list = _scores_to_rgba(atom_scores)
        for i, rgba in enumerate(rgba_list):
            atom_colors[i] = rgba[:3]  # RDKit uses (R, G, B) without alpha

        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=[],
            highlightBondColors={},
        )
    else:
        drawer.DrawMolecule(mol)

    if legend:
        drawer.DrawString(legend, (10, height - 15))

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ---------------------------------------------------------------------------
# Load failure CSV
# ---------------------------------------------------------------------------

def load_failure_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse atom_scores JSON column if present
    if 'atom_scores' in df.columns:
        def parse_scores(v):
            if pd.isna(v) or v == '' or v == '[]':
                return []
            try:
                return json.loads(v)
            except Exception:
                return []
        df['atom_scores'] = df['atom_scores'].apply(parse_scores)
    else:
        df['atom_scores'] = [[] for _ in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Render and save SVGs
# ---------------------------------------------------------------------------

def render_rows(df: pd.DataFrame, out_dir: str, prefix: str = "mol"):
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for i, row in df.iterrows():
        smiles = row.get('smiles', '')
        if not smiles:
            continue

        true_val  = row.get('true',  row.get('y_true',  float('nan')))
        pred_val  = row.get('pred',  row.get('y_pred',  float('nan')))
        abs_err   = abs(true_val - pred_val) if not (np.isnan(true_val) or np.isnan(pred_val)) else float('nan')
        atom_scores = row.get('atom_scores', [])

        legend = f"err={abs_err:.3f} | true={true_val:.3f} | pred={pred_val:.3f}"
        svg = render_molecule_svg(smiles, atom_scores=atom_scores if atom_scores else None,
                                  legend=legend)
        if not svg:
            print(f"  [warn] Could not render: {smiles[:40]}")
            continue

        fname = f"{prefix}_{i:04d}.svg"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, 'w') as f:
            f.write(svg)
        saved.append(fpath)
        print(f"  [saved] {fpath}")

    return saved


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Render molecule SVGs from failure CSV")
    p.add_argument('--csv', nargs='+', required=True,
                   help="Path(s) to failure CSV files")
    p.add_argument('--out', default='results/05_Interpretability/molecule_vis',
                   help="Output directory for SVG files")
    p.add_argument('--mode', choices=['worst', 'best', 'all'], default='worst',
                   help="Which molecules to render: worst-N, best-N, or all")
    p.add_argument('--n', type=int, default=10,
                   help="Number of molecules to render (for worst/best mode)")
    p.add_argument('--indices', nargs='+', type=int, default=None,
                   help="Specific row indices to render (overrides --mode and --n)")
    p.add_argument('--width',  type=int, default=400)
    p.add_argument('--height', type=int, default=300)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    for csv_path in args.csv:
        if not os.path.exists(csv_path):
            print(f"[warn] CSV not found: {csv_path}")
            continue

        print(f"\nLoading: {csv_path}")
        df = load_failure_csv(csv_path)
        print(f"  {len(df)} rows loaded")

        # Build a prefix from csv filename stem
        stem   = os.path.splitext(os.path.basename(csv_path))[0]
        subdir = os.path.join(args.out, stem)

        if args.indices is not None:
            valid_idx = [i for i in args.indices if i < len(df)]
            subset = df.iloc[valid_idx].reset_index(drop=False)
            prefix = f"{stem}_idx"
        else:
            # Sort by absolute error
            if 'true' in df.columns and 'pred' in df.columns:
                df['_abs_err'] = (df['true'] - df['pred']).abs()
            elif 'y_true' in df.columns and 'y_pred' in df.columns:
                df['_abs_err'] = (df['y_true'] - df['y_pred']).abs()
            else:
                df['_abs_err'] = 0.0

            if args.mode == 'worst':
                subset = df.nlargest(args.n, '_abs_err').reset_index(drop=False)
                prefix = f"{stem}_worst"
            elif args.mode == 'best':
                subset = df.nsmallest(args.n, '_abs_err').reset_index(drop=False)
                prefix = f"{stem}_best"
            else:  # all
                subset = df.reset_index(drop=False)
                prefix = f"{stem}_all"

        print(f"  Rendering {len(subset)} molecules → {subdir}/")
        render_rows(subset, subdir, prefix=prefix)

    print("\nDone!")


if __name__ == '__main__':
    main()
