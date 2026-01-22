#!/usr/bin/env python3

import argparse
import os
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

# CLI
def get_args():
    parser = argparse.ArgumentParser(
        description="LC-MS calibration + RT filtering + scoring on a single long CSV."
    )

    # NEW: input is a single CSV, not a directory
    parser.add_argument(
        "-i", "--input", required=True,
        help="Single long CSV with columns: File Name, Mass Feature ID, Retention Time (min), smiles, is_stereo"
    )

    parser.add_argument(
        "-p", "--preds", default=None,
        help="CSV containing columns: SMILES, Predicted RT"
    )
    parser.add_argument(
        "-a", "--append", action="store_true",
        help="Append predicted RT by mapping smiles -> Predicted RT using --preds"
    )

    parser.add_argument(
        "-t", "--type", choices=["hilic", "rp"], required=True,
        help="Used for naming outputs"
    )
    parser.add_argument(
        "-s", "--stats", action="store_true",
        help="Compute and write stats report"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save outputs (delta distribution CSV, optional processed CSV)"
    )

    # RT filtering
    parser.add_argument(
        "--rt_filter", action="store_true",
        help="Enable RT plausibility filtering using global ΔRT distribution"
    )
    parser.add_argument(
        "--k_std", type=float, default=2.0,
        help="Std-dev multiplier for RT plausibility window"
    )
    parser.add_argument(
        "--min_calib_n", type=int, default=50,
        help="Minimum calibration points required to enable RT filtering"
    )

    # Known standards override calibration (your new idea)
    parser.add_argument(
        "--knowns", default=None,
        help="Optional known standards long CSV (same schema as --input). If provided with --knowns_preds, calibration uses ONLY this file."
    )
    parser.add_argument(
        "--knowns_preds", default=None,
        help="Optional preds CSV for knowns (columns: SMILES, Predicted RT)."
    )

    # Stereo removal toggle
    parser.add_argument(
        "--drop_stereo", action="store_true",
        help="Drop rows where is_stereo is True before calibration/scoring."
    )

    return parser.parse_args()

# IO helpers
def load_smiles_dict(smiles_csv: Optional[str]) -> Optional[Dict[str, float]]:
    if smiles_csv is None:
        return None
    smiles_df = pd.read_csv(smiles_csv)
    if "SMILES" not in smiles_df.columns or "Predicted RT" not in smiles_df.columns:
        raise KeyError("Preds CSV must contain columns: 'SMILES' and 'Predicted RT'")
    # If duplicates exist, last one wins; if you prefer, de-dup explicitly.
    return smiles_df.set_index("SMILES")["Predicted RT"].to_dict()


def load_long_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    # Be explicit about required cols (you said there are exactly 5)
    df = pd.read_csv(path)

    required = {"File Name", "Mass Feature ID", "Retention Time (min)", "smiles", "is_stereo"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {path}: {sorted(missing)}")

    # Drop rows without smiles (same as your original logic)
    df = df.dropna(subset=["smiles"]).copy()

    return df


def remove_stereoisomers(df: pd.DataFrame) -> pd.DataFrame:
    # Handles True/False; if there are NaNs, they will be kept
    return df[df["is_stereo"] != True].copy()


def append_predicted_rt(df: pd.DataFrame, smiles_dict: Optional[Dict[str, float]]) -> pd.DataFrame:
    if smiles_dict is None:
        raise ValueError("--append was set but preds dict is None (did you pass --preds?)")
    df = df.copy()
    df["predicted_rt"] = df["smiles"].map(smiles_dict)
    return df

# Stats Helper Functions
def add_rt_diffs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rt_obs_sec = df["Retention Time (min)"] * 60.0
    rt_pred_sec = df["predicted_rt"]
    df["diff_rt"] = rt_pred_sec - rt_obs_sec
    df["abs_diff_rt"] = df["diff_rt"].abs()
    return df


def apply_rt_filter(df: pd.DataFrame, mu: float, sigma: float, k: float) -> pd.DataFrame:
    if not np.isfinite(sigma) or sigma <= 0:
        return df
    return df[np.abs(df["diff_rt"] - mu) <= (k * sigma)].copy()


def get_initial_unambig_delta_long(df: pd.DataFrame) -> np.ndarray:
    """
    Calibration ΔRT from MF IDs that appear exactly once *within each file*,
    matching your old per-file behavior.
    """
    # group within each "file"
    grp_cols = ["File Name", "Mass Feature ID"]
    size_per_group = df.groupby(grp_cols, sort=False).size()  # Series indexed by (File Name, Mass Feature ID)
    single_groups = size_per_group[size_per_group == 1].index

    # mark rows whose (File Name, Mass Feature ID) is one of those single groups
    idx = pd.MultiIndex.from_frame(df[grp_cols])
    mask = idx.isin(single_groups)

    return df.loc[mask, "diff_rt"].dropna().to_numpy()


def compute_before_counts(df: pd.DataFrame) -> Tuple[int, int]:
    """
    BEFORE counts: ambiguous vs unambiguous group counts per file.
    Equivalent to your old behavior aggregated across files.
    """
    sizes = df.groupby(["File Name", "Mass Feature ID"], sort=False).size()
    ambiguous = int((sizes > 1).sum())
    unambiguous = int((sizes == 1).sum())
    return ambiguous, unambiguous


def score_and_classify_ids(df: pd.DataFrame) -> Tuple[int, int, set, set]:
    """
    Apply ranks per (File Name, Mass Feature ID) and compute:
    - ambiguous_id_count_after
    - unambiguous_id_count_after
    - sets of ambiguous/unambiguous IDs as (File Name, Mass Feature ID) tuples
    """
    grp_cols = ["File Name", "Mass Feature ID"]
    df = df.copy()

    gb = df.groupby(grp_cols, sort=False)

    df["rank_mz_err"] = gb["m/z Error Score"].rank(method="dense", ascending=True)
    df["rank_entropy"] = gb["Entropy Similarity"].rank(method="dense", ascending=False)
    df["rank_delta_rt"] = gb["abs_diff_rt"].rank(method="dense", ascending=True)

    sizes = gb.size()  # index is (File Name, Mass Feature ID)
    single_groups = set(sizes[sizes == 1].index)

    cond = (df["rank_mz_err"] == 1) & (df["rank_entropy"] == 1) & (df["rank_delta_rt"] == 1)
    has_111 = cond.groupby(df[grp_cols], sort=False).any()
    groups_with_111 = set(has_111[has_111].index)

    unambiguous_groups = single_groups | groups_with_111
    ambiguous_groups = set(sizes[sizes > 1].index) - groups_with_111

    return (
        len(ambiguous_groups),
        len(unambiguous_groups),
        ambiguous_groups,
        unambiguous_groups,
    )

# Calibration
def calibrate_mu_sigma(
    df_source: pd.DataFrame,
    data_type: str,
    min_calib_n: int,
    k_std: float,
    save: bool,
    out_dir: str,
    calibration_mode_label: str,
    use_all_rows: bool
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Returns: (rt_filter_enabled, mu, sigma)
    """
    if use_all_rows:
        d = df_source["diff_rt"].dropna().to_numpy()
    else:
        d = get_initial_unambig_delta_long(df_source)

    # tidy records for export
    delta_df = pd.DataFrame({
        "delta_rt_sec": d,
        "abs_delta_rt_sec": np.abs(d),
        "chrom_type": data_type,
        "source": calibration_mode_label
    })

    if save and not delta_df.empty:
        out_name = os.path.join(out_dir, f"{data_type}_rt_delta_calibration.csv")
        delta_df.to_csv(out_name, index=False)
        print(f"[INFO] Saved RT delta distribution → {out_name}")

    if d.size < min_calib_n:
        print(f"[WARN] RT filtering requested, but only {d.size} calibration points found "
              f"(min required {min_calib_n}). RT filtering will be skipped.")
        return False, None, None

    mu = float(np.mean(d))
    sigma = float(np.std(d, ddof=1))
    lo = mu - k_std * sigma
    hi = mu + k_std * sigma
    print(f"[INFO] Global ΔRT calibration (sec): n={d.size}, mu={mu:.3f}, sigma={sigma:.3f}, "
          f"window=[{lo:.3f}, {hi:.3f}]")
    return True, mu, sigma



def main(args) -> int:
    data_type = args.type
    out_dir = f"{data_type}_Calibration_&_Scoring"
    os.makedirs(out_dir, exist_ok=True)

    # Load main long CSV
    df = load_long_csv(args.input)

    if args.drop_stereo:
        df = remove_stereoisomers(df)

    # Predicted RT mapping
    smiles_dict = load_smiles_dict(args.preds)

    if args.append:
        df = append_predicted_rt(df, smiles_dict)

    if "predicted_rt" not in df.columns:
        raise KeyError("'predicted_rt' not found. Use --append with --preds or ensure it exists in the input CSV.")

    df = add_rt_diffs(df)

    # --- Calibration (optional) ---
    rt_filter_enabled = bool(args.rt_filter)
    global_mu, global_sigma = None, None

    knowns_path = args.knowns
    knowns_preds = args.knowns_preds
    use_knowns = bool(knowns_path and knowns_preds)

    if rt_filter_enabled:
        if use_knowns:
            print("[INFO] Knowns detected: calibrating mu/sigma ONLY from knowns file.")
            df_knowns = pd.read_csv(knowns_path)


            known_dict = load_smiles_dict(knowns_preds)
            df_knowns = append_predicted_rt(df_knowns, known_dict)
            df_knowns = add_rt_diffs(df_knowns)

            # For true standards, "use_all_rows=True" is usually what you want.
            rt_filter_enabled, global_mu, global_sigma = calibrate_mu_sigma(
                df_source=df_knowns,
                data_type=data_type,
                min_calib_n=args.min_calib_n,
                k_std=args.k_std,
                save=args.save,
                out_dir=out_dir,
                calibration_mode_label="knowns",
                use_all_rows=True,
            )
        else:
            print("[INFO] No knowns provided: calibrating mu/sigma from initial unambiguous IDs per file.")
            rt_filter_enabled, global_mu, global_sigma = calibrate_mu_sigma(
                df_source=df,
                data_type=data_type,
                min_calib_n=args.min_calib_n,
                k_std=args.k_std,
                save=args.save,
                out_dir=out_dir,
                calibration_mode_label="input_long_csv",
                use_all_rows=False,  # matches your old per-file calibration behavior
            )

    # --- Pass 2: stats/scoring ---
    ambiguous_count_0 = unambiguous_count_0 = 0
    ambiguous_count_1 = unambiguous_count_1 = 0
    rt_pass_count = rt_fail_count = 0
    total_ambiguous_groups = set()
    total_unambiguous_groups = set()

    if args.stats:
        # BEFORE counts
        ambiguous_count_0, unambiguous_count_0 = compute_before_counts(df)

        # Apply RT filter (row-level) before ranking
        df_scoring = df
        if rt_filter_enabled and global_mu is not None and global_sigma is not None:
            n_before = len(df_scoring)
            df_scoring = apply_rt_filter(df_scoring, global_mu, global_sigma, args.k_std)
            n_after = len(df_scoring)
            rt_pass_count = n_after
            rt_fail_count = n_before - n_after

        # AFTER classification (group-level)
        ambiguous_count_1, unambiguous_count_1, amb_groups, unamb_groups = score_and_classify_ids(df_scoring)
        total_ambiguous_groups = amb_groups
        total_unambiguous_groups = unamb_groups

        # Optional: save processed/scoring dataframe (can be large)
        # TODO: add save appended parameter later
        # if args.save:
        #     out_path = os.path.join(out_dir, f"{data_type}_processed_scoring.csv")
        #     df_scoring.to_csv(out_path, index=False)
        #     print(f"[INFO] Saved processed scoring CSV → {out_path}")

        # Report
        total_counts = ambiguous_count_0 + unambiguous_count_0
        pct_amb0 = round(ambiguous_count_0 * 100 / total_counts, 2) if total_counts else 0
        pct_un0 = round(unambiguous_count_0 * 100 / total_counts, 2) if total_counts else 0
        pct_amb1 = round(ambiguous_count_1 * 100 / total_counts, 2) if total_counts else 0
        pct_un1 = round(unambiguous_count_1 * 100 / total_counts, 2) if total_counts else 0

        diff_amb = ambiguous_count_1 - ambiguous_count_0
        diff_un = unambiguous_count_1 - unambiguous_count_0

        report = f"""Before RT filtering / scoring:
Ambiguous groups:   {ambiguous_count_0} ({pct_amb0}%)
Unambiguous groups: {unambiguous_count_0} ({pct_un0}%)

After RT filtering + scoring:
Ambiguous groups:   {ambiguous_count_1} ({pct_amb1}%)
Unambiguous groups: {unambiguous_count_1} ({pct_un1}%)

Changes:
Δ Ambiguous:   {diff_amb} | % change: {round(diff_amb*100/ambiguous_count_0, 2) if ambiguous_count_0 else 0}%
Δ Unambiguous: {diff_un} | % change: {round(diff_un*100/unambiguous_count_0, 2) if unambiguous_count_0 else 0}%
"""

        out_report = os.path.join(out_dir, f"{data_type}_stats_report.txt")
        with open(out_report, "w") as f:
            f.write(report)
        print(f"[INFO] Wrote report → {out_report}")

    # RT filter summary (still useful even if stats off, but counts only available when stats runs)
    if rt_filter_enabled and args.stats:
        denom = rt_pass_count + rt_fail_count
        frac = (rt_fail_count / denom) if denom else 0.0
        print("\nRT FILTER SUMMARY")
        print(f"  Candidates passing RT filter: {rt_pass_count}")
        print(f"  Candidates failing RT filter: {rt_fail_count}")
        print(f"  Fraction removed: {frac:.3f}")

    return 0

if __name__ == "__main__":
    args = get_args()
    raise SystemExit(main(args))
