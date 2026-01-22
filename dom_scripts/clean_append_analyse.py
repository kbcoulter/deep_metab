#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Append or Analyse Existing Files")
    parser.add_argument("-i", "--input", required=True,
                        help="Input Directory with LC-MS Data (Raw or Appended)")
    parser.add_argument("-p", "--preds", default=None,
                        help="CSV containing SMILES and Predicted RT")
    parser.add_argument("-a", "--append", action="store_true",
                        help="Append predicted RT (lookup by SMILES) to each file")
    parser.add_argument("-t", "--type", choices=["hilic", "rp"], required=True,
                        help="Used for naming outputs")
    parser.add_argument("-s", "--stats", action="store_true",
                        help="Perform analysis and print out stats")
    parser.add_argument("--save", action="store_true",
                        help="Save outputs")

    # RT filtering
    parser.add_argument("--rt_filter", action="store_true",
                        help="Enable RT plausibility filtering using global ΔRT distribution")
    parser.add_argument("--k_std", type=float, default=2.0,
                        help="Std-dev multiplier for RT plausibility window")
    parser.add_argument("--min_calib_n", type=int, default=50,
                        help="Minimum calibration points required to enable RT filtering")
    parser.add_argument("--knowns", help="Specify file containing knowns if exists.",
                        default=None)
    parser.add_argument("--knowns_preds", help="Specify the preds file containing knowns if exists.",
                        default=None)
    
    # Extra shit
    parser.add_argument("--stereo_file", help="Specify the destereo file.",
                        default=None, required=True)
    parser.add_argument("--drop_stereo", action="store_true",
        help="Drop rows where is_stereo is True before calibration/scoring."
    )
    

    return parser.parse_args()


def load_smiles_dict(smiles_csv: str | None) -> dict | None:
    """
    Prepare a dictionary from a loaded smiles csv file.
    
    :param smiles_csv: A CSV file containing UNIQUE smiles and their Predicted Rentention Time.
    :type smiles_csv: str | None
    :return: A dictionary with smiles as key and their predicted rt as values.
    :rtype: dict[Any, Any] | None
    """
    if smiles_csv is None:
        return None
    smiles_df = pd.read_csv(smiles_csv)
    if "SMILES" not in smiles_df.columns or "Predicted RT" not in smiles_df.columns:
        raise KeyError("Preds CSV must contain columns: 'SMILES' and 'Predicted RT'")
    return smiles_df.set_index("SMILES")["Predicted RT"].to_dict()

# add in the predicted smiles value to the data
def load_and_prepare(file_path: str, append_switch: bool, smiles_dict: dict | None) -> pd.DataFrame:
    """
    Docstring for load_and_prepare
    
    :param file_path: Description
    :type file_path: str
    :param append_switch: Description
    :type append_switch: bool
    :param smiles_dict: Description
    :type smiles_dict: dict | None
    :return: Description
    :rtype: DataFrame
    """
    df = pd.read_csv(file_path)

    if "smiles" not in df.columns:
        raise KeyError(f"'smiles' column not found in {file_path}")

    # remove rows that doesnt contain smiles
    df = df.dropna(subset=["smiles"]).copy()

    if append_switch:
        if smiles_dict is None:
            raise ValueError("--append was set but --preds was not provided.")
        df["predicted_rt"] = df["smiles"].map(smiles_dict)

    if "predicted_rt" not in df.columns:
        raise KeyError(f"'predicted_rt' not found in {file_path}. Use --append with --preds or ensure it exists.")

    return df


def add_rt_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Docstring for add_rt_diffs
    
    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: DataFrame
    """
    df = df.copy()
    rt_obs_sec = df["Retention Time (min)"] * 60.0
    rt_pred_sec = df["predicted_rt"]
    df["diff_rt"] = rt_pred_sec - rt_obs_sec
    df["abs_diff_rt"] = df["diff_rt"].abs()
    return df

# get the initial hits of unambiguous
def get_initial_unambig_delta(df: pd.DataFrame) -> np.ndarray:
    """
    Docstring for get_initial_unambig_delta
    
    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: ndarray[Any, Any]
    """
    counts = df["Mass Feature ID"].value_counts()
    single_ids = counts[counts == 1].index
    calib = df[df["Mass Feature ID"].isin(single_ids)]
    return calib["diff_rt"].dropna().to_numpy()

# remove stereoisomers from the data
def remove_stereoisomers(df : pd.DataFrame):
    df = df.copy()
    df = df[df["is_stereo"] != True]
    return df

# flag the metabolites
def apply_rt_filter(df: pd.DataFrame, mu: float, sigma: float, k: float) -> pd.DataFrame:
    if not np.isfinite(sigma) or sigma <= 0:
        return df
    return df[np.abs(df["diff_rt"] - mu) <= (k * sigma)].copy()


def list_files(dir_path: str) -> list[str]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory '{dir_path}' not found.")
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


def main(args) -> int:
    dir_path = args.input
    data_type = args.type

    smiles_dict = load_smiles_dict(args.preds)

    # state (kept local, no globals)
    # post filter/ before scoring
    ambiguous_count_0 = 0 
    unambiguous_count_0 = 0
    # post filter/ scoring
    ambiguous_count_1 = 0
    unambiguous_count_1 = 0
    # filter status
    ambiguous_count_before_filter = 0
    unambiguous_count_before_filter = 0
    rt_pass_count = 0
    rt_fail_count = 0

    total_ambiguous_ids = set()
    total_unambiguous_ids = set()

    # destereo
    counts_w_stereo = 0
    counts_a_stereo = 0

    print("Processing files...")
    files = list_files(dir_path)
    print(f"Found {len(files)} files.")

    # Drop Stereos
    stereo_df = pd.read_csv(args.stereo_file)

    # ---- Pass 1: calibration  ----
    global_mu, global_sigma = None, None
    knowns_path = args.knowns
    knowns_smiles = args.knowns_preds
    use_knowns = bool(knowns_path and knowns_smiles) # cali source
    rt_filter_enabled = bool(args.rt_filter)

    all_delta = []
    all_delta_records = []

    if rt_filter_enabled:
        if use_knowns:
            # build dict from knowns predictions
            print("[INFO] Knowns Detected! Pass 1/2: Building Calibration of ΔRT distribution")
            known_dict = load_smiles_dict(knowns_smiles)

            # load knowns file and compute diffs
            df_csv = load_and_prepare(knowns_path, append_switch=True, smiles_dict=known_dict)

            df_csv = add_rt_diffs(df_csv)

            d = df_csv["diff_rt"].dropna().to_numpy()

            if d.size:
                all_delta.append(d)
                all_delta_records.append(pd.DataFrame({
                    "delta_rt_sec": d,
                    "abs_delta_rt_sec": np.abs(d),
                    "chrom_type": data_type,
                    "source": "knowns"
                }))

        else:
            for file in tqdm(files, desc="Pass 1/2: Building global ΔRT distribution"):
                file_path = os.path.join(dir_path, file)
                df_csv = load_and_prepare(file_path, append_switch=args.append, smiles_dict=smiles_dict)
                # Drop stereo
                if args.drop_stereo:
                    # track file name
                    df_csv["File Name"] = file

                    # left-merge to flag stereo rows
                    df_csv = df_csv.merge(
                        stereo_df.assign(_drop_stereo=True),
                        on=["File Name", "smiles"],
                        how="left"
                    )

                    # drop stereo matches
                    df_csv = df_csv[df_csv["_drop_stereo"].isna()].drop(columns="_drop_stereo")
                    df_csv = add_rt_diffs(df_csv)

                d = get_initial_unambig_delta(df_csv)  # your original strategy
                if d.size:
                    all_delta.append(d)
                    all_delta_records.append(pd.DataFrame({
                        "delta_rt_sec": d,
                        "abs_delta_rt_sec": np.abs(d),
                        "chrom_type": data_type,
                        "source": file
                    }))

        # Consolidate
        all_delta = np.concatenate(all_delta) if all_delta else np.array([]) # empty if nth
        delta_df = (
            pd.concat(all_delta_records, ignore_index=True)
            if all_delta_records
            else pd.DataFrame(columns=["delta_rt_sec", "abs_delta_rt_sec", "chrom_type", "source"])
        )

        # Export distribution
        if args.save and not delta_df.empty:
            out_name = f"{data_type}_rt_delta_calibration.csv"
            delta_df.to_csv(out_name, index=False)
            print(f"[INFO] Saved RT delta distribution → {out_name}")

        # Enable/disable filtering based on calib count
        if all_delta.size < args.min_calib_n:
            print(f"[WARN] RT filtering requested, but only {all_delta.size} calibration points found "
                f"(min required {args.min_calib_n}). RT filtering will be skipped.")
            rt_filter_enabled = False
        else:
            global_mu = float(np.mean(all_delta))
            global_sigma = float(np.std(all_delta, ddof=1))
            lo = global_mu - args.k_std * global_sigma
            hi = global_mu + args.k_std * global_sigma
            print(f"[INFO] Global ΔRT calibration (sec): n={all_delta.size}, mu={global_mu:.3f}, sigma={global_sigma:.3f}, "
                f"window=[{lo:.3f}, {hi:.3f}]")
    # ---- Pass 2: stats/scoring ----
    if args.stats:
        for file in tqdm(files, desc="Pass 2/2: Processing & scoring"):
            file_path = os.path.join(dir_path, file)
            df_csv = load_and_prepare(file_path, args.append, smiles_dict)
            # Drop stereo
            if args.drop_stereo:
                df_csv["File Name"] = file
                counts_w_stereo += df_csv["Mass Feature ID"].nunique()
                stereo_groups = (
                    stereo_df
                    .groupby(["File Name", "Mass Feature ID"])["is_stereo"]
                    .any()
                    .reset_index()
                )
                stereo_groups = stereo_groups[stereo_groups["is_stereo"]]

                drop_idx = pd.MultiIndex.from_frame(stereo_groups[["File Name", "Mass Feature ID"]])
                df_idx = pd.MultiIndex.from_frame(df_csv[["File Name", "Mass Feature ID"]])
                # drop stereo matches
                df_csv = df_csv[~df_idx.isin(drop_idx)]

                counts_a_stereo += df_csv["Mass Feature ID"].nunique()

            # add the rt differences
            df_csv = add_rt_diffs(df_csv)

            # get initial counts before applying the filtering
            # BEFORE scoring counts
            before_filter_counts = df_csv["Mass Feature ID"].value_counts() #<- after destereo
            ambiguous_count_before_filter += int((before_filter_counts > 1).sum())
            unambiguous_count_before_filter += int((before_filter_counts == 1).sum())

            # RT filter (if enabled)
            if rt_filter_enabled and global_mu is not None and global_sigma is not None:
                n_before = len(df_csv) # len before filter
                df_csv = apply_rt_filter(df_csv, global_mu, global_sigma, args.k_std)
                n_after = len(df_csv) # len after filter
                rt_pass_count += n_after
                rt_fail_count += (n_before - n_after)

            # BEFORE scoring counts
            counts = df_csv["Mass Feature ID"].value_counts() #<- after destereo OR filter
            # sanity check on NaNs values if exist
            ambiguous_count_0 += int((counts > 1).sum())
            unambiguous_count_0 += int((counts == 1).sum())

            # Ranking
            df_csv["rank_mz_err"] = df_csv.groupby("Mass Feature ID")["m/z Error Score"].rank(method="dense", ascending=True)
            df_csv["rank_entropy"] = df_csv.groupby("Mass Feature ID")["Entropy Similarity"].rank(method="dense", ascending=False)
            df_csv["rank_delta_rt"] = df_csv.groupby("Mass Feature ID")["abs_diff_rt"].rank(method="dense", ascending=True)

            # AFTER counts
            id_counts = df_csv["Mass Feature ID"].value_counts()
            single_occurrence_ids = set(id_counts[id_counts == 1].index)

            cond = (df_csv["rank_mz_err"] == 1) & (df_csv["rank_entropy"] == 1) & (df_csv["rank_delta_rt"] == 1)
            ids_with_111 = set(cond.groupby(df_csv["Mass Feature ID"]).any().loc[lambda s: s].index)

            unambiguous_ids = single_occurrence_ids | ids_with_111
            ambiguous_ids = set(id_counts[id_counts > 1].index) - ids_with_111

            total_unambiguous_ids.update(unambiguous_ids)
            total_ambiguous_ids.update(ambiguous_ids)

            unambiguous_count_1 += len(unambiguous_ids)
            ambiguous_count_1 += len(ambiguous_ids)

    # ---- Report ----
    if rt_filter_enabled:
        denom = rt_pass_count + rt_fail_count
        frac = (rt_fail_count / denom) if denom else 0.0
        print("\nRT FILTER SUMMARY")
        print(f"  Candidates passing RT filter: {rt_pass_count}")
        print(f"  Candidates failing RT filter: {rt_fail_count}")
        print(f"  Fraction removed: {frac:.3f}")

    if args.stats:
        total_counts_bf = ambiguous_count_before_filter + unambiguous_count_before_filter
        pct_amb_bf = round(ambiguous_count_before_filter*100/total_counts_bf, 2)
        pct_umb_bf = round(unambiguous_count_before_filter*100/total_counts_bf, 2)

        total_counts = ambiguous_count_0 + unambiguous_count_0
        pct_amb0 = round(ambiguous_count_0 * 100 / total_counts, 2) if total_counts else 0
        pct_un0 = round(unambiguous_count_0 * 100 / total_counts, 2) if total_counts else 0
        pct_amb1 = round(ambiguous_count_1 * 100 / total_counts, 2) if total_counts else 0
        pct_un1 = round(unambiguous_count_1 * 100 / total_counts, 2) if total_counts else 0

        diff_amb = ambiguous_count_1 - ambiguous_count_0
        diff_un = unambiguous_count_1 - unambiguous_count_0

        report = f"""Before RT filtering / scoring:
Ambiguous:   {ambiguous_count_before_filter} ({pct_amb_bf}%)
Unambiguous: {unambiguous_count_before_filter} ({pct_umb_bf}%)

After RT filtering:
Ambiguous:   {ambiguous_count_0} ({pct_amb0}%)
Unambiguous: {unambiguous_count_0} ({pct_un0}%)

After RT filtering + scoring:
Ambiguous:   {ambiguous_count_1} ({pct_amb1}%)
Unambiguous: {unambiguous_count_1} ({pct_un1}%)

Changes:
Δ Ambiguous:   {diff_amb} | % change: {round(diff_amb*100/ambiguous_count_0, 2) if ambiguous_count_0 else 0}%
Δ Unambiguous: {diff_un} | % change: {round(diff_un*100/unambiguous_count_0, 2) if unambiguous_count_0 else 0}%
"""
        if args.drop_stereo:
            pct_destereo = round(counts_a_stereo * 100 / counts_w_stereo, 2)
            pct_stere = round((counts_w_stereo - counts_a_stereo) * 100 / counts_w_stereo, 2)
            stereo_rpt = f"""Destereo Summary:
Before Destereoisomerizing: {counts_w_stereo}
Ambiguous Due to Stereoisomers: {counts_w_stereo - counts_a_stereo} MF IDs ({pct_stere}%)
Number of Destereo: {counts_a_stereo} MF IDS ({pct_destereo}%)
"""

            report = stereo_rpt + report
        out = f"{data_type}_stats_report.txt"
        out_dir = f'{data_type}_Calibration_&_Scoring'
        os.makedirs(f'{data_type}_Calibration_&_Scoring', exist_ok=True)
        with open(f'{out_dir}/{out}', "w") as f:
            f.write(report)
        print(f"[INFO] Wrote report to {out_dir}/{out}")

    return 0


if __name__ == "__main__":
    args = get_args()
    raise SystemExit(main(args))
