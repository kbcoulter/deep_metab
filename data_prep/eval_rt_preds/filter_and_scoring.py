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
    
    #
    
    # Extra shit
    parser.add_argument("--stereo_file", help="Specify the destereo file.",
                        default=None, required=True)
    parser.add_argument("--flag_stereo", action="store_true",
        help="Flag rows where is_stereo is True after."
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
    # filter status
    ambiguous_count_before_filter = 0
    unambiguous_count_before_filter = 0
    rt_pass_count = 0
    rt_fail_count = 0
    # post filter/ before scoring and destereo
    ambiguous_count_after_filter = 0 
    unambiguous_count_after_filter = 0
    new_unambiguous_from_filter_total = 0
    # rp specific variables only
    ambiguous_count_before_stereo = 0
    unambiguous_count_before_stereo = 0
    # post destereo
    ambiguous_count_post_stereo = 0
    unambiguous_count_post_stereo = 0
    # post filter/ scoring
    ambiguous_count_1 = 0 # final count of ambiguous annotations
    unambiguous_count_1 = 0

    # for scoring 
    total_ambiguous_ids = set()
    total_unambiguous_ids = set()

    # final counts
    final_unambiguous_count = 0 # unambi_after_filter + unambi_after_destereo + unambi_after_scoring

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
        if use_knowns and data_type != "rp": # ONLY USE HILIC KNOWNS!!!
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
            # add the rt differences
            df_csv = add_rt_diffs(df_csv)

            # RT filter (if enabled)
            if rt_filter_enabled and global_mu is not None and global_sigma is not None and data_type != "rp":
                # get initial counts before applying the filtering
                # BEFORE scoring counts
                before_filter_counts = df_csv["Mass Feature ID"].value_counts() #<- after destereo
                ambiguous_count_before_filter += int((before_filter_counts > 1).sum())
                unambiguous_count_before_filter += int((before_filter_counts == 1).sum())

                n_before = len(df_csv) # number of annotations before filter
                df_csv = apply_rt_filter(df_csv, global_mu, global_sigma, args.k_std)
                n_after = len(df_csv) # number of annotations after filter
                rt_pass_count += n_after
                rt_fail_count += (n_before - n_after)
                # AFTER FILTERING
                # BEFORE scoring counts
                after_filter_counts = df_csv["Mass Feature ID"].value_counts() #<- after destereo OR filter
                # sanity check on NaNs values if exist
                ambiguous_count_after_filter += int((after_filter_counts > 1).sum()) # after filtering
                unambiguous_count_after_filter += int((after_filter_counts == 1).sum()) # after filtering
                # NEW: track newly unambiguous IDs created by filtering
                ambig_before_ids = set(before_filter_counts[before_filter_counts > 1].index)
                unambig_after_ids = set(after_filter_counts[after_filter_counts == 1].index)

                new_unambig_from_filter = ambig_before_ids & unambig_after_ids
                new_unambiguous_from_filter_total += len(new_unambig_from_filter)
                # now just focus on the ambiguous ones only
                group_size = df_csv.groupby("Mass Feature ID")["Mass Feature ID"].transform("size")
                df_csv = df_csv[group_size > 1]

            #Destereoring
            # adding stereo and smiles_destereo
            # Grabbing counts first before destereo for rp data
            if data_type == "rp":
                id_counts = df_csv["Mass Feature ID"].value_counts()
                ambiguous_count_before_stereo += int((id_counts > 1).sum())
                unambiguous_count_before_stereo += int((id_counts == 1).sum())
            if args.flag_stereo:
                df_csv["File Name"] = file
                # adding stereo and smiles_destereo
                df_csv = df_csv.merge(
                    stereo_df[["File Name", "Mass Feature ID", "smiles", "smiles_destereo", "is_stereo"]],
                    on=["File Name", "Mass Feature ID", "smiles"],
                    how="left"
                )
                # group by mass feature id -> check if all smles_destereo are the same / is_stereo = True -> unambiguous annotations due to destoere
                # else: still ambiguous and will be put into the scoring framework
                group_cols = ["Mass Feature ID"]
                has_destereo = df_csv["smiles_destereo"].notna()

                nunique_destereo = (
                    df_csv.loc[has_destereo]
                    .groupby(group_cols)["smiles_destereo"]
                    .transform("nunique")
                )

                nunique_destereo = nunique_destereo.reindex(df_csv.index).fillna(0).astype(int)
                destereo_resolved = (nunique_destereo == 1)

                # get number of unambiguous
                destereo_unambi_counts = (
                    df_csv.loc[destereo_resolved, ["File Name", "Mass Feature ID"]]
                    .drop_duplicates()
                    .shape[0]
                    )
                unambiguous_count_post_stereo += (destereo_unambi_counts )#- unambiguous_count_after_filter) if data_type != "rp" else (destereo_unambi_counts - unambiguous_count_before_stereo)
                # update df csv to just the ambiguos one still
                df_csv = df_csv[~destereo_resolved]
                ambiguous_count_post_stereo += int((df_csv["Mass Feature ID"].value_counts() > 1).sum())
            
            # Ranking
            df_csv["rank_mz_err"] = df_csv.groupby("Mass Feature ID")["m/z Error Score"].rank(method="dense", ascending=True)
            df_csv["rank_entropy"] = df_csv.groupby("Mass Feature ID")["Entropy Similarity"].rank(method="dense", ascending=False)
            df_csv["rank_delta_rt"] = df_csv.groupby("Mass Feature ID")["abs_diff_rt"].rank(method="dense", ascending=True)

            # AFTER counts
            id_counts = df_csv["Mass Feature ID"].value_counts()
            #single_occurrence_ids = set(id_counts[id_counts == 1].index)

            cond = (df_csv["rank_mz_err"] == 1) & (df_csv["rank_entropy"] == 1) & (df_csv["rank_delta_rt"] == 1)
            ids_with_111 = set(cond.groupby(df_csv["Mass Feature ID"]).any().loc[lambda s: s].index)
            # tracking stereo count
            #id_is_stereo = df_csv.groupby("Mass Feature ID")["is_stereo"].any()

            unambiguous_ids = ids_with_111
            ambiguous_ids = set(id_counts[id_counts > 1].index) - ids_with_111

            total_unambiguous_ids.update(unambiguous_ids) # unmabgious count after scoring
            total_ambiguous_ids.update(ambiguous_ids) # ambiguous count after scoring

            unambiguous_count_1 += len(unambiguous_ids) # uniquely identified from scoring
            ambiguous_count_1 += len(ambiguous_ids) # final ambiguous left over

    # unambi after filter = 0 for rp
    final_unambiguous_count = unambiguous_count_post_stereo + unambiguous_count_1 
    # ---- Report ----
    if rt_filter_enabled:
        denom = rt_pass_count + rt_fail_count
        frac = (rt_fail_count / denom) if denom else 0.0
        print("\nRT FILTER SUMMARY")
        print(f"  Candidates passing RT filter: {rt_pass_count}")
        print(f"  Candidates failing RT filter: {rt_fail_count}")
        print(f"  Fraction removed: {frac:.3f}")

    if args.stats:
        report = """"""
        if rt_filter_enabled and data_type != "rp":
            # after filtering
            total_counts_bf = ambiguous_count_before_filter + unambiguous_count_before_filter
            pct_amb_bf = round(ambiguous_count_before_filter*100/total_counts_bf, 2)
            pct_umb_bf = round(unambiguous_count_before_filter*100/total_counts_bf, 2)

            # after filtering
            total_counts_af = ambiguous_count_after_filter + unambiguous_count_after_filter
            pct_amb0 = round(ambiguous_count_after_filter * 100 / total_counts_af, 2) if total_counts_af else 0
            pct_un0 = round(unambiguous_count_after_filter * 100 / total_counts_af, 2) if total_counts_af else 0

            fitler_report = f"""Before RT filtering:
Ambiguous:   {ambiguous_count_before_filter} ({pct_amb_bf}%)
Unambiguous: {unambiguous_count_before_filter} ({pct_umb_bf}%)

After RT filtering:
Ambiguous:   {ambiguous_count_after_filter} ({pct_amb0}%)
Unambiguous: {unambiguous_count_after_filter} ({pct_un0}%)

Number of unambiguous coming from previously ambiguous annotations: {new_unambiguous_from_filter_total}
"""
            report += fitler_report
        # after destereo
        if args.flag_stereo:
            # rp specific total counts
            total_counts_bd = ambiguous_count_before_stereo + unambiguous_count_before_stereo
            # after destereoring
            total_counts_ad = ambiguous_count_post_stereo + unambiguous_count_post_stereo # should literally be equal to ambiguos post filtering (HILIC ONLY)
            pct_ambd = round(ambiguous_count_post_stereo*100/total_counts_ad, 2)
            pct_unambd = round(unambiguous_count_post_stereo*100/total_counts_ad, 2)

            unambi_due_to_destreo = unambiguous_count_post_stereo-unambiguous_count_after_filter if data_type != "rp" else unambiguous_count_post_stereo-unambiguous_count_before_stereo
            destereo_report = f"""Before Destereo (Values shown for RP, 0 for HILIC since its the same as post filter)
Ambiguous:   {ambiguous_count_before_stereo} ({round(ambiguous_count_before_stereo*100/total_counts_bd, 2) if data_type != "hilic" else 0 }%)
Unambiguous: {unambiguous_count_before_stereo} ({round(unambiguous_count_before_stereo*100/total_counts_bd, 2) if data_type != "hilic" else 0 }%)

After Destereo (Following Data Takes into Account Changes from Ambigous Annotations Post Filtering):
Ambiguous:   {ambiguous_count_post_stereo} ({pct_ambd}%)
Unambiguous: {unambiguous_count_post_stereo} ({pct_unambd}%) 
Unambiguous Due to Destereo: {unambi_due_to_destreo} <- from just the Ambiguous Prop Before Destereo

"""
            report += destereo_report
        # after scoring
        total_counts_as = ambiguous_count_1 + unambiguous_count_1 # should be equal to the number of ambiguous post destereo
        pct_amb1 = round(ambiguous_count_1 * 100 / total_counts_as, 2) if total_counts_as else 0
        pct_un1 = round(unambiguous_count_1 * 100 / total_counts_as, 2) if total_counts_as else 0

        # final stats count
        diff_amb = (ambiguous_count_1 - ambiguous_count_before_filter 
                    if data_type != "rp" 
                    else ambiguous_count_1 - ambiguous_count_before_stereo)
        diff_un = (final_unambiguous_count - unambiguous_count_before_filter 
                   if data_type != "rp" 
                   else final_unambiguous_count - unambiguous_count_before_stereo)
        
        scoring_report = f"""After Scoring:
Ambiguous:   {ambiguous_count_1} ({pct_amb1}%)
Unambiguous: {unambiguous_count_1} ({pct_un1}%)

Final Changes:
Final Number of Ambiguous: {ambiguous_count_1} ({round(ambiguous_count_1*100/total_counts_af, 2) 
                                                 if data_type != "rp" 
                                                 else round(ambiguous_count_1*100/total_counts_bd, 2)}%)
Final Number of Unambiguous Annotations: {final_unambiguous_count} ({round(final_unambiguous_count*100/total_counts_af, 2) 
                                                 if data_type != "rp" 
                                                 else round(final_unambiguous_count*100/total_counts_bd, 2)}%)
Δ Ambiguous:   {diff_amb} | % change: {round(diff_amb*100/ambiguous_count_before_filter, 2) 
                                       if data_type != "rp" 
                                       else round(diff_amb*100/ambiguous_count_before_stereo, 2)}%
Δ Unambiguous: {diff_un} | % change: {round(diff_un*100/unambiguous_count_before_filter, 2) 
                                      if data_type != "rp" 
                                      else round(diff_un*100/unambiguous_count_before_stereo, 2)}%
"""
        report += scoring_report
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
