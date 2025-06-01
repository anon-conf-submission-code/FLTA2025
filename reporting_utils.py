import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from tabulate import tabulate

def save_logrank_results(results_df: pd.DataFrame, dataset_name: str, base_save_dir: str = '.'):
    """Saves log-rank test results to text and CSV files in a dataset-specific directory."""
    if results_df.empty:
        print(f"No log-rank results for {dataset_name} to save.")
        return

    dst = Path(base_save_dir) / dataset_name
    dst.mkdir(parents=True, exist_ok=True)

    # Save detailed results to CSV
    results_df.to_csv(dst/"logrank_results.csv", index=False)

    # Save human-readable summary to text file
    with open(dst/"logrank_results.txt", 'w') as f:
        f.write(f"Log-rank Test Results for {dataset_name}\n")
        f.write("Report generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("Methodology: Log-rank test performed per repetition between centralized data and \n")
        f.write("             combined federated surrogate data (node-specific DP curves used for surrogates).\n")
        f.write("             P-values are then summarized (mean, median, proportion significant) across repetitions.\n")
        f.write("=" * 90 + "\n\n")
        
        results_df_formatted = results_df.copy()
        
        # Formatting for display
        if 'Mean Log-rank p-value' in results_df_formatted.columns:
            results_df_formatted['Mean Log-rank p-value'] = results_df_formatted['Mean Log-rank p-value'].apply(
                lambda x: f"{x:.4e}" if pd.notnull(x) else "N/A")
        if 'Median Log-rank p-value' in results_df_formatted.columns:
            results_df_formatted['Median Log-rank p-value'] = results_df_formatted['Median Log-rank p-value'].apply(
                lambda x: f"{x:.4e}" if pd.notnull(x) else "N/A")
        if 'Proportion Significant (p<0.05)' in results_df_formatted.columns:
             results_df_formatted['Proportion Significant (p<0.05)'] = results_df_formatted['Proportion Significant (p<0.05)'].apply(
                 lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")

        f.write(tabulate(results_df_formatted, headers='keys', tablefmt='grid', showindex=False, numalign="left", stralign="left"))
        f.write("\n\n")
        
        # Overall summary: Proportion of configurations (method, epsilon) where mean p-value < 0.05, by partitioning strategy
        f.write("Summary: Proportion of (Method, Epsilon) Configurations with Mean p-value < 0.05\n")
        f.write("-" * 90 + "\n")

    print(f"✓ Log-rank results written to: {dst}/logrank_results.csv and {dst}/logrank_results.txt")

def save_timing_results(timing_results: pd.DataFrame, dataset_name: str, base_save_dir: str = '.'):
    """Saves timing results to CSV file in a dataset-specific directory."""
    if timing_results.empty:
        print(f"No timing results for {dataset_name} to save.")
        return

    dst = Path(base_save_dir) / dataset_name
    dst.mkdir(parents=True, exist_ok=True)
    timing_results.to_csv(dst/"timing_results.csv", index=False)
    print(f"✓ Timing results written to: {dst}/timing_results.csv")

def save_mae_table(mae_runs: pd.DataFrame,
                   dataset_name: str,
                   base_save_dir: str = '.'):
    """
    Builds (1) MAE ±95% CI table and (2) imbalance-penalty table,
    then saves both as CSV.
    """
    if mae_runs.empty:                                          # guard-rail
        print(f"No MAE records for {dataset_name} to generate tables.")
        return

    dst = Path(base_save_dir) / dataset_name
    dst.mkdir(parents=True, exist_ok=True)

    # --- T-1: MAE summary -----------------------------------------------
    # Ensure correct dtypes before groupby if not already done
    mae_runs['Epsilon'] = mae_runs['Epsilon'].astype(float)
    mae_runs['MAE'] = mae_runs['MAE'].astype(float)
    
    g = (mae_runs
         .groupby(["Partition","Method","Epsilon"])
         .MAE
         .agg(['mean','sem','count'])
         .reset_index())
    
    # Calculate 95% CI: mean ± 1.96 * SEM (Standard Error of the Mean)
    g["ci_lo"] = g["mean"] - 1.96*g["sem"]
    g["ci_hi"] = g["mean"] + 1.96*g["sem"]

    # Define specific columns to save for CSV to avoid issues with multi-index from agg
    t1_cols_to_report = ['Partition', 'Method', 'Epsilon', 'mean', 'sem', 'count', 'ci_lo', 'ci_hi']
    g_report = g[t1_cols_to_report].copy()

    g_report.to_csv(dst/"T1_MAE.csv", index=False)

    # --- T-3: imbalance penalty (ratio) --------------------------------
    # Use the 'mean' MAE calculated in 'g' for pivoting
    pivot_df = (g.pivot_table(index=["Method","Epsilon"],
                           columns="Partition",
                           values="mean"))
    
    # Check if all required partition types are present in the columns for ratio calculation
    required_partitions_for_ratio = {"Uniform", "Highly Imbalanced", "Non-uniform"}
    if required_partitions_for_ratio.issubset(pivot_df.columns):
        pivot_df["ΔMAE_60-20-20"] = pivot_df["Non-uniform"] / pivot_df["Uniform"]
        pivot_df["ΔMAE_90-5-5"]   = pivot_df["Highly Imbalanced"] / pivot_df["Uniform"]
    else:
        print(f"Skipping imbalance penalty calculation for {dataset_name}: Not all required partition types (Uniform, Non-uniform, Highly Imbalanced) found in MAE results for T3.")

    pivot_df_report = pivot_df.reset_index()
    pivot_df_report.to_csv(dst/"T3_ImbalancePenalty.csv", index=False)
        
    # --- NEW: condensed imbalance summaries ----------------------------
    _write_condensed_imbalance_tables(pivot_df, dst) 

    print(f"✓ MAE tables (T1_MAE, T3_ImbalancePenalty, T3b, T3c) written to: {dst}")

# ── NEW helper ────────────────────────────────────────────────────────────
def _write_condensed_imbalance_tables(pivot_df: pd.DataFrame,
                                      dst: Path, # Uses global Path type hint
                                      eps_subset: tuple[float, float] = (0.5, 2.0)
                                     ) -> None:
    """
    1) Worst–case Δ-MAE over all ε  (Table: T3b_ImbalanceWorst)
    2) Δ-MAE at two illustrative ε  (Table:  T3c_ImbalanceSubset)

    Parameters
    ----------
    pivot_df  : output of the main imbalance-penalty pivot
                (index=['Method','Epsilon'], columns include the two Δ columns)
    dst       : directory to write .csv
    eps_subset: pick two ε values to keep verbatim (edit as you like)
    """
    # -------- 1) Worst-case across ε  ------------------------------------
    # Ensure required columns are present before attempting to access them
    required_delta_cols = ['ΔMAE_60-20-20', 'ΔMAE_90-5-5']
    if not all(col in pivot_df.columns for col in required_delta_cols):
        print(f"Skipping T3b/T3c generation: Missing one or more columns ({required_delta_cols}) in pivot_df for imbalance summaries.")
        return
        
    worst = (pivot_df[required_delta_cols]
             .groupby(level='Method') # Assumes 'Method' is in the index
             .max()
             .reset_index()
             .rename(columns={'ΔMAE_60-20-20':'Δ60-20-20',
                              'ΔMAE_90-5-5'  :'Δ90-5-5'}))

    worst.to_csv(dst / "T3b_ImbalanceWorst.csv", index=False)

    # -------- 2) Picked ε subset  ---------------------------------------
    sub = (pivot_df
           .reset_index()                    # have Method, Epsilon as columns
           .query("Epsilon in @eps_subset")  # keep only the selected budgets
           .loc[:, ['Method','Epsilon'] + required_delta_cols] # Select Method, Epsilon and the delta columns
           .rename(columns={'ΔMAE_60-20-20':'Δ60-20-20',
                            'ΔMAE_90-5-5'  :'Δ90-5-5'})
           .sort_values(['Method','Epsilon']))

    sub.to_csv(dst / "T3c_ImbalanceSubset.csv", index=False)

def save_logrank_fp_rate_table(logrank_results: pd.DataFrame,
                                 dataset_name: str,
                                 base_save_dir: str = '.'):
    """
    Builds a Log-Rank False Positive (FP) Rate table (Type I error rate)
    and saves as CSV.
    FP rate is the proportion of repetitions where p < 0.05 (significant)
    when comparing DP-surrogate to true centralized data (H0: they are similar).
    """
    if logrank_results.empty:
        print(f"No log-rank results for {dataset_name} to generate FP rate table.")
        return

    dst = Path(base_save_dir) / dataset_name
    dst.mkdir(parents=True, exist_ok=True) 

    # --- T2: Log-Rank False Positive Rate --- 
    # The 'Proportion Significant (p<0.05)' column directly gives the FP rate for this context.
    # We just need to pivot and format it.
    fp_rate_table = logrank_results.pivot_table(
        index=['DP Method', 'Epsilon'],
        columns='Partitioning Strategy',
        values='Proportion Significant (p<0.05)'
    )

    if fp_rate_table.empty:
        print(f"Could not generate T2_LogRank_FP_Rate table for {dataset_name} (pivot resulted in empty table).")
        return

    fp_rate_table_report = fp_rate_table.reset_index()
    fp_rate_table_report.to_csv(dst/"T2_LogRank_FP_Rate.csv", index=False)

    print(f"✓ Log-Rank FP Rate table (T2_LogRank_FP_Rate) written to: {dst}")

def save_method_ranking_tables(mae_runs: pd.DataFrame,
                               dataset_name: str,
                               base_save_dir: str = '.'):
    """
    Builds two compact method-ranking tables from `mae_runs`
        • T4_BestEps         – best ε for each (Partition, Method)
        • T5_AvgRank         – average rank of every Method
    Saves as CSV.
    """
    if mae_runs.empty:
        print(f"No MAE records for {dataset_name}; ranking tables skipped.")
        return

    # ----------- tidy up ---------------------------------------------------
    mae_runs = mae_runs.copy()
    mae_runs['Epsilon'] = mae_runs['Epsilon'].astype(float)
    mae_runs['MAE']     = mae_runs['MAE'].astype(float)

    dst = Path(base_save_dir) / dataset_name
    dst.mkdir(parents=True, exist_ok=True)

    # ----------- table T4: best ε per method × partition -------------------
    g = (mae_runs
         .groupby(['Partition','Method','Epsilon'])
         .MAE.mean()
         .reset_index())

    idx = (g
           .sort_values(['Partition','Method','MAE'])  # smallest MAE first
           .groupby(['Partition','Method'])
           .head(1))                                   # keep the winner

    t4 = idx.rename(columns={'Epsilon':'Best ε',
                             'MAE'    :'MAE at ε*'})

    t4.to_csv(dst/"T4_BestEps.csv", index=False)

    # ----------- table T5: average rank across everything ------------------
    # rank 1 = lowest MAE within a given (Partition, ε) combination
    g['Rank'] = g.groupby(['Partition','Epsilon'])['MAE'] \
                 .rank(method='min', ascending=True)

    t5 = (g.groupby('Method')
            .Rank.mean()
            .sort_values()
            .reset_index()
            .rename(columns={'Rank':'Avg Rank'}))

    t5.to_csv(dst/"T5_AvgRank.csv", index=False)

    print(f"✓ Method-ranking tables (T4_BestEps, T5_AvgRank) saved to {dst}") 