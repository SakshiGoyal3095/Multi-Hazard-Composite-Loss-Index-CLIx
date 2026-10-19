import pandas as pd
import numpy as np

def compute_kmo(corr_matrix):
    R = corr_matrix.values
    R_inv = np.linalg.inv(R)

    diag_sqrt = np.sqrt(np.diag(R_inv))
    partial_corr = -R_inv / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(partial_corr, 0)

    r_sq = R ** 2
    np.fill_diagonal(r_sq, 0)
    p_sq = partial_corr ** 2

    overall_kmo = r_sq.sum() / (r_sq.sum() + p_sq.sum())

    msa_per_indicator = {}
    cols = corr_matrix.columns
    for i, col in enumerate(cols):
        r_sum = r_sq[i, :].sum()
        p_sum = p_sq[i, :].sum()
        msa_per_indicator[col] = r_sum / (r_sum + p_sum)

    return overall_kmo, msa_per_indicator

# drops the lowest-MSA indicator one at a time and recomputes, because removing one indicator changes every other indicator's MSA (Kaiser, 1974)
def iterative_kmo_dropping(df, dim_map, dimension, threshold=0.5, min_indicators=3):
    cols = [c for c in df.columns if dim_map.get(c) == dimension]
    dropped_log = []
    current_cols = cols.copy()

    while True:
        corr_matrix = df[current_cols].corr(method='spearman')
        overall_kmo, msa_per_indicator = compute_kmo(corr_matrix)

        min_indicator = min(msa_per_indicator, key=msa_per_indicator.get)
        min_msa = msa_per_indicator[min_indicator]

        # min_indicators stops a weak dimension from being emptied entirely
        if min_msa >= threshold or len(current_cols) <= min_indicators:
            break

        dropped_log.append({
            'Dimension': dimension,
            'Dropped_Indicator': min_indicator,
            'MSA_at_drop': min_msa,
            'Overall_KMO_before_drop': overall_kmo
        })
        current_cols.remove(min_indicator)

    final_corr_matrix = df[current_cols].corr(method='spearman')
    final_kmo, final_msa = compute_kmo(final_corr_matrix)

    return current_cols, final_kmo, final_msa, dropped_log

def build_kmo_results_table(kmo_results):
    rows = []
    for dim, result in kmo_results.items():
        for ind, msa in result['final_msa'].items():
            rows.append({
                'Dimension': dim,
                'Indicator': ind,
                'MSA': msa,
                'Dimension_Final_KMO': result['final_kmo']
            })
    return pd.DataFrame(rows).sort_values(['Dimension', 'Indicator']).reset_index(drop=True)

def export_kmo_results(kmo_results_table, dropped_df, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        kmo_results_table.to_excel(writer, sheet_name='KMO_Final_Results', index=False)
        dropped_df.to_excel(writer, sheet_name='KMO_Dropped_Log', index=False)

# every indicator dropped before PCA, in order, with the reason
def build_full_drop_summary(pipeline_label, starting_count, raw_zero_var_dropped,
                            post_treatment_zero_var_dropped, kmo_dropped_df, kmo_threshold=0.5):
    rows = []
    running_count = starting_count

    for ind in raw_zero_var_dropped:
        running_count -= 1
        rows.append({
            'Stage': '3.3.1 Zero-Variance (raw data)',
            'Indicator': ind,
            'Reason': 'Identical value across all blocks; no variance for PCA',
            'Indicators Remaining After Drop': running_count
        })

    for ind in post_treatment_zero_var_dropped:
        running_count -= 1
        rows.append({
            'Stage': '3.3.2 Post-Treatment Zero-Variance',
            'Indicator': ind,
            'Reason': 'Winsorisation collapsed indicator to a single value; no remaining variance',
            'Indicators Remaining After Drop': running_count
        })

    for _, row in kmo_dropped_df.iterrows():
        running_count -= 1
        rows.append({
            'Stage': '3.4.1 KMO Adequacy Testing',
            'Indicator': row['Dropped_Indicator'],
            'Reason': f"MSA = {row['MSA_at_drop']:.3f}, below {kmo_threshold} threshold (Dimension: {row['Dimension']})",
            'Indicators Remaining After Drop': running_count
        })

    summary_df = pd.DataFrame(rows)
    summary_df.insert(0, 'Pipeline', pipeline_label)
    summary_df.insert(1, 'Sno', range(1, len(summary_df) + 1))
    return summary_df

def export_full_drop_summary(summary_df, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        summary_df.to_excel(writer, sheet_name='Full_Drop_Summary', index=False)


# Note on the minimum-indicator floor:
# The loop stops if a dimension reaches min_indicators, even if MSA is still below the threshold. This prevents a dimension being emptied. It was not triggered in the Uttarakhand test case (smallest retained set: 4 indicators).