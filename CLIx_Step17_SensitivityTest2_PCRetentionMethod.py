import pandas as pd

from CLIx_Step15_SensitivityCommon import run_core_pipeline, rank_correlation

def retention_method_test(merged_df, dim_map, baseline_clix, baseline_pca_input, rho_threshold=0.80,
                          **pipeline_settings):
    clix_kaiser, pca_kaiser, _ = run_core_pipeline(merged_df, dim_map, retention_method='kaiser', **pipeline_settings)
    rho = rank_correlation(baseline_clix, clix_kaiser)

    rows = []
    for dim in sorted(baseline_pca_input.keys()):
        rows.append({
            'Dimension': dim,
            'Components_80pct_CumVar': baseline_pca_input[dim]['n_retained'],
            'CumVar_at_80pct': round(baseline_pca_input[dim]['cumvar_at_retained'], 4),
            'Components_Kaiser': pca_kaiser[dim]['n_retained'],
            'CumVar_at_Kaiser': round(pca_kaiser[dim]['cumvar_at_retained'], 4),
            'Total_Components_Available': len(baseline_pca_input[dim]['eigenvalues']),
            'Spearman_rho_overall': round(rho, 4),
            f'Robust_rho_above_{rho_threshold}': 'Yes' if rho >= rho_threshold else 'No'
        })
    return pd.DataFrame(rows), rho


## Note (Test 2, component retention): the Kaiser criterion (eigenvalue > 1) replaces 80% cumulative variance in Step 8. rho above 0.80 means the retention rule does not substantially change the rankings.