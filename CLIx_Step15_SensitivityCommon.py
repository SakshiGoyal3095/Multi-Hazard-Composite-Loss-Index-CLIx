import pandas as pd
from scipy.stats import spearmanr

from CLIx_Step1_ZeroVarianceCheck import check_zero_variance, drop_zero_variance
from CLIx_Step2_SkewnessKurtosisCheck import apply_skew_kurtosis_treatment, drop_post_treatment_zero_variance
from CLIx_Step4_Standardisation import zscore_standardise
from CLIx_Step6_KMOTest import iterative_kmo_dropping
from CLIx_Step8_PCA import (build_pca_input_matrices, eigendecompose, compute_cumulative_variance,
                            retain_by_cumulative_variance, apply_varimax, flag_significant_loadings,
                            compute_pc_scores, build_flip_map_from_loadings, harmonize_pc_signs)
from CLIx_Step9_WithinDimensionWeights import compute_within_dimension_weights, compute_dimension_scores
from CLIx_Step10_CLIxScoreComputation import compute_clix_raw, compute_clix_rescaled, add_clix_rank

def retain_by_kaiser(pca_input):
    for dim, result in pca_input.items():
        eigenvalues = result['eigenvalues']
        n_retained = max(int((eigenvalues > 1).sum()), 1)
        result['n_retained'] = n_retained
        result['cumvar_at_retained'] = result['cumulative_variance'][n_retained - 1]
        result['retained_eigenvalues'] = eigenvalues[:n_retained]
        result['retained_eigenvectors'] = result['eigenvectors'][:, :n_retained]
    return pca_input

# Steps 1-10 without exports, so each test reruns exactly the main method
def run_core_pipeline(merged_df, dim_map, retention_method='cumulative', kmo_threshold=0.5,
                      min_indicators=3, variance_threshold=0.80, loading_threshold=0.4):
    variance_check = check_zero_variance(merged_df, dim_map)
    cleaned_df, _ = drop_zero_variance(merged_df, variance_check)
    treated_df, _ = apply_skew_kurtosis_treatment(cleaned_df, dim_map)
    treated_df, _ = drop_post_treatment_zero_variance(treated_df)
    standardized_df = zscore_standardise(treated_df)

    kmo_results = {}
    for dim in sorted(set(dim_map.values())):
        surviving_cols, final_kmo, final_msa, _ = iterative_kmo_dropping(
            standardized_df, dim_map, dim, threshold=kmo_threshold, min_indicators=min_indicators)
        kmo_results[dim] = {'surviving_indicators': surviving_cols, 'final_kmo': final_kmo, 'final_msa': final_msa}

    pca_input = build_pca_input_matrices(standardized_df, kmo_results)
    pca_input = eigendecompose(pca_input)
    pca_input = compute_cumulative_variance(pca_input)
    if retention_method == 'kaiser':
        pca_input = retain_by_kaiser(pca_input)
    else:
        pca_input = retain_by_cumulative_variance(pca_input, threshold=variance_threshold)
    pca_input = apply_varimax(pca_input)
    pca_input = flag_significant_loadings(pca_input, threshold=loading_threshold)
    pca_input = compute_within_dimension_weights(pca_input)

    pc_scores = compute_pc_scores(standardized_df, pca_input)
    flip_map = build_flip_map_from_loadings(pca_input)
    pc_scores_harmonized = harmonize_pc_signs(pc_scores, flip_map)
    dimension_scores = compute_dimension_scores(pc_scores_harmonized, pca_input)

    clix_df = compute_clix_raw(dimension_scores)
    clix_df, _, _ = compute_clix_rescaled(clix_df)
    clix_df = add_clix_rank(clix_df)
    return clix_df, pca_input, dimension_scores

# blocks matched on District + Block, since block names can repeat across districts
def rank_correlation(baseline_clix, other_clix):
    keys = ['District', 'Block']
    merged = baseline_clix[keys + ['CLIx_Rank']].merge(
        other_clix[keys + ['CLIx_Rank']], on=keys, suffixes=('_baseline', '_alternative'))
    rho, _ = spearmanr(merged['CLIx_Rank_baseline'], merged['CLIx_Rank_alternative'])
    return rho

def export_sensitivity_results(sheets, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


## Note: the four sensitivity tests (Section 3.7) all use run_core_pipeline, so they test exactly the method used for the main results (OECD, 2008; Mazziotta & Pareto, 2025).