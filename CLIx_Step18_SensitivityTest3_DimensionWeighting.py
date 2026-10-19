import pandas as pd

from CLIx_Step10_CLIxScoreComputation import compute_clix_raw, compute_clix_rescaled, add_clix_rank
from CLIx_Step15_SensitivityCommon import rank_correlation

# each dimension weighted by its retained variance; used only as a comparison
def compute_evr_based_dimension_weights(pca_input):
    total_variance = {dim: result['retained_eigenvalues'].sum() for dim, result in pca_input.items()}
    grand_total = sum(total_variance.values())
    return {dim: v / grand_total for dim, v in total_variance.items()}

def compute_clix_with_custom_weights(dimension_scores, dim_weights):
    clix_df = compute_clix_raw(dimension_scores)
    clix_df['CLIx_raw'] = sum(clix_df[f'DS_{dim}'] * w for dim, w in dim_weights.items())
    clix_df, _, _ = compute_clix_rescaled(clix_df)
    return add_clix_rank(clix_df)

def dimension_weighting_test(baseline_clix, baseline_pca_input, baseline_dimension_scores):
    evr_weights = compute_evr_based_dimension_weights(baseline_pca_input)
    clix_weighted = compute_clix_with_custom_weights(baseline_dimension_scores, evr_weights)
    rho = rank_correlation(baseline_clix, clix_weighted)

    equal_weight = 1 / len(evr_weights)
    weights_df = pd.DataFrame([{'Dimension': dim, 'Equal_Weight': equal_weight, 'EVR_Based_Weight': w}
                               for dim, w in evr_weights.items()])
    weights_df['Spearman_rho_overall'] = rho
    return weights_df, rho


## Note (Test 3, dimension weighting): EVR-based dimension weights replace equal weights. rho above 0.80 means equal weighting does not substantially change the rankings. EVR-based weights are a comparison only: eigenvalues from separate correlation matrices are not directly comparable across dimensions, which is why the main index uses equal weights.