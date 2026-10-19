import pandas as pd

from CLIx_Step15_SensitivityCommon import run_core_pipeline, rank_correlation

def indicator_exclusion_test(merged_df, dim_map, baseline_clix, baseline_pca_input, **pipeline_settings):
    retained = {ind for result in baseline_pca_input.values() for ind in result['surviving_indicators']}
    all_indicators = [c for c in merged_df.columns if c not in ('District', 'Block')]

    results = []
    for indicator in all_indicators:
        reduced_df = merged_df.drop(columns=[indicator])
        reduced_map = {k: v for k, v in dim_map.items() if k != indicator}
        try:
            clix_reduced, _, _ = run_core_pipeline(reduced_df, reduced_map, **pipeline_settings)
            rho, error = rank_correlation(baseline_clix, clix_reduced), ''
        except Exception as e:
            rho, error = None, str(e)
        results.append({
            'Indicator': indicator,
            'Dimension': dim_map[indicator],
            'Retained_in_Baseline': indicator in retained,
            'Spearman_rho': rho,
            'Error': error
        })
    return pd.DataFrame(results)

# retained_only=True excludes indicators already dropped before PCA, whose removal trivially gives rho = 1 and inflates the mean
def build_exclusion_summary(exclusion_results, threshold=0.80, retained_only=False):
    data = exclusion_results[exclusion_results['Retained_in_Baseline']] if retained_only else exclusion_results
    data = data.dropna(subset=['Spearman_rho'])

    overall = pd.DataFrame([{
        'Minimum_rho': data['Spearman_rho'].min(),
        'Maximum_rho': data['Spearman_rho'].max(),
        'Mean_rho': data['Spearman_rho'].mean(),
        'Indicators_below_threshold': (data['Spearman_rho'] < threshold).sum(),
        'Total_indicators_tested': len(data),
        'Most_sensitive_indicator': data.loc[data['Spearman_rho'].idxmin(), 'Indicator'],
        'Threshold_used': threshold,
        'Retained_only': retained_only
    }])
    by_dimension = data.groupby('Dimension')['Spearman_rho'].agg(['mean', 'min', 'max']).reset_index()
    return overall, by_dimension


## Note (Test 1, indicator exclusion): each indicator is removed in turn and CLIx is rerun from Step 1. Spearman's rho between baseline and new rankings above 0.80 means no single indicator drives the ranking. Results sheets: Test1_IndicatorExclusion, Test1_IE_Summary, Test1_IE_ByDimension.