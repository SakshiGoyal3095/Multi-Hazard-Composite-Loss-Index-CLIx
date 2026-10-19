import pandas as pd

# w_c = EVR_c / sum(EVR) over retained PCs only, so weights sum to 1 within each dimension and a PC explaining little variance cannot match one explaining most (Greco et al., 2019; OECD, 2008)
def compute_within_dimension_weights(pca_input):
    for dim, result in pca_input.items():
        evr_retained = result['explained_variance_ratio'][:result['n_retained']]
        result['within_dim_weights'] = evr_retained / evr_retained.sum()
    return pca_input

# DS_d = sum(w_c * F_c), using harmonised PC scores so every term points the same way
def compute_dimension_scores(pc_scores_harmonized, pca_input):
    dimension_scores = {}
    for dim, score_df in pc_scores_harmonized.items():
        weights = pca_input[dim]['within_dim_weights']
        pc_cols = [f'PC{i+1}' for i in range(pca_input[dim]['n_retained'])]
        result_df = score_df[['District', 'Block']].copy()
        result_df[f'DS_{dim}'] = score_df[pc_cols].values @ weights
        dimension_scores[dim] = result_df
    return dimension_scores

def merge_dimension_scores(dimension_scores):
    merged = None
    for dim, df in dimension_scores.items():
        merged = df.copy() if merged is None else merged.merge(df, on=['District', 'Block'])
    return merged

# names come from the 'PC Name' column of Retained_Loadings sheets, filled in by hand
def load_component_names(results_path, dimensions):
    component_names_dict = {}
    for dim in dimensions:
        try:
            df = pd.read_excel(results_path, sheet_name=f'Retained_Loadings_{dim}'[:31])
        except ValueError:
            continue
        for pc in df['Component'].unique():
            names = df.loc[df['Component'] == pc, 'PC Name'].dropna()
            if len(names) > 0 and str(names.iloc[0]).strip() != '':
                component_names_dict[(dim, pc)] = names.iloc[0]
    return component_names_dict

def build_within_dim_summary(pca_input, component_names_dict=None):
    component_names_dict = component_names_dict or {}
    rows = []
    for dim, result in pca_input.items():
        weights = result['within_dim_weights']
        evr = result['explained_variance_ratio'][:result['n_retained']]
        for i, (w, e) in enumerate(zip(weights, evr), start=1):
            pc_label = f'PC{i}'
            rows.append({
                'Dimension': dim,
                'Component': pc_label,
                'Component_Name': component_names_dict.get((dim, pc_label), ''),
                'Raw_EVR': e,
                'Within_Dimension_Weight': w,
                'Dimension_Weight_Sum_Check': round(weights.sum(), 4)
            })
    return pd.DataFrame(rows)

# blocks with zero on every raw indicator of a dimension sit at the same point in indicator space, so they get identical PC and dimension scores. This traces them through every stage to show the tie comes from the source data, not the code.
def build_zero_value_verification(dim, raw_df, treated_df, standardized_df,
                                  pc_scores_harmonized, dimension_scores, pca_input, dim_map):
    raw_cols = [c for c in raw_df.columns if dim_map.get(c) == dim]
    zero_mask = (raw_df[raw_cols] == 0).all(axis=1)
    tied_blocks = raw_df.loc[zero_mask, 'Block'].tolist()
    if not tied_blocks:
        return None

    surviving_cols = pca_input[dim]['surviving_indicators']
    pc_cols = [c for c in pc_scores_harmonized[dim].columns if c not in ('District', 'Block')]

    stages = {
        'RawData': raw_df.loc[raw_df['Block'].isin(tied_blocks), ['District', 'Block'] + raw_cols],
        'Treated': treated_df.loc[treated_df['Block'].isin(tied_blocks), ['District', 'Block'] + surviving_cols],
        'Standardized': standardized_df.loc[standardized_df['Block'].isin(tied_blocks), ['District', 'Block'] + surviving_cols],
        'PCScores': pc_scores_harmonized[dim].loc[pc_scores_harmonized[dim]['Block'].isin(tied_blocks), ['District', 'Block'] + pc_cols],
        'DimensionScore': dimension_scores[dim].loc[dimension_scores[dim]['Block'].isin(tied_blocks)]
    }

    summary = pd.DataFrame([{
        'Dimension': dim,
        'Number_of_tied_blocks': len(tied_blocks),
        'Total_blocks': len(raw_df),
        'Percentage': f"{len(tied_blocks) / len(raw_df) * 100:.1f}%",
        'Explanation': (f'These blocks record zero on every raw {dim} indicator. Since values are identical '
                        f'at source, treatment, standardisation, PCA, harmonisation and dimension scoring '
                        f'correctly produce identical results for them. This is a feature of the source data, '
                        f'not a computational error.'),
        'Tied_Blocks': ', '.join(tied_blocks)
    }])
    return summary, stages

def export_weights_and_dimension_scores(pca_input, dimension_scores, output_path, component_names_dict=None):
    weights_rows = []
    for dim, result in pca_input.items():
        for i, w in enumerate(result['within_dim_weights'], start=1):
            weights_rows.append({'Dimension': dim, 'Component': f'PC{i}', 'Within_Dimension_Weight': w})

    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        pd.DataFrame(weights_rows).to_excel(writer, sheet_name='Within_Dimension_Weights', index=False)
        merge_dimension_scores(dimension_scores).to_excel(writer, sheet_name='Dimension_Scores', index=False)
        build_within_dim_summary(pca_input, component_names_dict).to_excel(
            writer, sheet_name='Component_Name_Weight_Summary', index=False)

def export_zero_value_verification(dim, summary, stages, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        summary.to_excel(writer, sheet_name=f'ZeroLoss_{dim}_Summary'[:31], index=False)
        for stage_name, stage_df in stages.items():
            stage_df.to_excel(writer, sheet_name=f'ZeroLoss_{dim}_{stage_name}'[:31], index=False)


## Notes:
# Within-dimension weights (Section 3.6.1, Eq. 8): each dimension keeps several PCs, and each explains a different share of that dimension's variance. Weighting by EVR lets PC1, which captures the largest pattern, count for more than a PC capturing very little. Weights are normalised over retained PCs only, so they sum to 1 per dimension. This avoids subjective expert weighting.

# Dimension score (Eq. 9): for each block, each harmonised PC score is multiplied by its weight and summed.

# Tied blocks (verified in the test case): In CLIx-FL, 28 of 95 blocks returned the same DS_AAL value (-0.3609). This was checked step by step before accepting the result:
# 1. Harmonised AAL PC scores contained no missing (NaN) values.
# 2. At full precision, the 28 DS_AAL values were identical, not just similar after rounding.
# 3. PC1 to PC6 scores were identical across all 28 blocks.
# 4. In the raw and treated data, all AAL indicators for these blocks were exactly 0.
# Conclusion: these blocks record no flood-related loss in the source data (Uttarakhand Disaster Risk Atlas). With identical inputs, every later step correctly produces identical outputs. No computational error was found, and the tie is a genuine feature of the data. build_zero_value_verification reproduces this check for any dimension and exports one sheet per stage (raw, treated, standardised, PC scores, dimension score).