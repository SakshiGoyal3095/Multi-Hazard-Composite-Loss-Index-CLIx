import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# equal dimension weights: the mean of the DS columns equals 0.25 x each of the four dimensions, so no dimension dominates because it has more indicators or PCs
def compute_clix_raw(dimension_scores):
    merged = None
    for dim, df in dimension_scores.items():
        merged = df.copy() if merged is None else merged.merge(df, on=['District', 'Block'])

    dim_cols = [c for c in merged.columns if c.startswith('DS_')]
    merged['CLIx_raw'] = merged[dim_cols].mean(axis=1)
    return merged

# min-max rescaling to 0-100 (Mazziotta & Pareto, 2025); the result is relative to the blocks assessed, not an absolute loss measure
def compute_clix_rescaled(clix_df, decimals=3):
    clix_min = clix_df['CLIx_raw'].min()
    clix_max = clix_df['CLIx_raw'].max()
    clix_df = clix_df.copy()
    clix_df['CLIx'] = np.round(((clix_df['CLIx_raw'] - clix_min) / (clix_max - clix_min)) * 100, decimals)
    return clix_df, clix_min, clix_max

# ranked on the unrounded score so rounding cannot create false ties
def add_clix_rank(clix_df):
    clix_df = clix_df.copy()
    clix_df['CLIx_Rank'] = clix_df['CLIx_raw'].rank(ascending=False, method='min').astype(int)
    return clix_df

# gaps between consecutive ranked blocks; an unusually large gap at the top is what justified a sixth (Extreme) class in Step 11
def compute_gap_analysis(clix_df):
    clix_df_sorted = clix_df.sort_values('CLIx', ascending=False).reset_index(drop=True)
    clix_df_sorted['Gap_to_next'] = clix_df_sorted['CLIx'].diff(-1).abs()
    return clix_df_sorted

def export_clix_scores(clix_df, clix_df_sorted, output_path):
    ranked = clix_df[['District', 'Block', 'CLIx_raw', 'CLIx', 'CLIx_Rank']].sort_values('CLIx_Rank').reset_index(drop=True)
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        clix_df.to_excel(writer, sheet_name='CLIx_Final', index=False)
        ranked.to_excel(writer, sheet_name='Blocks_Ranked', index=False)
        clix_df_sorted.to_excel(writer, sheet_name='CLIx_Gap_Analysis', index=False)

# Plotting figure
def save_clix_histogram(clix_df, output_folder, filename, pipeline_label=''):
    os.makedirs(output_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clix_df['CLIx'], bins=20, edgecolor='black', color='skyblue')
    ax.axvline(clix_df['CLIx'].mean(), color='red', linestyle='--', linewidth=1.2,
               label=f"Mean = {clix_df['CLIx'].mean():.2f}")
    ax.axvline(clix_df['CLIx'].median(), color='green', linestyle='--', linewidth=1.2,
               label=f"Median = {clix_df['CLIx'].median():.2f}")
    ax.set_xlabel('CLIx (0-100)')
    ax.set_ylabel('Number of blocks')
    title_prefix = f'{pipeline_label} | ' if pipeline_label else ''
    ax.set_title(f'{title_prefix}Distribution of CLIx Scores')
    ax.legend()
    plt.tight_layout()

    fig.savefig(os.path.join(output_folder, filename), dpi=150)
    plt.close(fig)


## Notes:
# CLIx score (Section 3.6.2, Eq. 10):
# CLIx = 0.25 DS_AAL + 0.25 DS_Exp + 0.25 DS_Vul + 0.25 DS_CapDef, computed as the mean of the four dimension scores. Equal weights give each dimension the same influence, whatever its number of indicators or components.

# Rescaling (Eq. 11): CLIx scores are min-max rescaled to 0-100 within each implementation. A score shows a block's position among the blocks assessed and is a relative measure of loss potential. Scores are not comparable across separate runs or regions.

# Gap analysis: the histogram showed most blocks clustered in the middle range. Gaps between consecutive ranked blocks were therefore checked. In both implementations, the top block (Bahadrabad in CLIx-EQ, Doiwala in CLIx-FL) was separated from the next block by a gap several times larger than any other, which supports the six-class Jenks scheme in Step 11.