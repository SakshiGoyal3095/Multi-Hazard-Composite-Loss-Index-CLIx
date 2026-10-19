import os
import pandas as pd
import matplotlib.pyplot as plt

# sample standard deviation (n-1), pandas default
def zscore_standardise(df):
    standardized_df = df.copy()
    indicator_cols = [c for c in df.columns if c not in ('District', 'Block')]
    for col in indicator_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        # a zero-variance column here would silently become empty after division
        if std_val == 0:
            raise ValueError(f"'{col}' has zero variance. Run drop_post_treatment_zero_variance before standardising.")
        standardized_df[col] = (df[col] - mean_val) / std_val
    return standardized_df

def export_standardization_step(standardized_df, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        standardized_df.to_excel(writer, sheet_name='Standardized_Data', index=False)

# Plotting figures
# visual check that indicators are on one scale after standardisation; outliers are identified in Step 3, not from this plot
def plot_standardised_boxplots(standardized_df, dim_map, dimension, pipeline_label, output_folder):
    cols = [c for c in standardized_df.columns if dim_map.get(c) == dimension]
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5), 6))
    standardized_df[cols].boxplot(ax=ax, rot=90)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Z-score')
    ax.set_title(f'{pipeline_label} | {dimension} — Standardised Indicators (distribution check)')
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f'{pipeline_label}_{dimension}_Standardised_Boxplot.png'), dpi=150)
    plt.close(fig)
    

# Note: standardisation is applied to the treated data (after winsorisation, log-transformation and the post-treatment zero-variance drop in Step 2), not the raw data.