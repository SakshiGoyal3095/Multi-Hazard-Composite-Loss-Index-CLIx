import os
import pandas as pd
import matplotlib.pyplot as plt

# 3xIQR targets extreme values only; quartiles come from non-zero values because zero-inflated indicators otherwise collapse Q1 = Q3 and flag every non-zero block (Yang et al., 2011; Templ et al., 2020)

def flag_outliers(df, dim_map, threshold=3):
    indicator_cols = [c for c in df.columns if c not in ('District', 'Block')]
    flags = pd.DataFrame(index=df.index)
    flags['District'] = df['District']
    flags['Block'] = df['Block']

    summary_rows = []
    for col in indicator_cols:
        nonzero_mask = df[col] != 0
        nonzero_values = df.loc[nonzero_mask, col]

        if len(nonzero_values) < 4:
            summary_rows.append({
                'Indicator': col, 'Dimension': dim_map[col],
                'Q1': None, 'Q3': None, 'IQR': None,
                'Lower_Bound': None, 'Upper_Bound': None,
                'Outlier_Count': 0, 'Outlier_Blocks': 'Not evaluable (too few non-zero values)'
            })
            flags[col] = False
            continue

        q1 = nonzero_values.quantile(0.25)
        q3 = nonzero_values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        is_outlier = pd.Series(False, index=df.index)
        is_outlier[nonzero_mask] = (nonzero_values < lower_bound) | (nonzero_values > upper_bound)
        flags[col] = is_outlier

        summary_rows.append({
            'Indicator': col, 'Dimension': dim_map[col],
            'Q1': q1, 'Q3': q3, 'IQR': iqr,
            'Lower_Bound': lower_bound, 'Upper_Bound': upper_bound,
            'Outlier_Count': is_outlier.sum(),
            'Outlier_Blocks': ', '.join(df.loc[is_outlier, 'Block'].tolist()) if is_outlier.sum() > 0 else 'None'
        })

    outlier_summary = pd.DataFrame(summary_rows).sort_values(['Dimension', 'Indicator']).reset_index(drop=True)
    return flags, outlier_summary


def export_outlier_step(outlier_summary, outlier_flags, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        outlier_summary.to_excel(writer, sheet_name='Outlier_Summary', index=False)
        outlier_flags.to_excel(writer, sheet_name='Outlier_Flags', index=False)

# Plotting figures
# boxes use non-zero values only and whiskers at threshold x IQR, so the red points are exactly the blocks flagged by flag_outliers. Each indicator is scaled by its own mean and standard deviation so all fit one axis; the scaling is linear, so the outliers do not change.

def plot_outlier_boxplots(df, dim_map, dimension, pipeline_label, output_folder, threshold=3):
    cols = [c for c in df.columns if dim_map.get(c) == dimension]
    plotted_cols, data = [], []
    for col in cols:
        nonzero = df.loc[df[col] != 0, col]
        if len(nonzero) < 4:
            continue
        plotted_cols.append(col)
        data.append(((nonzero - df[col].mean()) / df[col].std()).values)
    if not plotted_cols:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(plotted_cols) * 0.5), 6))
    ax.boxplot(data, whis=threshold, flierprops=dict(marker='o', markerfacecolor='red',
                                                     markeredgecolor='red', markersize=4))
    ax.set_xticks(range(1, len(plotted_cols) + 1))
    ax.set_xticklabels(plotted_cols, rotation=90)
    ax.set_ylabel('Scaled value (non-zero blocks only)')
    ax.set_title(f'{pipeline_label} | {dimension} — Treated Indicators, Outliers at {threshold}xIQR (non-zero values)')
    skipped = len(cols) - len(plotted_cols)
    if skipped:
        fig.text(0.01, 0.01, f'{skipped} indicator(s) not shown: fewer than 4 non-zero values.',
                 fontsize=8, style='italic')
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f'{pipeline_label}_{dimension}_Outlier_Boxplot.png'), dpi=150)
    plt.close(fig)
    
        
# Note on flagged blocks:
# Flagged blocks are retained, not removed or modified. In disaster loss indices, extreme values often reflect real geographic differences between administrative units (Cutter et al., 2003; Schmidtlein et al., 2008). This step only reports them, so the data passed to standardisation is the treated data from Step 2.