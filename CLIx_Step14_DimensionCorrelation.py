import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

DIMENSION_SCORE_COLUMNS = ['DS_AAL', 'DS_Exposure', 'DS_Vulnerability', 'DS_Capacity_Deficit']
DIMENSION_LABELS = ['AAL', 'Exposure', 'Vulnerability', 'Capacity Deficit']

def compute_dimension_correlation(clix_df, dim_cols=DIMENSION_SCORE_COLUMNS):
    n = len(dim_cols)
    rho_matrix = np.ones((n, n))
    pval_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rho, pval = stats.spearmanr(clix_df[dim_cols[i]], clix_df[dim_cols[j]])
                rho_matrix[i, j] = rho
                pval_matrix[i, j] = pval
    return rho_matrix, pval_matrix

def significance_stars(pval):
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05:
        return '*'
    return ' (ns)'

def export_dimension_correlation(rho_matrix, pval_matrix, output_path, labels=DIMENSION_LABELS):
    rho_df = pd.DataFrame(rho_matrix, index=labels, columns=labels)
    pval_df = pd.DataFrame(pval_matrix, index=labels, columns=labels)

    annotated = rho_df.copy().astype(object)
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotated.iloc[i, j] = '1.00' if i == j else f'{rho_matrix[i, j]:.2f}{significance_stars(pval_matrix[i, j])}'

    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        rho_df.to_excel(writer, sheet_name='Dim_Spearman_Rho')
        pval_df.to_excel(writer, sheet_name='Dim_Spearman_Pvalues')
        annotated.to_excel(writer, sheet_name='Dim_Spearman_Corr')


# Plotting figure
def plot_dimension_correlation(rho_matrix, pval_matrix, pipeline_label, output_folder, n_blocks, dpi=300):
    tick_labels = ['AAL', 'Exposure', 'Vulnerability', 'Capacity\nDeficit']
    n = len(tick_labels)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cmap = mcolors.LinearSegmentedColormap.from_list('diverging', ['#2166AC', '#F7F7F7', '#D62728'], N=256)
    im = ax.imshow(rho_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto', interpolation='nearest')

    for i in range(n):
        for j in range(n):
            rho_val = rho_matrix[i, j]
            pval = pval_matrix[i, j]
            if i == j:
                annotation = '1.00'
            elif pval < 0.05:
                annotation = f'{rho_val:.2f}{significance_stars(pval)}'
            else:
                annotation = f'{rho_val:.2f}\nns'
            text_colour = 'white' if abs(rho_val) > 0.55 else 'black'
            ax.text(j, i, annotation, ha='center', va='center', fontsize=9, color=text_colour, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label('Spearman ρ', fontsize=9)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    fig.text(0.01, 0.01, '* p < 0.05   ** p < 0.01   *** p < 0.001   ns = not significant',
             fontsize=7, color='#555555', va='bottom')
    ax.set_title(f'{pipeline_label}: Spearman Rank Correlation\nAmong Dimension Scores (n = {n_blocks} blocks)',
                 fontsize=10, pad=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f'{pipeline_label}_SpearmanCorr.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


## Notes:
# Purpose: checks how strongly the four dimension scores move together across blocks. Low or non-significant correlations show that no single dimension can stand in for the composite score, which supports combining all four into CLIx.

# Method: Spearman rank correlation between each pair of dimension scores, with p-values. Spearman is used for consistency with the rest of the pipeline and because dimension scores are not assumed to be normally distributed.

# Interpretation: results differ by hazard and are discussed in the paper (Section 5), not in code.