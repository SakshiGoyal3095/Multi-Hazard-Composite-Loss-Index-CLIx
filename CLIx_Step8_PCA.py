import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def build_pca_input_matrices(df, kmo_results):
    pca_input = {}
    for dim, result in kmo_results.items():
        surviving_cols = result['surviving_indicators']
        corr_matrix = df[surviving_cols].corr(method='spearman')
        pca_input[dim] = {
            'surviving_indicators': surviving_cols,
            'correlation_matrix': corr_matrix
        }
    return pca_input

def eigendecompose(pca_input):
    for dim, result in pca_input.items():
        R = result['correlation_matrix'].values
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        # eigh returns ascending order; PCA convention is descending
        sort_idx = np.argsort(eigenvalues)[::-1]
        result['eigenvalues'] = eigenvalues[sort_idx]
        result['eigenvectors'] = eigenvectors[:, sort_idx]
    return pca_input

def compute_cumulative_variance(pca_input):
    for dim, result in pca_input.items():
        all_eigenvalues = result['eigenvalues']
        explained_variance_ratio = all_eigenvalues / all_eigenvalues.sum()
        result['explained_variance_ratio'] = explained_variance_ratio
        result['cumulative_variance'] = np.cumsum(explained_variance_ratio)
    return pca_input

# 80% cumulative variance chosen over Kaiser (eigenvalue > 1), which covered only 60-70% of variance in the test case (OECD, 2008)
def retain_by_cumulative_variance(pca_input, threshold=0.80):
    for dim, result in pca_input.items():
        cumulative = result['cumulative_variance']
        n_retained = int(np.searchsorted(cumulative, threshold) + 1)
        n_retained = min(n_retained, len(cumulative))

        result['n_retained'] = n_retained
        result['cumvar_at_retained'] = cumulative[n_retained - 1]
        result['retained_eigenvalues'] = result['eigenvalues'][:n_retained]
        result['retained_eigenvectors'] = result['eigenvectors'][:, :n_retained]
    return pca_input

# Kaiser normalisation stops indicators with high communality from dominating the rotation (Kaiser, 1958)
def varimax_rotation(loadings, max_iter=500, tol=1e-6, kaiser_normalize=True):
    n_indicators, n_components = loadings.shape
    if n_components < 2:
        return loadings

    if kaiser_normalize:
        communalities = np.sqrt((loadings ** 2).sum(axis=1))
        communalities[communalities == 0] = 1
        normalized = loadings / communalities[:, np.newaxis]
    else:
        normalized = loadings.copy()

    rotated = normalized.copy()
    for _ in range(max_iter):
        old_rotated = rotated.copy()
        for i in range(n_components):
            for j in range(i + 1, n_components):
                x = rotated[:, i]
                y = rotated[:, j]
                u = x**2 - y**2
                v = 2 * x * y
                num = 2 * (u @ v - (u.sum() * v.sum()) / n_indicators)
                den = (u @ u - v @ v) - ((u.sum()**2 - v.sum()**2) / n_indicators)
                angle = 0.25 * np.arctan2(num, den)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotated[:, [i, j]] = rotated[:, [i, j]] @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        if np.abs(rotated - old_rotated).max() < tol:
            break

    if kaiser_normalize:
        rotated = rotated * communalities[:, np.newaxis]

    return rotated

def apply_varimax(pca_input):
    for dim, result in pca_input.items():
        loadings = result['retained_eigenvectors'] * np.sqrt(result['retained_eigenvalues'])
        result['loadings'] = loadings
        result['rotated_loadings'] = varimax_rotation(loadings)
    return pca_input

# |loading| >= 0.40 (Comrey & Lee, 2013)
def flag_significant_loadings(pca_input, threshold=0.4):
    for dim, result in pca_input.items():
        loadings_df = pd.DataFrame(
            result['rotated_loadings'],
            index=result['surviving_indicators'],
            columns=[f'PC{i+1}' for i in range(result['n_retained'])]
        )
        result['loadings_df'] = loadings_df
        result['significant_loadings'] = loadings_df.abs() >= threshold
    return pca_input

# regression method F = X R^-1 Lambda (DiStefano et al., 2009)
def compute_pc_scores(df, pca_input):
    pc_scores_all = {}
    for dim, result in pca_input.items():
        surviving_cols = result['surviving_indicators']
        X = df[surviving_cols].values
        R_inv = np.linalg.inv(df[surviving_cols].corr(method='spearman').values)
        F = X @ R_inv @ result['rotated_loadings']

        pc_score_df = pd.DataFrame(F, columns=[f'PC{i+1}' for i in range(result['n_retained'])])
        pc_score_df.insert(0, 'Block', df['Block'].values)
        pc_score_df.insert(0, 'District', df['District'].values)
        pc_scores_all[dim] = pc_score_df
    return pc_scores_all

# PC signs are arbitrary (Jolliffe, 2002). A PC is flipped when its significant loadings sum negative, so higher scores always mean more loss, exposure, vulnerability or capacity deficit (Cutter et al., 2003)
def build_flip_map_from_loadings(pca_input):
    flip_map = {}
    for dim, result in pca_input.items():
        loadings_df = result['loadings_df']
        significant_df = result['significant_loadings']
        dim_flips = {}
        for i in range(result['n_retained']):
            pc = f'PC{i+1}'
            sig_loadings = loadings_df.loc[significant_df[pc], pc]
            dim_flips[pc] = np.sign(sig_loadings.sum()) < 0
        flip_map[dim] = dim_flips
    return flip_map

def harmonize_pc_signs(pc_scores, flip_map):
    harmonized = {}
    for dim, score_df in pc_scores.items():
        df = score_df.copy()
        for pc, should_flip in flip_map[dim].items():
            if should_flip:
                df[pc] = -df[pc]
        harmonized[dim] = df
    return harmonized

def export_pca_outputs(pca_input, pc_scores, pc_scores_harmonized, flip_map, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for dim, result in pca_input.items():
            result['correlation_matrix'].to_excel(writer, sheet_name=f'Spearman_PostKMO_{dim}'[:31])

        eigen_rows, cumvar_rows, retention_rows, loading_rows, flip_rows = [], [], [], [], []
        for dim, result in pca_input.items():
            for i, (ev, evr, cv) in enumerate(zip(result['eigenvalues'], result['explained_variance_ratio'],
                                                   result['cumulative_variance']), start=1):
                eigen_rows.append({'Dimension': dim, 'Component': f'PC{i}', 'Eigenvalue': ev})
                cumvar_rows.append({'Dimension': dim, 'Component': f'PC{i}',
                                    'Explained_Variance_Ratio': evr, 'Cumulative_Variance': cv})
            retention_rows.append({
                'Dimension': dim,
                'Components_Retained': result['n_retained'],
                'Total_Components_Available': len(result['eigenvalues']),
                'Cumulative_Variance_Explained': result['cumvar_at_retained']
            })
            loadings_out = result['loadings_df'].reset_index().rename(columns={'index': 'Indicator'})
            loadings_out.insert(0, 'Dimension', dim)
            loading_rows.append(loadings_out)
            for pc, flipped in flip_map[dim].items():
                flip_rows.append({'Dimension': dim, 'Component': pc, 'Flipped': flipped})

        pd.DataFrame(eigen_rows).to_excel(writer, sheet_name='Eigendecomposition', index=False)
        pd.DataFrame(cumvar_rows).to_excel(writer, sheet_name='Cumulative_Variance', index=False)
        pd.DataFrame(retention_rows).to_excel(writer, sheet_name='Component_Retention', index=False)
        pd.concat(loading_rows, ignore_index=True).to_excel(writer, sheet_name='Varimax_Rotated_Loadings', index=False)
        pd.DataFrame(flip_rows).to_excel(writer, sheet_name='Sign_Harmonization_Log', index=False)

        for dim, result in pca_input.items():
            loadings_df = result['loadings_df']
            significant_df = result['significant_loadings']

            display_df = loadings_df.copy()
            for pc in display_df.columns:
                display_df[pc] = display_df[pc].round(3).astype(str)
                display_df.loc[significant_df[pc], pc] = display_df.loc[significant_df[pc], pc] + ' *'
            display_df.reset_index().rename(columns={'index': 'Indicator'}).to_excel(
                writer, sheet_name=f'Significant_Loadings_{dim}'[:31], index=False)

            # PC Name is left blank: naming components is a manual, context-specific step
            rows = []
            for pc in loadings_df.columns:
                sig_series = loadings_df.loc[significant_df[pc], pc].sort_values(ascending=False)
                for ind, val in sig_series.items():
                    rows.append({'Component': pc, 'Indicator': ind, 'Loading_Value': round(val, 3),
                                 'Direction': 'Positive' if val > 0 else 'Negative', 'PC Name': ''})
            pd.DataFrame(rows).to_excel(writer, sheet_name=f'Retained_Loadings_{dim}'[:31], index=False)

        for dim, score_df in pc_scores.items():
            score_df.to_excel(writer, sheet_name=f'PC_Scores_{dim}'[:31], index=False)
        for dim, score_df in pc_scores_harmonized.items():
            score_df.to_excel(writer, sheet_name=f'PC_Scores_Harmonized_{dim}'[:31], index=False)


# Plotting figures
def plot_scree(pca_input, dim, pipeline_label, output_folder, n_blocks,
               study_area='', variance_threshold=0.80):
    result = pca_input[dim]
    eigenvalues = result['eigenvalues']
    cumulative = result['cumulative_variance'] * 100
    n_retained = result['n_retained']
    n_components = len(eigenvalues)
    x = np.arange(1, n_components + 1)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    colors = ['#1f4e8c' if i < n_retained else '#b8cbe8' for i in range(n_components)]
    ax1.bar(x, eigenvalues, color=colors, label='Eigenvalue (λ)', zorder=2)
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, label='Kaiser criterion (λ = 1.0)', zorder=1)
    ax1.set_xlabel('Component Number', fontsize=13)
    ax1.set_ylabel('Eigenvalue (λ)', fontsize=13, color='#1f4e8c')
    ax1.tick_params(axis='y', labelcolor='#1f4e8c')
    ax1.set_xticks(x)

    threshold_pct = variance_threshold * 100
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color='#c62828', marker='o', linewidth=2.5, label='Cumulative variance (%)', zorder=3)
    ax2.axhline(threshold_pct, color='#c62828', linestyle=':', linewidth=1.5,
                label=f'{threshold_pct:.0f}% threshold', zorder=1)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=13, color='#c62828')
    ax2.tick_params(axis='y', labelcolor='#c62828')
    ax2.set_ylim(0, 105)

    retained_cum = cumulative[n_retained - 1]
    ax1.axvline(n_retained + 0.5, color='#4472c4', linestyle='-.', linewidth=1.3, zorder=1)
    ax2.annotate(f'{n_retained} PCs\nretained', xy=(n_retained, retained_cum),
                 xytext=(n_retained + 0.3, retained_cum + 12), fontsize=11, color='#4472c4', ha='left')
    ax2.annotate(f'{retained_cum:.1f}%', xy=(n_retained, retained_cum),
                 xytext=(n_retained + 0.4, retained_cum - 8), fontsize=11, color='#c62828',
                 arrowprops=dict(arrowstyle='->', color='#c62828', lw=1))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10, framealpha=0.9)

    area_text = f'{study_area} ' if study_area else ''
    plt.title(f'{pipeline_label} | {dim} Dimension — Scree Plot\n{area_text}(n = {n_blocks} blocks)',
              fontsize=15, fontweight='bold')
    fig.text(0.1, -0.02,
             f"Bars = eigenvalues (λ); filled bars = retained components (n={n_retained}). "
             f"Red line = cumulative variance explained. Dashed line = Kaiser criterion (λ=1.0). "
             f"Retention threshold: ≥{threshold_pct:.0f}% cumulative variance.",
             fontsize=9, style='italic', ha='left')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, f'{pipeline_label}_{dim}_Scree_Plot.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)

# prefixes match the test case indicator naming; other datasets keep their names as they are
DEFAULT_LABEL_PREFIXES = ['AAEL_', 'AAHL_', 'EQ_SigExp_', 'EQ_LS_SigExp_', 'EQ_Exp_', 'V_', 'CD_',
                          'F_SigExp_', 'FF_SigExp_', 'P_LS_SigExp_', 'F_Exp_', 'FF_Exp_']

def clean_indicator_label(name, prefixes=DEFAULT_LABEL_PREFIXES):
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
            break
    name = name.replace('_RsL', '').replace('_RsCr', '').replace('_km2', '')
    name = name.replace('_Count', '').replace('_km', '')
    return name.replace('_', ' ')

def plot_loading_heatmap(pca_input, dim, pipeline_label, output_folder, n_blocks,
                         fig_number='', dataset_label='', study_area='', threshold=0.40):
    result = pca_input[dim]
    loadings_df = result['loadings_df']
    evr = result['explained_variance_ratio'][:result['n_retained']] * 100

    labels = [clean_indicator_label(i) for i in loadings_df.index]
    data = loadings_df.values
    n_indicators, n_components = data.shape

    fig, ax = plt.subplots(figsize=(max(8, n_components * 1.6), max(6, n_indicators * 0.55)))

    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(data, cmap=plt.cm.RdBu_r, norm=norm, aspect='auto')

    for i in range(n_indicators):
        for j in range(n_components):
            val = data[i, j]
            is_sig = abs(val) >= threshold
            text_color = 'white' if abs(val) > 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9.5,
                    color=text_color, fontweight='bold' if is_sig else 'normal')
            if is_sig:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor='#1f3864', linewidth=1.8))

    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f'PC{i+1}\n({evr[i]:.1f}%)' for i in range(n_components)],
                       fontsize=10.5, color='#1f3864', fontweight='bold')
    ax.set_yticks(range(n_indicators))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Factor Loading', fontsize=11)
    cbar.ax.axhline(threshold, color='#1f3864', linestyle='--', linewidth=1)
    cbar.ax.axhline(-threshold, color='#1f3864', linestyle='--', linewidth=1)

    fig_prefix = f'Fig. {fig_number} | ' if fig_number else ''
    plt.suptitle(f'{fig_prefix}{dim.replace("_", " ")} Dimension — Varimax Rotated Factor Loadings',
                 fontsize=14, fontweight='bold', color='#1f3864', y=1.0)
    subtitle_parts = [p for p in [dataset_label, study_area] if p]
    ax.set_title(f'{" | ".join(subtitle_parts)} (n = {n_blocks} blocks)', fontsize=11.5, color='#1f3864', pad=10)

    fig.text(0.02, -0.01,
             f"Rotated factor loadings from Varimax orthogonal rotation. Values in each cell indicate the correlation "
             f"between the variable and component. Bold values and outlined cells: |loading| ≥ {threshold:.2f} "
             f"(interpretation threshold). PC labels show % variance explained.",
             fontsize=8.5, style='italic', ha='left')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    file_tag = f'{fig_number}_' if fig_number else ''
    filepath = os.path.join(output_folder, f'{pipeline_label}_{file_tag}Loadings_{dim}.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)


## Notes:
# Order of operations in this step:
# 1. Spearman correlation matrix built from KMO-surviving indicators, per dimension.
# 2. Eigendecomposition of each matrix, sorted by descending eigenvalue.
# 3. Explained and cumulative variance computed for all components.
# 4. Components retained until cumulative variance reaches 80%.
# 5. Varimax rotation with Kaiser normalisation on the retained loadings.
# 6. Indicators with |loading| >= 0.40 flagged as significant for each PC.
# 7. PC scores computed by the regression method, F = X R^-1 Lambda.
# 8. Sign harmonisation, so higher scores always mean more loss, exposure, vulnerability or capacity deficit.

# Why a Spearman matrix and not sklearn's PCA: sklearn runs PCA on the covariance matrix. Running eigendecomposition directly on the Spearman matrix keeps PCA consistent with the KMO and Bartlett tests (Steps 5-7) and limits the influence of residual skew.

# Why 80% cumulative variance and not Kaiser (eigenvalue > 1): in the test case, Kaiser retained components covering only 60-70% of variance.

# Why Kaiser normalisation in Varimax: without it, indicators loaded across several PCs and components were hard to interpret. Normalisation gives each indicator equal weight in the rotation.

# Why sign harmonisation: PC signs are mathematically arbitrary. Without harmonisation, PC scores cannot be added into a dimension score. A constrained (non-negative) PCA was tested as an alternative and was not feasible for this data.

# Limitation: for PCs with mixed-sign significant loadings, harmonisation follows the dominant direction and only partly resolves the conflict (3 vulnerability PCs in the test case).