import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DIMENSION_SCORE_COLUMNS = ['DS_AAL', 'DS_Exposure', 'DS_Vulnerability', 'DS_Capacity_Deficit']

# percentile rank on a 0-1 scale, relative to all blocks in the run
def compute_percentile_ranks(clix_df):
    clix_df = clix_df.copy()
    for col in DIMENSION_SCORE_COLUMNS:
        clix_df[f'PR_{col}'] = clix_df[col].rank(pct=True)
    return clix_df

# conditions are checked in priority order; the first match wins
def classify_loss_regime(row, H=0.67, L=0.33):
    prs = {
        'AAL': row['PR_DS_AAL'],
        'Exposure': row['PR_DS_Exposure'],
        'Vulnerability': row['PR_DS_Vulnerability'],
        'CapacityDeficit': row['PR_DS_Capacity_Deficit']
    }
    if all(v >= H for v in prs.values()):
        return 'Cascading'
    if all(v <= L for v in prs.values()):
        return 'Resilient'
    if prs['Exposure'] >= H and prs['Vulnerability'] >= H:
        return 'Exposure- and Vulnerability-Driven'
    if prs['Exposure'] >= H:
        return 'Exposure-Driven'
    if prs['Vulnerability'] >= H:
        return 'Vulnerability-Driven'
    return 'Mixed/Moderate'

def assign_loss_regimes(clix_df, H=0.67, L=0.33):
    clix_df = compute_percentile_ranks(clix_df)
    clix_df['Loss_Regime'] = clix_df.apply(lambda row: classify_loss_regime(row, H=H, L=L), axis=1)
    return clix_df

def build_final_master_summary(clix_df):
    summary_df = clix_df[['District', 'Block'] + DIMENSION_SCORE_COLUMNS +
                         ['CLIx_raw', 'CLIx', 'CLIx_Rank', 'Loss_Class', 'Loss_Class_Label', 'Loss_Regime']].copy()
    summary_df = summary_df.rename(columns={'CLIx': 'CLIx_rescaled_0-100'})
    return summary_df.sort_values('CLIx_Rank').reset_index(drop=True)

def export_loss_regime(clix_df, output_path):
    pr_cols = [f'PR_{c}' for c in DIMENSION_SCORE_COLUMNS]
    regime_df = clix_df[['District', 'Block'] + pr_cols + ['Loss_Regime']].sort_values('Loss_Regime')
    regime_counts = clix_df['Loss_Regime'].value_counts().rename_axis('Loss_Regime').reset_index(name='Block_Count')

    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        regime_df.to_excel(writer, sheet_name='Loss_Regime', index=False)
        regime_counts.to_excel(writer, sheet_name='Loss_Regime_Counts', index=False)
        build_final_master_summary(clix_df).to_excel(writer, sheet_name='CLIx_Master_Summary', index=False)


# Plotting figure
def plot_clix_heatmap(clix_df, pipeline_label, output_folder, fig_number='', dataset_label='',
                      study_area='', categories=('Extreme', 'Very High', 'High')):
    # original index is kept so each row's colours come from the same block
    subset = clix_df[clix_df['Loss_Class_Label'].isin(categories)].sort_values('CLIx_Rank')
    n = len(subset)
    n_blocks = len(clix_df)

    dim_labels = ['AAL', 'Exposure', 'Vulnerability', 'Capacity\nDeficit']

    # colours normalised per column across all blocks, then subset
    norm_full = clix_df[DIMENSION_SCORE_COLUMNS].apply(lambda c: (c - c.mean()) / c.std())
    data = norm_full.loc[subset.index, DIMENSION_SCORE_COLUMNS].values
    subset = subset.reset_index(drop=True)

    row_labels = [f"#{r.CLIx_Rank}  {r.Block}, {r.District}" for r in subset.itertuples()]

    regime_markers = {
        'Cascading': '▲', 'Exposure-Driven': '●', 'Vulnerability-Driven': '■',
        'Exposure- and Vulnerability-Driven': '◆', 'Resilient': '○', 'Mixed/Moderate': '·'
    }

    fig_height = max(6, n * 0.45) + 2.6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_height),
                                   gridspec_kw={'width_ratios': [3.2, 1.3], 'wspace': 0.35})
    # fixed margins in inches, so title, colour bar and note never collide whatever the block count
    fig.subplots_adjust(top=1 - 1.0 / fig_height, bottom=1.9 / fig_height, left=0.17, right=0.86)

    norm = mcolors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    im = ax1.imshow(data, cmap=plt.cm.RdBu_r, norm=norm, aspect='auto')

    for i in range(n):
        for j, col in enumerate(DIMENSION_SCORE_COLUMNS):
            raw_val = subset[col].iloc[i]
            text_color = 'white' if abs(data[i, j]) > 1.3 else 'black'
            ax1.text(j, i, f'{raw_val:.2f}', ha='center', va='center', fontsize=9, color=text_color)

    ax1.set_xticks(range(len(DIMENSION_SCORE_COLUMNS)))
    ax1.set_xticklabels(dim_labels, fontsize=10.5, fontweight='bold', color='#1f3864')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(row_labels, fontsize=9.5)
    ax1.tick_params(length=0)

    # markers sit just right of the heatmap so they never cover the block names
    marker_x = len(DIMENSION_SCORE_COLUMNS) - 0.5 + 0.3
    for i, row in enumerate(subset.itertuples()):
        marker = regime_markers.get(row.Loss_Regime, '·')
        ax1.text(marker_x, i, marker, fontsize=13, color='#a83214', ha='center', va='center', clip_on=False)

    cat_colors = {'Extreme': '#67000d', 'Very High': '#a50f15', 'High': '#f16913'}
    bar_colors = [cat_colors.get(c, 'gray') for c in subset['Loss_Class_Label']]
    ax2.barh(range(n), subset['CLIx'], color=bar_colors)
    # same row order as the heatmap (rank 1 at the top), so rows line up across panels
    ax2.set_ylim(n - 0.5, -0.5)
    ax2.set_yticks([])
    ax2.set_xlim(0, 115)
    ax2.set_xlabel('CLIx Score (0–100)', fontsize=10.5)
    for i, v in enumerate(subset['CLIx']):
        ax2.text(v + 1.5, i, f'{v:.1f}', va='center', fontsize=9)

    # separate colour bar axes in the bottom margin, so both panels keep identical rows
    heat_box = ax1.get_position()
    cbar_ax = fig.add_axes([heat_box.x0 + heat_box.width * 0.2, 0.95 / fig_height, heat_box.width * 0.6, 0.15 / fig_height])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Dimension Score (column z-score)', fontsize=9.5)
    cbar.ax.xaxis.set_label_position('top')

    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=10, label=cat)
                       for cat, c in cat_colors.items() if cat in categories]
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9,
               title='Loss Category (Jenks)', framealpha=0.9)

    fig_prefix = f'Fig. {fig_number} | ' if fig_number else ''
    subtitle_parts = [p for p in [dataset_label, study_area] if p]
    plt.suptitle(f'{fig_prefix}Composite Loss Index (CLIx) — Dimension Score Profiles, '
                 f'{" and ".join(categories)} Loss Potential Blocks\n'
                 f'{" | ".join(subtitle_parts)} (n = {n_blocks} blocks)',
                 fontsize=13, fontweight='bold', color='#1f3864', y=1 - 0.2 / fig_height, va='top')

    fig.text(0.02, 0.15 / fig_height,
             f"Heatmap of dimension scores for all {n} blocks classified as {'/'.join(categories)} loss potential "
             f"(Jenks Natural Breaks). Cell values = actual dimension scores; colours normalised within each column "
             f"across all {n_blocks} blocks. CLIx Score (right) = final composite score (0–100), coloured by Loss "
             f"Category.\nMarkers = Loss Regime: ▲ Cascading, ◆ Exposure- and Vulnerability-Driven, "
             f"● Exposure-Driven, ■ Vulnerability-Driven, ○ Resilient, · Mixed/Moderate.",
             fontsize=8.5, style='italic', ha='left', va='bottom')

    os.makedirs(output_folder, exist_ok=True)
    file_tag = f'{fig_number}_' if fig_number else ''
    filepath = os.path.join(output_folder, f'{pipeline_label}_{file_tag}CLIx_Heatmap.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)


## Notes:
# Loss regime classification (Section 3.6.4): each dimension score is converted to a percentile rank (0-1) across all blocks. Tertile thresholds H = 0.67 and L = 0.33 split blocks into three equal groups. Tertiles were chosen over quartiles because, with 95 blocks, each threshold group still holds enough blocks for stable interpretation.

# Regimes, in priority order:
# 1. Cascading: all four dimensions >= H
# 2. Resilient: all four dimensions <= L
# 3. Exposure- and Vulnerability-Driven: exposure and vulnerability both >= H
# 4. Exposure-Driven: exposure >= H
# 5. Vulnerability-Driven: vulnerability >= H
# 6. Mixed/Moderate: all remaining blocks

# How the final rules were reached: a first version required the driving dimension to be the only one above H. This left 59 (CLIx-EQ) and 64 (CLIx-FL) blocks unclassified. A gap-based rule (top dimension ahead of the second by 0.10-0.25) was also tested and rejected. The final version adds the combined exposure-vulnerability regime and applies the rules in priority order.

# Expected pattern, not an error: exposure and vulnerability each have exactly 32 blocks at or above H in both implementations. This is a property of percentile ranking. Every dimension has roughly the top third of blocks above the 67th percentile, whatever it measures.