import pandas as pd

# asset labels for the test case indicator names; other datasets fall back to the raw name
DEFAULT_ASSET_NAME_MAP = {
    "Residents_Count": "Residents",
    "KatchchaStructures_Count": "Kachha Structures",
    "PuccaStructures_Count": "Pucca Structures",
    "EducationalInstitutes_Count": "Educational Institutes",
    "HealthCentres_Count": "Health Centres",
    "FirePoliceStations_Count": "Fire and Police Stations",
    "TransportHubs_Count": "Transport Hubs",
    "HydropowerStations_Count": "Hydropower Stations",
    "Substations_Count": "Power Substations",
    "AgriculturalArea_km2": "Agricultural Area",
    "HorticulturalArea_km2": "Horticultural Area",
    "IndustrialArea_km2": "Industrial Area",
    "CommercialZones_m2": "Commercial Zones",
    "PavedRoads_km": "Paved Roads",
    "UnpavedRoads_km": "Unpaved Roads",
    "TourismRevenue_RsCr": "Tourism Revenue",
    "Tourists_Count": "Tourists"
}

DEFAULT_HAZARD_PREFIXES = ["EQ_SigExp_", "EQ_LS_SigExp_", "EQ_Exp_",
                           "F_SigExp_", "F_Exp_", "FF_SigExp_", "FF_Exp_", "P_LS_SigExp_"]

def get_asset_name(indicator_col, asset_name_map=DEFAULT_ASSET_NAME_MAP, hazard_prefixes=DEFAULT_HAZARD_PREFIXES):
    name = indicator_col
    for prefix in hazard_prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return asset_name_map.get(name, name)

# top assets per exposure PC, ranked by |loading|, so the exposure column names physical assets instead of abstract components
def get_top_assets_per_pc(pca_input, dimension='Exposure', top_n=3, loading_threshold=0.40):
    loadings_df = pca_input[dimension]['loadings_df']
    pc_top_assets = {}
    for pc in loadings_df.columns:
        abs_loadings = loadings_df[pc].abs()
        top = abs_loadings[abs_loadings >= loading_threshold].sort_values(ascending=False).head(top_n)
        pc_top_assets[pc] = [get_asset_name(ind) for ind in top.index]
    return pc_top_assets

# PCs within threshold_pct of the block's highest score count as dominant (max 2). use_absolute_scores=True ranks by |score|; False ranks by score, so only above-average components can be a block's loss driver
def get_dominant_pcs(score_row, threshold_pct=0.15, max_pcs=2, use_absolute_scores=True):
    pc_cols = [c for c in score_row.index if c not in ('District', 'Block')]
    scores = score_row[pc_cols].astype(float)
    if use_absolute_scores:
        scores = scores.abs()
    max_val = scores.max()
    if max_val <= 0:
        return []
    close = scores[scores >= max_val * (1 - threshold_pct)].index.tolist()
    return sorted(close, key=lambda c: scores[c], reverse=True)[:max_pcs]

def combine_deduplicate(*item_lists):
    combined = []
    for items in item_lists:
        for item in items:
            if item not in combined:
                combined.append(item)
    return combined

# one row per block; cells hold lists so EQ and FL tables can be merged later
def build_dominant_driver_table(pc_scores_harmonized, pca_input, clix_df, component_names_dict=None,
                                exposure_top_n=3, threshold_pct=0.15, use_absolute_scores=True):
    component_names_dict = component_names_dict or {}
    exposure_assets = get_top_assets_per_pc(pca_input, 'Exposure', top_n=exposure_top_n)

    rows = []
    ranked = clix_df[['District', 'Block', 'CLIx_Rank']].sort_values('CLIx_Rank')
    for _, block_row in ranked.iterrows():
        key = (block_row['District'], block_row['Block'])
        row = {'Rank': block_row['CLIx_Rank'], 'District': key[0], 'Block': key[1]}

        for dim, score_df in pc_scores_harmonized.items():
            score_row = score_df.set_index(['District', 'Block']).loc[key]
            dominant_pcs = get_dominant_pcs(score_row, threshold_pct, use_absolute_scores=use_absolute_scores)
            if dim == 'Exposure':
                row[dim] = combine_deduplicate(*[exposure_assets.get(pc, []) for pc in dominant_pcs])
            else:
                row[dim] = [component_names_dict.get((dim, pc), pc) for pc in dominant_pcs]
        rows.append(row)
    return pd.DataFrame(rows)

# merges two hazard tables (e.g. CLIx-EQ and CLIx-FL) block by block, ordered by the first table's rank
def combine_driver_tables(first_table, second_table):
    merged = first_table.merge(second_table, on=['District', 'Block'], suffixes=('', '_second'))
    dims = [c for c in first_table.columns if c not in ('Rank', 'District', 'Block')]
    for dim in dims:
        merged[dim] = [combine_deduplicate(a, b) for a, b in zip(merged[dim], merged[f'{dim}_second'])]
    keep = ['Rank', 'District', 'Block'] + dims
    return merged[keep].sort_values('Rank').reset_index(drop=True)

def format_driver_table(driver_table):
    formatted = driver_table.copy()
    for col in formatted.columns:
        if col in ('Rank', 'District', 'Block'):
            continue
        formatted[col] = formatted[col].apply(lambda items: ';\n'.join(items) if items else '—')
    return formatted

def export_dominant_drivers(driver_table, output_path, sheet_name='Dominant_Loss_Drivers'):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        format_driver_table(driver_table).to_excel(writer, sheet_name=sheet_name[:31], index=False)


## Notes:
# Purpose: identifies which sectors drive loss in each block, so CLIx results can feed directly into block-level planning (DM Action Plans, GPDP, PDNA).
        
# Method: for each block and dimension, the PCs scoring within 15% of the block's highest harmonised PC score are taken as dominant (at most two). AAL, vulnerability and capacity deficit report the names of those PCs. Exposure reports the top three physical assets (highest |loading|) of each dominant PC, because exposure components group many assets and a PC name alone was too broad.

# Exposure asset limit: listing every significant indicator produced too many entries per block. Thresholds of 0.50-0.80 were tested; limiting to the three highest-loading assets per PC kept the table readable without dropping the main drivers.

# Component names: PC names are a manual interpretation of the loadings and must be entered in the 'PC Name' column of the Retained_Loadings sheets. Unnamed PCs appear as PC1, PC2...

# Combining hazards: combine_driver_tables merges two implementations (e.g. CLIx-EQ and CLIx-FL) into one table, removing duplicate entries and ordering blocks by the first table's rank.