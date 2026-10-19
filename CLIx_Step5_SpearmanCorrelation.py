import pandas as pd

# one matrix per dimension, so KMO, Bartlett and PCA all run within dimensions;
# Spearman is rank-based and robust to the skew left after treatment
def compute_spearman_matrices(df, dim_map):
    dimensions = sorted(set(dim_map.values()))
    matrices = {}
    for dim in dimensions:
        cols = [c for c in df.columns if dim_map.get(c) == dim]
        corr_matrix = df[cols].corr(method='spearman')
        matrices[dim] = corr_matrix
    return matrices

def export_spearman_matrices(spearman_matrices, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for dim, matrix in spearman_matrices.items():
            # Excel caps sheet names at 31 characters
            sheet_name = f'Spearman_PreKMO_{dim}'[:31]
            matrix.to_excel(writer, sheet_name=sheet_name)