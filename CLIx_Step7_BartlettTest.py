import pandas as pd
import numpy as np
from scipy.stats import chi2

def bartlett_test(corr_matrix, n):
    R = corr_matrix.values
    k = R.shape[0]
    det_R = np.linalg.det(R)

    chi_sq_stat = -((n - 1) - (2 * k + 5) / 6) * np.log(det_R)
    df = k * (k - 1) / 2
    # sf keeps precision for very small p-values, where 1 - cdf rounds to 0
    p_value = chi2.sf(chi_sq_stat, df)

    return chi_sq_stat, df, p_value

# runs on KMO-surviving indicators only: the test confirms the final set used in PCA
def run_bartlett_all(standardized_df, kmo_results):
    bartlett_results = {}
    n = standardized_df.shape[0]
    for dim, result in kmo_results.items():
        surviving_cols = result['surviving_indicators']
        corr_matrix = standardized_df[surviving_cols].corr(method='spearman')
        chi_sq_stat, df, p_value = bartlett_test(corr_matrix, n)
        bartlett_results[dim] = {'chi_sq': chi_sq_stat, 'df': df, 'p_value': p_value}
    return bartlett_results

def build_bartlett_results_table(bartlett_results):
    rows = []
    for dim, result in bartlett_results.items():
        rows.append({
            'Dimension': dim,
            'Chi_Square': result['chi_sq'],
            'Degrees_of_Freedom': result['df'],
            'p_value': result['p_value'],
            'Significant_(p<0.001)': result['p_value'] < 0.001
        })
    return pd.DataFrame(rows)

def export_bartlett_results(bartlett_results_table, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        bartlett_results_table.to_excel(writer, sheet_name='Bartletts_Test', index=False)