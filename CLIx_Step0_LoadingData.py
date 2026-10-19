import pandas as pd

def load_implementation(path):
    aal = pd.read_excel(path, sheet_name='AAL').dropna(how='all')
    exp = pd.read_excel(path, sheet_name='Exposure').dropna(how='all')
    vuln = pd.read_excel(path, sheet_name='Vulnerability').dropna(how='all')
    cd = pd.read_excel(path, sheet_name='Capacity_Deficit').dropna(how='all')
    return aal, exp, vuln, cd

def print_sheet_summary(aal, exp, vuln, cd):
    for name, df in [('AAL', aal), ('Exposure', exp), ('Vulnerability', vuln), ('Capacity_Deficit', cd)]:
        n_indicators = len([c for c in df.columns if c not in ('District', 'Block')])
        print(f"{name}: {df.shape[0]} blocks, {n_indicators} indicators")

# every indicator remembers its dimension, so later steps can run per dimension
def tag_dimensions(aal, exp, vuln, cd):
    dim_map = {}
    for col in aal.columns:
        if col not in ('District', 'Block'):
            dim_map[col] = 'AAL'
    for col in exp.columns:
        if col not in ('District', 'Block'):
            dim_map[col] = 'Exposure'
    for col in vuln.columns:
        if col not in ('District', 'Block'):
            dim_map[col] = 'Vulnerability'
    for col in cd.columns:
        if col not in ('District', 'Block'):
            dim_map[col] = 'Capacity_Deficit'
    return dim_map

def merge_sheets(aal, exp, vuln, cd):
    merged = aal.merge(exp, on=['District', 'Block']).merge(vuln, on=['District', 'Block']).merge(cd, on=['District', 'Block'])
    return merged