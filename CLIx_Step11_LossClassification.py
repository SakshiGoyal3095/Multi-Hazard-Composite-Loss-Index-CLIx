import pandas as pd

# Jenks natural breaks (Jenks, 1967): minimises within-class variance, so class edges fall at real gaps in the data instead of at equal intervals
def jenks_breaks(data, n_classes):
    data_sorted = sorted(data)
    mat1 = [[0] * (n_classes + 1) for _ in range(len(data_sorted) + 1)]
    mat2 = [[0] * (n_classes + 1) for _ in range(len(data_sorted) + 1)]

    for i in range(1, n_classes + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_sorted) + 1):
            mat2[j][i] = float('inf')

    v = 0.0
    for l in range(2, len(data_sorted) + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = data_sorted[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, n_classes + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v

    k = len(data_sorted)
    kclass = [0.0] * (n_classes + 1)
    kclass[n_classes] = data_sorted[-1]
    kclass[0] = data_sorted[0]
    countNum = n_classes
    while countNum >= 2:
        idx = int(mat1[k][countNum]) - 2
        kclass[countNum - 1] = data_sorted[idx]
        k = int(mat1[k][countNum] - 1)
        countNum -= 1

    return kclass

def assign_jenks_classes(data, breaks):
    classes = []
    for val in data:
        for i in range(len(breaks) - 1):
            if val <= breaks[i + 1] or i == len(breaks) - 2:
                classes.append(i + 1)
                break
    return classes

# goodness of variance fit: share of total variance explained by the classes (1 = perfect)
def goodness_of_variance_fit(data, breaks):
    data = sorted(data)
    sdam = sum((x - sum(data) / len(data)) ** 2 for x in data)

    classes = []
    current_class = []
    for x in data:
        if current_class and x > breaks[len(classes) + 1]:
            classes.append(current_class)
            current_class = []
        current_class.append(x)
    classes.append(current_class)

    sdcm = sum(sum((x - sum(c) / len(c)) ** 2 for x in c) for c in classes)
    return (sdam - sdcm) / sdam


DEFAULT_CLASS_LABELS = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High', 6: 'Extreme'}

# classified on CLIx_raw; rescaling to 0-100 is linear, so classes are identical
def classify_loss(clix_df, n_classes=6, class_labels=None):
    if class_labels is None:
        class_labels = DEFAULT_CLASS_LABELS if n_classes == 6 else {i: f'Class {i}' for i in range(1, n_classes + 1)}
    if len(class_labels) != n_classes:
        raise ValueError(f"class_labels has {len(class_labels)} labels but n_classes is {n_classes}.")

    data = clix_df['CLIx_raw'].tolist()
    breaks = jenks_breaks(data, n_classes)
    clix_df = clix_df.copy()
    clix_df['Loss_Class'] = assign_jenks_classes(data, breaks)
    clix_df['Loss_Class_Label'] = clix_df['Loss_Class'].map(class_labels)
    gvf = goodness_of_variance_fit(data, breaks)
    return clix_df, breaks, gvf

# checks whether the top class holds the same blocks across several class counts; a block isolated in every solution is a stable feature of the data
def compare_class_solutions(clix_df, class_counts=(4, 5, 6)):
    data = clix_df['CLIx_raw'].tolist()
    rows = []
    for n in class_counts:
        breaks = jenks_breaks(data, n)
        classes = assign_jenks_classes(data, breaks)
        top_blocks = clix_df.loc[[c == n for c in classes], 'Block'].tolist()
        rows.append({
            'N_Classes': n,
            'GVF': goodness_of_variance_fit(data, breaks),
            'Top_Class_Block_Count': len(top_blocks),
            'Top_Class_Blocks': ', '.join(top_blocks)
        })
    return pd.DataFrame(rows)

def export_loss_classification(clix_df, breaks, gvf, comparison_df, output_path):
    export_df = clix_df[['District', 'Block', 'CLIx_raw', 'CLIx', 'CLIx_Rank', 'Loss_Class', 'Loss_Class_Label']]\
        .sort_values('CLIx', ascending=False).reset_index(drop=True)
    breaks_df = pd.DataFrame({
        'Class': list(range(1, len(breaks))),
        'Lower_Break_CLIx_raw': breaks[:-1],
        'Upper_Break_CLIx_raw': breaks[1:]
    })
    breaks_df['GVF'] = gvf

    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        export_df.to_excel(writer, sheet_name='Loss_Classification', index=False)
        breaks_df.to_excel(writer, sheet_name='Jenks_Breaks', index=False)
        comparison_df.to_excel(writer, sheet_name='Jenks_Class_Comparison', index=False)


## Notes:
# Loss classification (Section 3.6.3): CLIx scores are grouped into Very Low, Low, Moderate, High, Very High and Extreme using Jenks natural breaks.

# Why six classes and not the usual five: in both implementations, one block (Bahadrabad in CLIx-EQ, Doiwala in CLIx-FL) is separated from the next block by a gap several times larger than any other gap (see CLIx_Gap_Analysis, Step 10). This block sits alone in the top class in the four-, five- and six-class solutions (compare_class_solutions), so its isolation is a stable feature of the data. A sixth class, Extreme, represents it explicitly.

# Implementation check: breaks from this code were compared with QGIS natural breaks for the 5-class solution. GVF was 0.916 for this code and 0.914 for QGIS, so this implementation was retained.