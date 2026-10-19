import os
import pandas as pd

from CLIx_Step0_LoadingData import load_implementation, print_sheet_summary, tag_dimensions, merge_sheets
from CLIx_Step1_ZeroVarianceCheck import check_zero_variance, drop_zero_variance, build_summary, export_variance_step
from CLIx_Step2_SkewnessKurtosisCheck import (check_distribution, apply_skew_kurtosis_treatment,
                                              drop_post_treatment_zero_variance, build_treatment_summary,
                                              export_distribution_step, export_treatment_step,
                                              save_raw_histograms, save_treated_histograms)
from CLIx_Step3_OutlierDetection import flag_outliers, export_outlier_step, plot_outlier_boxplots
from CLIx_Step4_Standardisation import zscore_standardise, export_standardization_step, plot_standardised_boxplots
from CLIx_Step5_SpearmanCorrelation import compute_spearman_matrices, export_spearman_matrices
from CLIx_Step6_KMOTest import (iterative_kmo_dropping, build_kmo_results_table, export_kmo_results,
                                build_full_drop_summary, export_full_drop_summary)
from CLIx_Step7_BartlettTest import run_bartlett_all, build_bartlett_results_table, export_bartlett_results
from CLIx_Step8_PCA import (build_pca_input_matrices, eigendecompose, compute_cumulative_variance,
                            retain_by_cumulative_variance, apply_varimax, flag_significant_loadings,
                            compute_pc_scores, build_flip_map_from_loadings, harmonize_pc_signs,
                            export_pca_outputs, plot_scree, plot_loading_heatmap)
from CLIx_Step9_WithinDimensionWeights import (compute_within_dimension_weights, compute_dimension_scores,
                                               load_component_names, build_zero_value_verification,
                                               export_weights_and_dimension_scores, export_zero_value_verification)
from CLIx_Step10_CLIxScoreComputation import (compute_clix_raw, compute_clix_rescaled, add_clix_rank,
                                              compute_gap_analysis, export_clix_scores, save_clix_histogram)
from CLIx_Step11_LossClassification import classify_loss, compare_class_solutions, export_loss_classification
from CLIx_Step12_LossRegimeClassification import assign_loss_regimes, export_loss_regime, plot_clix_heatmap
from CLIx_Step13_DominantLossDriversPerBlock import build_dominant_driver_table, export_dominant_drivers
from CLIx_Step14_DimensionCorrelation import (compute_dimension_correlation, export_dimension_correlation,
                                              plot_dimension_correlation)
from CLIx_Step15_SensitivityCommon import export_sensitivity_results
from CLIx_Step16_SensitivityTest1_IndicatorExclusion import indicator_exclusion_test, build_exclusion_summary
from CLIx_Step17_SensitivityTest2_PCRetentionMethod import retention_method_test
from CLIx_Step18_SensitivityTest3_DimensionWeighting import dimension_weighting_test
from CLIx_Step19_SensitivityTest4_RegimeThreshold import regime_threshold_test



## SETTINGS: edit this block only

INPUT_FILE = "input_template_CLIx.xlsx"      # Excel file with sheets AAL, Exposure, Vulnerability, Capacity_Deficit; file_name_format: "input_template_CLIx-EQ.xlsx" or "input_template_CLIx-FL.xlsx"
PIPELINE_LABEL = ""              # e.g. "CLIx-EQ" or "CLIx-FL"; used in folder names, file names and figure titles
OUTPUT_FOLDER = f"output_{PIPELINE_LABEL}"   # one output folder per implementation, e.g. output_CLIx-EQ or output_CLIx-FL_
STUDY_AREA = ""              # optional, shown in figure titles e.g. "Uttarakhand"
DATASET_LABEL = ""   # optional, shown in figure titles, chnage it to "Earthquake and Seismic-Induced Landslides Dataset" for CLIx-EQ; or "Flood and Precipitation-Induced Landslide Dataset" for CLIx-FL 


## Keep these as it is, no changes are required after this line

# optional: a previous results workbook where the 'PC Name' column of the Retained_Loadings sheets has been filled in by hand; None leaves PCs unnamed
COMPONENT_NAMES_FILE = None

# Section 3.3, pre-processing
SKEW_THRESHOLD = 2.0
KURTOSIS_THRESHOLD = 3.5
LOG_THRESHOLD = 3.5
WINSOR_LOWER = 0.02
WINSOR_UPPER = 0.98
OUTLIER_IQR_MULTIPLIER = 3

# Section 3.4, adequacy testing
KMO_THRESHOLD = 0.5
KMO_MIN_INDICATORS = 3

# Section 3.5, PCA
VARIANCE_RETAINED = 0.80
LOADING_THRESHOLD = 0.40

# Section 3.6, classification
JENKS_CLASSES = 6
HIGH_TERTILE = 0.67
LOW_TERTILE = 0.33

# Section 3.6, dominant loss drivers
EXPOSURE_TOP_ASSETS = 3
DOMINANT_PC_THRESHOLD = 0.15
USE_ABSOLUTE_SCORES = True

# Section 3.7, sensitivity analysis (Test 1 reruns the pipeline once per indicator and takes a few minutes)
RUN_SENSITIVITY = True
SENSITIVITY_RHO_THRESHOLD = 0.80
QUARTILE_HIGH = 0.75
QUARTILE_LOW = 0.25
EXCLUSION_RETAINED_ONLY = False


def create_results_workbook(output_path, settings):
    # every later step appends sheets, so the workbook must exist first;
    # the Settings sheet records exactly how this run was configured
    settings_df = pd.DataFrame([{'Setting': k, 'Value': str(v)} for k, v in settings.items()])
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        settings_df.to_excel(writer, sheet_name='Run_Settings', index=False)


# one figure folder per step, numbered so they sort in pipeline order
FIGURE_FOLDERS = {
    'raw_histograms': 'Step02_Histograms_RawData',
    'treated_histograms': 'Step02_Histograms_TreatedData',
    'outlier_boxplots': 'Step03_Boxplots_Outliers_3xIQR',
    'standardised_boxplots': 'Step04_Boxplots_Standardised',
    'scree': 'Step08_PCA_ScreePlots',
    'loadings': 'Step08_PCA_LoadingHeatmaps',
    'clix_histogram': 'Step10_CLIx_Histogram',
    'clix_heatmap': 'Step12_CLIx_Heatmap',
    'dimension_correlation': 'Step14_DimensionCorrelation'
}


def run_clix():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    figure_root = os.path.join(OUTPUT_FOLDER, 'Figures')
    fig_dir = {key: os.path.join(figure_root, name) for key, name in FIGURE_FOLDERS.items()}
    output_path = os.path.join(OUTPUT_FOLDER, f'{PIPELINE_LABEL}_Results.xlsx')
    create_results_workbook(output_path, {k: v for k, v in globals().items() if k.isupper()})

    print(f"Running {PIPELINE_LABEL} on {INPUT_FILE}")

    # Step 0: load data (Section 3.2)
    aal, exp, vuln, cd = load_implementation(INPUT_FILE)
    print_sheet_summary(aal, exp, vuln, cd)
    dim_map = tag_dimensions(aal, exp, vuln, cd)
    merged_df = merge_sheets(aal, exp, vuln, cd)
    n_blocks = len(merged_df)
    starting_count = merged_df.shape[1] - 2

    # Step 1: zero variance (3.3.1)
    variance_check = check_zero_variance(merged_df, dim_map)
    cleaned_df, dropped_zero_variance = drop_zero_variance(merged_df, variance_check)
    export_variance_step(variance_check, cleaned_df,
                         build_summary(dropped_zero_variance, starting_count, n_blocks), output_path)

    # Step 2: extreme values (3.3.2)
    export_distribution_step(check_distribution(cleaned_df, dim_map, SKEW_THRESHOLD, KURTOSIS_THRESHOLD), output_path)
    save_raw_histograms(cleaned_df, fig_dir['raw_histograms'])
    treated_df, treatment_log_df = apply_skew_kurtosis_treatment(
        cleaned_df, dim_map, SKEW_THRESHOLD, KURTOSIS_THRESHOLD, LOG_THRESHOLD, WINSOR_LOWER, WINSOR_UPPER)
    treated_df, dropped_post_treatment = drop_post_treatment_zero_variance(treated_df)
    export_treatment_step(treatment_log_df, treated_df,
                          build_treatment_summary(treatment_log_df, LOG_THRESHOLD), output_path, LOG_THRESHOLD)
    save_treated_histograms(treated_df, treatment_log_df, fig_dir['treated_histograms'])

    # Step 3: outlier detection, flags only (3.3.3)
    outlier_flags, outlier_summary = flag_outliers(treated_df, dim_map, OUTLIER_IQR_MULTIPLIER)
    export_outlier_step(outlier_summary, outlier_flags, output_path)
    for dim in sorted(set(dim_map.values())):
        plot_outlier_boxplots(treated_df, dim_map, dim, PIPELINE_LABEL, fig_dir['outlier_boxplots'],
                              OUTLIER_IQR_MULTIPLIER)

    # Step 4: z-score standardisation (3.3.4)
    standardized_df = zscore_standardise(treated_df)
    export_standardization_step(standardized_df, output_path)
    for dim in sorted(set(dim_map.values())):
        plot_standardised_boxplots(standardized_df, dim_map, dim, PIPELINE_LABEL, fig_dir['standardised_boxplots'])
    print(f"Pre-processing done: {standardized_df.shape[1] - 2} of {starting_count} indicators remain")

    # Step 5: Spearman matrices (3.4)
    export_spearman_matrices(compute_spearman_matrices(standardized_df, dim_map), output_path)

    # Step 6: KMO (3.4.1)
    kmo_results, all_dropped = {}, []
    for dim in sorted(set(dim_map.values())):
        surviving_cols, final_kmo, final_msa, dropped_log = iterative_kmo_dropping(
            standardized_df, dim_map, dim, KMO_THRESHOLD, KMO_MIN_INDICATORS)
        kmo_results[dim] = {'surviving_indicators': surviving_cols, 'final_kmo': final_kmo, 'final_msa': final_msa}
        all_dropped.extend(dropped_log)
        print(f"  {dim}: KMO = {final_kmo:.3f}, {len(surviving_cols)} indicators retained")
    kmo_dropped_df = pd.DataFrame(all_dropped, columns=['Dimension', 'Dropped_Indicator', 'MSA_at_drop',
                                                         'Overall_KMO_before_drop'])
    export_kmo_results(build_kmo_results_table(kmo_results), kmo_dropped_df, output_path)
    export_full_drop_summary(build_full_drop_summary(PIPELINE_LABEL, starting_count, dropped_zero_variance,
                                                     dropped_post_treatment, kmo_dropped_df, KMO_THRESHOLD),
                             output_path)

    # Step 7: Bartlett (3.4.2)
    export_bartlett_results(build_bartlett_results_table(run_bartlett_all(standardized_df, kmo_results)), output_path)

    # Step 8: PCA (3.5)
    pca_input = build_pca_input_matrices(standardized_df, kmo_results)
    pca_input = eigendecompose(pca_input)
    pca_input = compute_cumulative_variance(pca_input)
    pca_input = retain_by_cumulative_variance(pca_input, VARIANCE_RETAINED)
    pca_input = apply_varimax(pca_input)
    pca_input = flag_significant_loadings(pca_input, LOADING_THRESHOLD)
    pc_scores = compute_pc_scores(standardized_df, pca_input)
    flip_map = build_flip_map_from_loadings(pca_input)
    pc_scores_harmonized = harmonize_pc_signs(pc_scores, flip_map)
    export_pca_outputs(pca_input, pc_scores, pc_scores_harmonized, flip_map, output_path)
    for dim, result in pca_input.items():
        print(f"  {dim}: {result['n_retained']} PCs retained ({result['cumvar_at_retained']:.1%} variance)")
        plot_scree(pca_input, dim, PIPELINE_LABEL, fig_dir['scree'], n_blocks, STUDY_AREA, VARIANCE_RETAINED)
        plot_loading_heatmap(pca_input, dim, PIPELINE_LABEL, fig_dir['loadings'], n_blocks,
                             dataset_label=DATASET_LABEL, study_area=STUDY_AREA, threshold=LOADING_THRESHOLD)

    # Step 9: within-dimension weights and dimension scores (3.6.1)
    pca_input = compute_within_dimension_weights(pca_input)
    dimension_scores = compute_dimension_scores(pc_scores_harmonized, pca_input)
    component_names = load_component_names(COMPONENT_NAMES_FILE, pca_input.keys()) if COMPONENT_NAMES_FILE else {}
    export_weights_and_dimension_scores(pca_input, dimension_scores, output_path, component_names)
    for dim in pca_input.keys():
        verification = build_zero_value_verification(dim, merged_df, treated_df, standardized_df,
                                                     pc_scores_harmonized, dimension_scores, pca_input, dim_map)
        if verification is not None:
            export_zero_value_verification(dim, verification[0], verification[1], output_path)

    # Step 10: CLIx score, rescaling and ranking (3.6.2)
    clix_df = compute_clix_raw(dimension_scores)
    clix_df, _, _ = compute_clix_rescaled(clix_df)
    clix_df = add_clix_rank(clix_df)
    export_clix_scores(clix_df, compute_gap_analysis(clix_df), output_path)
    save_clix_histogram(clix_df, fig_dir['clix_histogram'], f'{PIPELINE_LABEL}_CLIx_Histogram.png', PIPELINE_LABEL)

    # Step 11: Jenks loss classification (3.6.3)
    clix_df, breaks, gvf = classify_loss(clix_df, JENKS_CLASSES)
    export_loss_classification(clix_df, breaks, gvf, compare_class_solutions(clix_df), output_path)

    # Step 12: loss regimes (3.6.4)
    clix_df = assign_loss_regimes(clix_df, HIGH_TERTILE, LOW_TERTILE)
    export_loss_regime(clix_df, output_path)
    plot_clix_heatmap(clix_df, PIPELINE_LABEL, fig_dir['clix_heatmap'], dataset_label=DATASET_LABEL, study_area=STUDY_AREA)

    # Step 13: dominant loss drivers per block
    driver_table = build_dominant_driver_table(pc_scores_harmonized, pca_input, clix_df, component_names,
                                               EXPOSURE_TOP_ASSETS, DOMINANT_PC_THRESHOLD, USE_ABSOLUTE_SCORES)
    export_dominant_drivers(driver_table, output_path)

    # Step 14: correlation among dimension scores
    rho_matrix, pval_matrix = compute_dimension_correlation(clix_df)
    export_dimension_correlation(rho_matrix, pval_matrix, output_path)
    plot_dimension_correlation(rho_matrix, pval_matrix, PIPELINE_LABEL, fig_dir['dimension_correlation'], n_blocks)

    top = clix_df.sort_values('CLIx_Rank').iloc[0]
    bottom = clix_df.sort_values('CLIx_Rank').iloc[-1]
    print(f"Rank 1: {top['Block']} ({top['District']}), CLIx = {top['CLIx']:.1f}")
    print(f"Rank {bottom['CLIx_Rank']}: {bottom['Block']} ({bottom['District']}), CLIx = {bottom['CLIx']:.1f}")

    # Steps 15-19: sensitivity analysis (3.7)
    if RUN_SENSITIVITY:
        print("Running sensitivity analysis...")
        pipeline_settings = dict(kmo_threshold=KMO_THRESHOLD, min_indicators=KMO_MIN_INDICATORS,
                                 variance_threshold=VARIANCE_RETAINED, loading_threshold=LOADING_THRESHOLD)

        exclusion_results = indicator_exclusion_test(merged_df, dim_map, clix_df, pca_input, **pipeline_settings)
        exclusion_overall, exclusion_by_dim = build_exclusion_summary(
            exclusion_results, SENSITIVITY_RHO_THRESHOLD, EXCLUSION_RETAINED_ONLY)
        retention_df, rho_retention = retention_method_test(merged_df, dim_map, clix_df, pca_input,
                                                            SENSITIVITY_RHO_THRESHOLD, **pipeline_settings)
        weighting_df, rho_weighting = dimension_weighting_test(clix_df, pca_input, dimension_scores)
        regime_detail, regime_summary, kappa = regime_threshold_test(clix_df, HIGH_TERTILE, LOW_TERTILE,
                                                                      QUARTILE_HIGH, QUARTILE_LOW)

        export_sensitivity_results({
            'Test1_IndicatorExclusion': exclusion_results,
            'Test1_IE_Summary': exclusion_overall,
            'Test1_IE_ByDimension': exclusion_by_dim,
            'Test2_RetentionMethod': retention_df,
            'Test3_DimWeighting': weighting_df,
            'Test4_RegimeThreshold': regime_detail,
            'Test4_RegimeThreshold_Sum': regime_summary
        }, output_path)

        print(f"  Test 1, minimum rho: {exclusion_overall['Minimum_rho'].iloc[0]:.3f}")
        print(f"  Test 2, rho (80% vs Kaiser): {rho_retention:.3f}")
        print(f"  Test 3, rho (equal vs EVR weights): {rho_weighting:.3f}")
        print(f"  Test 4, kappa (tertile vs quartile): {kappa:.3f}")

    print(f"Done. Results: {output_path}")
    print(f"Figures: {figure_root}")
    return clix_df


if __name__ == '__main__':
    run_clix()