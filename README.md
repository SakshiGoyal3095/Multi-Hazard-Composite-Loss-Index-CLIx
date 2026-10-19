# Composite Loss Index (CLIx)

Python code for the Multi-Hazard Composite Loss Index (CLIx) at block-level.
CLIx combines four dimensions: average annual loss (AAL), exposure, vulnerability and capacity deficit.
CLIx is run independently on earthquake- and seismic-induced landslide hazard dataset (CLIx-EQ) and flood- and precipitation-induced landslide hazard dataset (CLIx-FL).

The code accompanies the paper:

> Goyal, S., Mukherjee, M., & Roy, S. (under review). Multi-Hazard Composite Loss Index for Sub-District Prioritisation and Targeted Interventions: Application to Seismic and Hydro-Meterological Hazards in the Indian Himalaya. *Ecological Indicators*.

The test case covers 95 development blocks in Uttarakhand, India, with two implementations: CLIx-EQ (earthquake and seismic-induced landslides) and CLIx-FL (fluvial flood, flash flood and precipitation-induced landslides).

## Requirements

- Python 3.13 (tested)
- Packages listed in `requirements.txt`

Install the packages:
```
py -3.13 -m pip install -r requirements.txt
```

## Repository contents

| File | Paper section | Purpose |
|---|---|---|
| `run_Step0-19_CLIx.py` | – | Runs all steps in order. Edit only its SETTINGS block. |
| `CLIx_Step0_LoadingData.py` | 3.2 | Loads the four dimension sheets |
| `CLIx_Step1_ZeroVarianceCheck.py` | 3.3.1 | Drops zero-variance indicators |
| `CLIx_Step2_SkewnessKurtosisCheck.py` | 3.3.2 | Winsorisation and log-transformation |
| `CLIx_Step3_OutlierDetection.py` | 3.3.3 | Flags outliers (3×IQR, non-zero values) |
| `CLIx_Step4_Standardisation.py` | 3.3.4 | Z-score standardisation |
| `CLIx_Step5_SpearmanCorrelation.py` | 3.4 | Spearman correlation matrices |
| `CLIx_Step6_KMOTest.py` | 3.4.1 | Iterative KMO adequacy testing |
| `CLIx_Step7_BartlettTest.py` | 3.4.2 | Bartlett's test of sphericity |
| `CLIx_Step8_PCA.py` | 3.5 | PCA, Varimax rotation, PC scores, sign harmonisation |
| `CLIx_Step9_WithinDimensionWeights.py` | 3.6.1 | EVR weights and dimension scores |
| `CLIx_Step10_CLIxScoreComputation.py` | 3.6.2 | CLIx score, rescaling and ranking |
| `CLIx_Step11_LossClassification.py` | 3.6.3 | Jenks natural breaks classification |
| `CLIx_Step12_LossRegimeClassification.py` | 3.6.4 | Loss regime classification |
| `CLIx_Step13_DominantLossDriversPerBlock.py` | – | Dominant loss drivers per block |
| `CLIx_Step14_DimensionCorrelation.py` | – | Correlation among dimension scores |
| `CLIx_Step15_SensitivityCommon.py` | 3.7 | Shared functions for the sensitivity tests |
| `CLIx_Step16` to `CLIx_Step19` | 3.7 | Four sensitivity tests |
| `input_template_CLIx-EQ.xlsx` | 3.2 | Input template, CLIx-EQ indicators |
| `input_template_CLIx-FL.xlsx` | 3.2 | Input template, CLIx-FL indicators |

## How to run

1. Fill an input template with your data. Each template has an Instructions sheet and an Indicators sheet.
2. Save it in the same folder as the code.
3. Open `run_Step0-19_CLIx.py` and set, in the SETTINGS block:
   - `INPUT_FILE`: the name of your filled template
   - `PIPELINE_LABEL`: for example `CLIx-EQ` or `CLIx-FL`
   - `STUDY_AREA` and `DATASET_LABEL` (optional, used in figure titles)
4. Run:
```
py -3.13 run_Step0-19_CLIx.py
```

A full run, including the sensitivity analysis, takes a few minutes. Set `RUN_SENSITIVITY = False` for a faster run.

## Input format

Four sheets named `AAL`, `Exposure`, `Vulnerability` and `Capacity_Deficit`. Columns A and B are `District` and `Block`, then one numeric column per indicator. Each indicator must be prepared so that a higher value means higher average annual loss, exposure, vulnerability or capacity deficit.

## Outputs

Each run creates a folder `output_<PIPELINE_LABEL>` containing:

- `<PIPELINE_LABEL>_Results.xlsx`: every intermediate and final result, one sheet per step, plus a `Run_Settings` sheet recording the settings used
- `Figures/`: one subfolder per step (histograms, boxplots, scree plots, loading heatmaps, CLIx histogram, CLIx heatmap, dimension correlation)

## Settings

All thresholds are set in the SETTINGS block of `run_Step0-19_CLIx.py`. The defaults are the values used in the paper (e.g. KMO ≥ 0.5, 80% cumulative variance, |loading| ≥ 0.40, six Jenks classes, tertile thresholds 0.67 and 0.33).

## Notes

- Principal component names are a manual interpretation of the loadings. The code leaves the `PC Name` column of the `Retained_Loadings` sheets blank for the user to fill. Filled names can be read back through `COMPONENT_NAMES_FILE`.
- CLIx scores are relative to the units assessed in a run. Scores from separate runs or regions are not directly comparable.

## Data availability

The input data for the Uttarakhand test case were derived from the Uttarakhand Disaster Risk Atlas 2019 and the Census of India 2011, and are available from the corresponding author on request. The templates contain indicator names only, not data.

## Citation

If you use this code, please cite the paper above and the archived code:

> Goyal, S., Mukherjee, M., & Roy, S. (2026). CLIx: Multi-Hazard Composite Loss Index (v1.0). Zenodo. https://doi.org/10.5281/zenodo.23480029

## Licence

MIT License. See `LICENSE`.
