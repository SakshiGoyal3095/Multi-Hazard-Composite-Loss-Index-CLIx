import pandas as pd
from sklearn.metrics import cohen_kappa_score

from CLIx_Step12_LossRegimeClassification import compute_percentile_ranks, classify_loss_regime

def regime_threshold_test(baseline_clix, baseline_H=0.67, baseline_L=0.33, alternative_H=0.75, alternative_L=0.25):
    df = compute_percentile_ranks(baseline_clix)
    df['Regime_Tertile'] = df.apply(lambda r: classify_loss_regime(r, H=baseline_H, L=baseline_L), axis=1)
    df['Regime_Quartile'] = df.apply(lambda r: classify_loss_regime(r, H=alternative_H, L=alternative_L), axis=1)
    kappa = cohen_kappa_score(df['Regime_Tertile'], df['Regime_Quartile'])

    detail = df[['District', 'Block', 'Regime_Tertile', 'Regime_Quartile']].copy()
    detail['Changed_Classification'] = detail['Regime_Tertile'] != detail['Regime_Quartile']
    summary = pd.DataFrame([{
        'Kappa_Tertile_vs_Quartile': kappa,
        'Blocks_Changed': int(detail['Changed_Classification'].sum()),
        'Total_Blocks': len(detail)
    }])
    return detail, summary, kappa


## Note (Test 4, regime thresholds): quartiles (H = 0.75, L = 0.25) replace tertiles in Step 12. Cohen's kappa above 0.60 means substantial agreement (Landis & Koch, 1977). A quintile threshold was also tested earlier and removed after a borderline kappa.