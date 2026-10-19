"""Comparable revised-DIC plots independent of unpublished reporting changes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_conditional_comparison(output, cases, forward_states, inverse_states, fit_names):
    """Keep the saved comparison in physical units and label registration scope.

    Subtract the common observed component mean only for the displacement
    display, retaining strains, rotations and all prediction residuals.
    """
    rows = []
    for case, forward, inverse in zip(cases, forward_states, inverse_states):
        observed = case.observation.u_measured.reshape(-1, 2)
        xy = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        predicted = [state['prediction'].reshape(-1, 2) for state in (forward, inverse)]
        residuals = [p-observed for p in predicted]
        fields = [observed, *predicted]
        fig, axes = plt.subplots(2, 5, figsize=(16, 5.7), constrained_layout=True)
        for component, label in enumerate(('ux', 'uy')):
            origin = observed[:, component].mean()
            low = min(f[:, component].min()-origin for f in fields)
            high = max(f[:, component].max()-origin for f in fields)
            limit = max(np.max(np.abs(e[:, component])) for e in residuals)
            limit = max(limit, 1e-12)
            for column, values in enumerate(fields + residuals):
                artist = axes[component, column].scatter(
                    *xy.T, c=values[:, component]-(origin if column < 3 else 0),
                    s=4, rasterized=True, cmap='viridis' if column < 3 else 'coolwarm',
                    vmin=low if column < 3 else -limit, vmax=high if column < 3 else limit)
                titles = ('DIC', 'Independent calibration', 'Inverse back-substitution',
                          'Independent - DIC', 'Inverse - DIC')
                axes[component, column].set(title=f'{titles[column]}: {label}',
                                             xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
                axes[component, column].set_title(f'{titles[column]}: {label}', fontsize=9)
                if column in (2, 4):
                    fig.colorbar(artist, ax=list(axes[component, :3] if column == 2 else
                                                 axes[component, 3:]), label='mm', shrink=.7)
        fig.suptitle(f'{case.name}: conditional registration; displacement display centered only', fontsize=11)
        fig.savefig(Path(output) / f'{case.name}_forward_inverse.png', dpi=250)
        fig.savefig(Path(output) / f'{case.name}_forward_inverse.svg')
        plt.close(fig)
        for model, state, error in zip(('forward', 'inverse'), (forward, inverse), residuals):
            rows.append({'case': case.name, 'role': 'fit' if case.name in fit_names else 'prediction',
                         'model': model, 'weighted_rms_mm': state['weighted_rms'],
                         'rmse_ux_mm': np.sqrt(np.mean(error[:, 0]**2)),
                         'rmse_uy_mm': np.sqrt(np.mean(error[:, 1]**2)),
                         'absolute_p95_ux_mm': np.quantile(np.abs(error[:, 0]), .95),
                         'absolute_p95_uy_mm': np.quantile(np.abs(error[:, 1]), .95)})
    table = pd.DataFrame(rows)
    table.to_csv(Path(output) / 'forward_inverse_comparison.csv', index=False)
    return table
