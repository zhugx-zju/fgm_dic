# Validation Notes

This file is a lightweight validation checklist and record template for the research branches in this repository.

## How To Use

- Keep the checklist concise.
- Record only the tests that matter for the current branch.
- Prefer short statements with dates, case names, and outcome summaries.
- If a test fails, keep the failure note instead of deleting it.

## Common Record Template

Use this structure when documenting a branch:

```text
Branch:
Date:
Code revision:

Scope:

Validation cases:
- Case:
  Purpose:
  Expected behavior:
  Observed behavior:
  Status:

Open issues:
- ...
```

## Recommended Minimum Validation By Branch

### `main`

Suggested checks:

- forward linear solve runs successfully on the default case
- reaction forces balance the applied force
- regularization gradient finite-difference check passes
- inverse baseline case reruns reproducibly with the saved configuration
- plotting scripts can reload the latest saved results

### `disp_linear`

Suggested checks:

- prescribed-displacement boundary condition is applied to the intended DOFs
- reaction-force output is reasonable and sign-consistent
- for a simple comparable case, displacement-controlled and force-controlled results are mutually consistent
- small changes in prescribed displacement scale the response linearly
- saved output files and plots are still readable by the branch plotting scripts

### `disp_nonlinear`

Suggested checks:

- load stepping runs without solver breakdown on a simple benchmark
- Newton or nonlinear iterations show consistent residual reduction
- constitutive update behaves correctly on a single-element or single-point benchmark
- finite-strain implementation reproduces the linear branch in the small-deformation limit when applicable
- reaction force versus imposed displacement curve is physically plausible
- state-variable updates are repeatable after reruns from the same input

## Current Status Template

### `main`

- Status: baseline branch
- Last checked:
- Notes:

### `force_bilateral_traction`

- Status: two-end traction boundary condition and existing nodal-modulus inverse solver connected
- Last checked: 2026-10-14
- Checks: edge resultants, homogeneous analytic displacement, gauge reactions, adjoint gradient finite difference
- Executed: `python -m unittest discover -s tests -v` (5 passed); default `55 x 10` forward solve (finite displacement, maximum gauge reaction `3.1e-16`); small-mesh inverse smoke run reduced objective from `3.41e-5` to `5.58e-10` in 20 iterations (iteration cap reached)
- Open: paired full-specimen boundary-error study, DIC gauge registration, seven-zone parameterization and identifiability
- Synthetic `E(x)` case: `EY=1.0`, `Ex=2.0`, `55 x 10` mesh, `F=0.01 N`, `gamma=1e-6`, zero added noise; the original 616-parameter `E(x,y)` inverse converged in 342 iterations. Full-field MAE `0.2123%`, y-mean x-profile MAE `0.1779%`, reconstructed transverse RMS spread `0.1709%` relative to mean true modulus. Largest local relative error `2.7182%`, concentrated near the loaded ends. The fixed-gamma workflow ran end to end; the 50-point L-curve workflow was not rerun.
- Boundary-error diagnosis, same synthetic displacement, mesh, initial field, and `gamma=1e-6`: the original `gradient` penalty had mean absolute relative error `0.6493%` at `x=0..10 mm`, `0.02570%` at `x=20..90 mm`, and `0.9571%` at `x=100..110 mm`. Its gradient penalty was about 44 times its data term at the final iterate. Lowering `gamma` to `1e-8` reduced the corresponding end errors to `0.1706%` and `0.1805%`, but setting `gamma=0` did not converge within 2000 iterations and produced large 2D field errors. These checks support regularization bias as a major cause, while not proving unique recovery from displacements.
- Affine-detrended gradient penalty: `R(E)=integral[(E_x-mean(E_x))^2+(E_y-mean(E_y))^2] dA`. It retains all 616 nodal inverse unknowns and allows nonzero constant gradients in both directions. Unit checks cover its affine nullspace and finite-difference gradient; a separate finite-difference check covers the full adjoint objective in log modulus. The fixed-gamma zero-noise run converged in 306 iterations: full-field MAE `8.53e-6%`, left-end MAE `1.67e-5%`, right-end MAE `7.67e-6%`, maximum error `1.53e-4%`. The result is especially favorable because the synthetic truth is exactly in the penalty's nullspace. A single exploratory 1% noise realization (seed 42) reduced end MAE from `0.6588%`/`3.0070%` to `0.2550%`/`2.3648%` and full-field MAE from `1.2050%` to `1.0975%`; this is not a multi-seed robustness result. Seven-zone modulus profiles, DIC noise/correlation, and full L-curve selection remain untested with this penalty. The original result was preserved in `noise_0.00pct/`; the new run is in `noise_0.00pct_affine_detrended_fixed/`.
- A five-point small-mesh L-curve smoke scan (`4 x 2` elements, zero noise) gave identical residual and penalty norms at all gamma values after the first solve, hence zero discrete curvature. This scan contains no usable corner. The L-curve routine now raises an actionable error on flat/invalid curvature instead of reporting an arbitrary middle gamma. The full `55 x 10`, 50-point L-curve was not run; `gamma=1e-6` is the fixed value used for the comparison above.
- Exponential-profile check (`python -m script.benchmark_exponential_regularization`): same two-end force (`0.01 N` per end), thickness (`3.2 mm`), `55 x 10` mesh, independent nodal `E(x,y)` unknowns, common initial field, and fixed `gamma=1e-6` for both penalties. For true `E(x)=exp[ln(2)x/110] MPa` with zero noise, original-gradient vs affine-detrended full-field MAE was `0.2256%` vs `0.07086%`; left-end (`0..10 mm`) `0.4539%` vs `0.1846%`; right-end (`100..110 mm`) `1.2699%` vs `0.3426%`. Both optimizers converged. With one 1% displacement-noise draw (seed 42), full MAE was `1.2886%` vs `1.2123%` and right-end MAE `3.3571%` vs `2.7145%`. Results, identical measured displacement per comparison, logs, fields, and common-scale PNG/PDF figures are saved in ignored `Geo_*/benchmark_exponential_55x10_zero/` and `Geo_*/benchmark_exponential_55x10_noise1_seed42/` folders.
- Stronger exponential-profile boundary case: `E(x)=exp[ln(8)x/110] MPa`, same zero-noise protocol and `gamma=1e-6`, gave full MAE `4.4167%` vs `2.7853%`, with right-end MAE `23.7202%` vs `14.3102%`; the left-end MAE worsened from `1.4291%` to `2.1444%`. Thus the affine-detrended penalty improves the whole field in this controlled comparison but does not adequately recover the high-curvature end at the current gamma, nor improve every region. An exploratory `gamma=1e-8` rerun reduced full MAE to `0.6841%` vs `0.4828%`, with right-end MAE `4.9719%` vs `2.8976%`; this gamma was inspected using known synthetic truth and is **not** an independently selected parameter. The two strong-contrast runs are saved under ignored `Geo_*/benchmark_exponential_55x10_8mpa_zero/` and `Geo_*/benchmark_exponential_55x10_8mpa_gamma1e-8_zero/`.

### `disp_linear`

- Status: planned / in development
- Last checked:
- Notes:

### `disp_nonlinear`

- Status: planned / in development
- Last checked:
- Notes:
