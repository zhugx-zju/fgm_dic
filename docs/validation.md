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

### `disp_linear`

- Status: planned / in development
- Last checked:
- Notes:

### `disp_nonlinear`

- Status: planned / in development
- Last checked:
- Notes:
