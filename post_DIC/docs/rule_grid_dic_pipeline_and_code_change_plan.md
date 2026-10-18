# Rule-Grid DIC Pipeline And Code Change Plan

## Goal

This document defines a cleaner processing path for **DIC exported grid data** and summarizes the corresponding code changes needed in this repository.

The purpose is to support later code modification work with a concise but complete technical reference.


## Core Decision

For this project, DIC exported data should be handled through two distinct paths:

- **grid-data path**: for inverse analysis
- **pixel-data path**: for cloud plots, visualization, and diagnosis

They should not share the same preprocessing logic by default.


## Why A Separate Grid-Data Path Is Needed

The current preprocessing logic treats DIC input mostly as scattered data.

That is not the best fit for exported grid data because:

- grid data already has a regular row-column structure
- orientation correction can be handled as 2D array reordering
- using scattered interpolation too early discards useful topology
- the current flow is harder to interpret physically

For grid data, the coordinate-alignment problem is not primarily a generic geometric-rotation problem. In practice, it is mostly:

- local rigid-body removal near the fixed end
- fixed-point displacement referencing for boundary-condition consistency
- 2D array rotation / flipping / axis swapping


## Recommended Grid-Data Pipeline

The recommended formal pipeline for DIC exported grid data is:

1. load raw grid CSV
2. rebuild the 2D grid structure
3. estimate and remove rigid-body motion from a fixed-end strip
4. apply fixed-point referencing to match the inverse framework boundary convention
5. rotate / flip / transpose the 2D arrays into FE orientation
6. convert the aligned grid field into FE comparison data
7. run inverse analysis

Short form:

```text
raw grid
-> fixed-end local rigid-body removal
-> fixed-point boundary-condition alignment
-> 2D array orientation transform
-> FE-aligned displacement field
-> inverse solver
```


## Physical Meaning Of Each Step

### 1. Raw Grid Input

Use the raw grid export as the formal inverse-input source.

Reason:

- it preserves the original displacement field
- it avoids software-side hidden rigid-body assumptions as the default formal input
- it is easier to track boundary-condition meaning from this starting point


### 2. Fixed-End Local Rigid-Body Removal

The raw field may not visibly show "one end fixed, one end stretched" because it contains:

- overall translation
- small overall rotation

This step should use a strip near the physical fixed end to estimate rigid-body motion.

Two levels are possible:

- translation-only estimate using strip-average `DX, DY`
- rigid estimate using `tx, ty, theta`

Preferred long-term option:

- fit `tx, ty, theta` from the fixed-end strip by least squares

This step should remove the bulk motion while preserving the deformation gradient associated with tension.


### 3. Fixed-Point Referencing

This step must remain in the project, but its role should be stated correctly.

Its role is:

- not the main rigid-body-removal method
- but the final displacement reference alignment step used to match the inverse framework boundary convention

So fixed-point referencing should happen **after** the local rigid-body-removal step.


### 4. 2D Array Orientation Transform

For regular grid data, orientation correction should be handled mainly by 2D array operations.

Typical operations:

- `flipud`
- `fliplr`
- transpose
- optional sign change of displacement components when the physical coordinate basis is rotated

This is the preferred way to implement the current `180` degree orientation mapping for grid data.

Important distinction:

- for grid data, orientation correction is primarily an array-topology problem
- for scattered/pixel data, orientation correction remains a generic coordinate-transform problem


### 5. FE-Aligned Displacement Field

After the previous steps, the project should work with an intermediate field whose meaning is explicit:

- fixed end corresponds to the FE fixed boundary
- loading end corresponds to the FE prescribed-displacement boundary
- tensile direction corresponds to positive FE `x`
- displacement zero level matches the inverse framework boundary convention

This is the field that should be used for FE comparison and inverse-input preparation.


## Recommended Interpretation Of Pixel Data

Pixel data should not be the default inverse input.

Its preferred uses are:

- cloud-plot rendering
- visual comparison against the DIC software display
- diagnosis of orientation and rigid-body-removal effects

Short rule:

- **grid data**: inverse analysis
- **pixel data**: visualization and diagnosis


## Recommended Code Refactor Direction

The repository should gradually separate the processing logic into:

- a **grid-data inverse path**
- a **pixel/scattered visualization path**

This does not require rewriting everything at once.


## Required Code-Level Changes

### A. Add A Grid-Aware Loader

Current issue:

- CSV loading reads data as flat rows only

Needed change:

- add a loader that reconstructs the exported grid as structured 2D arrays

Recommended new functionality:

- parse `I, J` or equivalent row-column indices when available
- rebuild:
  - `X_grid`
  - `Y_grid`
  - `UX_grid`
  - `UY_grid`
- retain flat-vector exports only as derived data, not as the primary internal representation

Suggested file:

- new helpers in `lib/processing_utils.py` or a new module such as `lib/dic_grid_utils.py`


### B. Add Fixed-End Strip Rigid-Body Removal

Current issue:

- the project only has single-point fixed-point subtraction

Needed change:

- add a preprocessing function that estimates rigid-body motion from a strip near the fixed end

Recommended capability:

- translation-only mode
- `tx, ty, theta` least-squares mode

Recommended output:

- corrected `UX_grid`
- corrected `UY_grid`
- fitted rigid parameters
- metadata describing the strip used

Suggested location:

- new module such as `lib/rigid_body_reference.py`


### C. Keep Fixed-Point Referencing, But Move It Later In The Pipeline

Current issue:

- fixed-point subtraction is currently the main maintained correction

Needed change:

- keep it
- but apply it after local rigid-body removal

Meaning in the code:

- fixed-point remains a required step for inverse boundary-condition consistency
- but it should no longer be described as the main rigid-body-removal mechanism

Suggested impact:

- update `lib/fixed_point_reference.py`
- update calling order in `script/run_export_fixed_point_corrected_csv.py`
- update calling order in `lib/dic_inverse_input.py`


### D. Add Grid-Oriented Orientation Transform

Current issue:

- `align_dic_to_fe_axes` is designed around generic coordinate rotation and scattered data

Needed change:

- add a grid-specific orientation function operating directly on 2D arrays

Recommended functionality:

- rotate `180` degrees by array reordering
- flip horizontally / vertically
- transpose axes when needed
- rotate displacement components consistently with the chosen orientation mapping

Suggested location:

- new module such as `lib/dic_grid_orientation.py`


### E. Separate Grid Path From Scatter Path

Current issue:

- the project currently mixes regular-grid and scattered-data concepts too early

Needed change:

- add explicit branching in the preprocessing workflow

Recommended split:

- grid path:
  - grid reconstruction
  - fixed-end rigid removal
  - fixed-point reference
  - 2D array orientation correction
  - FE conversion
- pixel/scatter path:
  - visualization-oriented coordinate handling
  - optional generic scattered interpolation

Suggested impact:

- revise `lib/dic_inverse_input.py`
- revise `script/run_prepare_dic_inverse_input.py`


### F. Delay Scattered Interpolation Until It Is Truly Needed

Current issue:

- the current workflow converts aligned data into scattered interpolation early

Needed change:

- preserve 2D grid form through most preprocessing steps
- only interpolate when converting to FE comparison data

This reduces:

- unnecessary interpolation error
- loss of grid topology
- ambiguity in orientation reasoning


### G. Revisit FE Comparison Construction

Current issue:

- current workflow builds a full-domain `U_measured` by interpolation to FE nodes

Short-term acceptable path:

- continue building FE nodal input from the aligned grid field

Recommended improvement:

- only use ROI-supported locations in the data term
- later move to FE evaluation at measurement points instead of full nodal replacement

This is a later step, not the first refactor priority.


## Immediate Implementation Order

The most practical code-modification order is:

1. add grid reconstruction utilities
2. add fixed-end rigid-body-removal preprocessing
3. keep fixed-point referencing after rigid removal
4. add grid-specific orientation transform
5. update inverse-input preparation to use the new grid path
6. leave the pixel/scatter path mainly for plotting and diagnosis


## Minimal Design Requirements For New Functions

Any new grid-data preprocessing functions should expose enough metadata for later debugging.

Recommended metadata:

- chosen fixed-end strip bounds
- fitted rigid-body parameters `tx, ty, theta`
- chosen fixed point
- orientation operations applied
- whether sign changes were applied to displacement components

This is important because the user needs the preprocessing meaning to remain auditable.


## Recommended Final Internal Data Representation

For the grid-data inverse path, the preferred intermediate representation is:

```text
X_grid
Y_grid
UX_grid
UY_grid
metadata
```

where `metadata` records:

- rigid-body-removal parameters
- fixed-point information
- orientation-transform history

The flattened FE input vector should be treated as a downstream product, not as the main representation of DIC data.


## Summary

The project should move away from a single generic "DIC as scattered points" workflow.

For inverse analysis with exported grid data, the better approach is:

- preserve the regular grid structure
- remove rigid motion using the fixed-end neighborhood
- use fixed-point referencing to enforce boundary-condition consistency
- implement orientation correction as array operations
- only then convert to FE comparison data

This is the most direct path to making the preprocessing:

- physically interpretable
- easier to debug
- better matched to the current inverse framework
