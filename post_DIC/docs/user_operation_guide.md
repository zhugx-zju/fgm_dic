# User Operation Guide

This document defines the CSV format and the recommended script order for this repository.

## Scope

The current maintained preprocessing workflow focuses on two DIC exports:

- `grid` data: used as the main structured displacement source
- `pixel` data: used for cloud-plot visualization, but processed with the same preprocessing sequence

The current preprocessing sequence is:

1. load CSV
2. rebuild structured 2D arrays from the grid index columns
3. remove rigid-body motion using a fixed-end strip
4. apply fixed-point displacement reference
5. apply 2D grid orientation transform
6. save `npy` arrays and displacement plots

At this stage, this guide does **not** describe the later DIC-to-FE conversion.

## CSV Requirements

### Required column order

The code currently reads the CSV by **column position**, not by matching the header text.

That means the most important requirement is:

- the data columns must stay in the expected order
- the file must contain 15 columns
- the first row is treated as a header or description row and skipped by the loader

The expected column layout is:

| Excel column | Recommended visible name | Internal meaning |
| --- | --- | --- |
| A | `Type:` | export marker column |
| B | `X` | current x coordinate |
| C | `Y` | current y coordinate |
| D | `Z` | current z coordinate |
| E | `DX` | x displacement |
| F | `DY` | y displacement |
| G | `DZ` | z displacement |
| H | `Disp_Combined` | displacement magnitude |
| I | `EXX` | strain component |
| J | `EYY` | strain component |
| K | `EXY` | strain component |
| L | `E1` | principal strain |
| M | `E2` | principal strain |
| N | `grid_row` | row-style grid index |
| O | `grid_col` | column-style grid index |

### Important note about N/O columns

For this project, the N/O columns are the key columns that let us rebuild the 2D structured grid.

Recommended convention:

- column `N`: `grid_row`
- column `O`: `grid_col`

These two columns do **not** have to be literally named `grid_row` and `grid_col` in the CSV header, because the current loader maps them by position.

But for users operating the files in Excel, it is strongly recommended to label them as:

- `grid_row`
- `grid_col`

This keeps the meaning explicit and avoids confusion with the displacement and strain columns.

### Header naming recommendation

If you are exporting or cleaning the CSV manually, use this header row:

```text
Type:,X,Y,Z,DX,DY,DZ,Disp_Combined,EXX,EYY,EXY,E1,E2,grid_row,grid_col
```

If the DIC software exports localized names for displacement magnitude or strain columns, that is acceptable as long as:

- the column order is unchanged
- the N/O index columns are still present

### Data integrity rules

Before running the scripts, check these items:

- `X`, `Y`, `DX`, `DY` must be numeric
- `N`, `O` must be numeric integer-like grid indices
- one CSV should correspond to one single stage
- do not sort only part of the table in Excel
- do not delete the N/O columns

If the N/O columns are removed, the code can no longer rebuild the structured grid reliably.

## File Placement

Recommended input locations in the current config:

- grid CSV: `data/stage_00463.comdat_comdata_grid.csv`
- pixel CSV: `data/stage_00463.comdat_comdata_pixel.csv`

These paths are configured in [script/config.py](/D:/MyProject/post_DIC/script/config.py:49).

If you use different files, update:

- `DIC_EXPORT_BASENAME`

## Config Fields To Edit

Before running the preprocessing scripts, review these settings in [script/config.py](/D:/MyProject/post_DIC/script/config.py:49):

- `DIC_EXPORT_BASENAME`
- `PREPROCESS_FIXED_POINT`
- `PREPROCESS_FIXED_END_SIDE`
- `PREPROCESS_STRIP_WIDTH_RATIO`
- `PREPROCESS_STRIP_Y_MARGIN_RATIO`
- `PREPROCESS_ORIENTATION_OPERATIONS`
- `PREPROCESS_CMAP`

Recommended meaning:

- `PREPROCESS_FIXED_POINT`: the corner used for the final displacement zero reference
- `PREPROCESS_FIXED_END_SIDE`: the physical fixed-end side used for rigid-body removal
- `PREPROCESS_STRIP_WIDTH_RATIO`: width ratio of the fixed-end strip used for rigid-body estimation
- `PREPROCESS_STRIP_Y_MARGIN_RATIO`: optional top/bottom margin removed from the strip before rigid fitting
- `PREPROCESS_ORIENTATION_OPERATIONS`: ordered 2D grid operations such as `["rot180"]`, `["fliplr"]`, or `["rot90", "transpose"]`

### Editable variable options

The user-editable variables in `script/config.py` currently have these options:

| Variable | Meaning | Allowed or typical options |
| --- | --- | --- |
| `DIC_EXPORT_BASENAME` | Base exported DIC filename | Example: `stage_00463.comdat_comdata` |
| `PREPROCESS_FIXED_POINT` | Final fixed-point reference corner | `None`, `top_right`, `bottom_right`, `top_left`, `bottom_left` |
| `PREPROCESS_FIXED_END_SIDE` | Physical fixed-end side for rigid-body removal | `left`, `right` |
| `PREPROCESS_STRIP_WIDTH_RATIO` | Fixed-end strip width ratio | Typical values: `0.05` to `0.20` |
| `PREPROCESS_STRIP_Y_MARGIN_RATIO` | Strip top/bottom margin ratio | Typical values: `0.00` to `0.15` |
| `PREPROCESS_ORIENTATION_OPERATIONS` | Ordered 2D orientation operations | Items from `rot0`, `rot90`, `rot180`, `rot270`, `fliplr`, `flipud`, `transpose` |
| `PREPROCESS_CMAP` | Plot colormap | Typical values: `jet`, `viridis`, `plasma`, `inferno`, `coolwarm` |

If you also use the inverse workflow later, these optional variables are still available:

| Variable | Meaning | Allowed or typical options |
| --- | --- | --- |
| `GEO_L`, `GEO_H` | FE geometry size | positive floats |
| `NEL_X`, `NEL_Y` | FE mesh density | positive integers |
| `NU` | Poisson ratio | typical range `0.0` to `0.49` |
| `E_MIN`, `E_MAX` | Elastic modulus bounds | positive floats with `E_MIN < E_MAX` |
| `GAMMA`, `GAMMA_MIN`, `GAMMA_MAX` | Regularization settings | positive floats |
| `N_GAMMA` | Number of gamma scan points | positive integer |
| `MAX_ITER` | Optimization iteration limit | positive integer |
| `FTOL`, `GTOL` | Optimization tolerances | positive floats |
| `OBSERVATION_SAMPLING_MODE` | How real DIC measurement points are selected for inversion | `uniform_subgrid`, `fe_gauss_nearest` |
| `OBSERVATION_UNIFORM_SUBGRID_NX`, `OBSERVATION_UNIFORM_SUBGRID_NY` | Subgrid size for `uniform_subgrid` sampling | positive integers |
| `OBSERVATION_UX_COMPONENT_WEIGHT`, `OBSERVATION_UY_COMPONENT_WEIGHT` | Relative data-term weights for displacement components | positive floats |
| `TRUE_MODULUS_FILE` | Optional comparison modulus file | `None` or a file path |
| `TENSILE_END_FORCE` | Experimental force | positive float |
| `PRESCRIBED_RIGHT_EDGE_DISPLACEMENT` | Experimental boundary displacement | float |
| `RESULTS_ROOT_DIR` | Inverse results directory | directory name or path |

## Script Execution Order

### Preprocessing only

If your goal is only to preprocess the grid/pixel data, save `npy`, and make displacement plots, run:

```bash
python script/run_preprocess_dic.py
```

This script currently processes both:

- the grid CSV
- the pixel CSV

For each one it will:

- rebuild the 2D grid
- remove fixed-end rigid-body motion
- apply fixed-point reference
- apply the configured 2D orientation transform
- save:
  - `<case_name>_x_grid.npy`
  - `<case_name>_y_grid.npy`
  - `<case_name>_ux_grid.npy`
  - `<case_name>_uy_grid.npy`
  - `<case_name>_ux.png`
  - `<case_name>_uy.png`

Inputs and preprocessing outputs are both written to `data/`.

### Plot regeneration only

If you only want to regenerate the preprocessed displacement plots, run:

```bash
python script/run_plot.py
```

This script re-runs the same preprocessing logic for:

- grid data
- pixel data

and only writes the plot images.

## Recommended User Workflow

For routine use, follow this order:

1. Prepare the grid and pixel CSV files.
2. Confirm the N/O columns are present and represent grid row/column indices.
3. Update [script/config.py](/D:/MyProject/post_DIC/script/config.py:49).
4. Run `python script/run_preprocess_dic.py`.
5. Check the output `npy` files and `ux/uy` plots under `data/`.
6. If the orientation is wrong, adjust:
   - `PREPROCESS_ORIENTATION_OPERATIONS`
7. Run the preprocessing script again until the orientation and reference convention are correct.

## Practical Notes

- The code currently assigns internal names to the 15 columns by position.
- The structured preprocessing path now uses the same names as the user guide for the last two columns: `grid_row` and `grid_col`.
- The visible header text in the CSV is less important than keeping the column order stable.
- For user readability, it is still better to rename N/O as `grid_row` and `grid_col`.
- Grid data and pixel data are both preprocessed through the same rigid-removal, fixed-point, and orientation sequence in the current workflow.
- The intended working directory for both source CSV files and preprocessing outputs is `data/`.
- Grid/pixel input filenames and output case names are derived automatically from `DIC_EXPORT_BASENAME`.
