# post_DIC

`post_DIC` is a DIC post-processing project for CSV files exported by the measurement software.

The current maintained workflow is centered on **structured preprocessing** of two DIC exports:

- `grid` data: the main structured displacement source
- `pixel` data: used for cloud-plot visualization, but processed through the same preprocessing sequence

The current maintained preprocessing sequence is:

1. load CSV
2. rebuild the 2D grid from the exported grid-index columns
3. remove rigid-body motion using a fixed-end strip
4. apply fixed-point displacement reference
5. apply a 2D grid orientation transform
6. save `npy` arrays and displacement plots

For a user-facing CSV format and script-order guide, see [docs/user_operation_guide.md](/D:/MyProject/post_DIC/docs/user_operation_guide.md).

## Current Status

This repository also contains an inverse-analysis workflow under `fgm_asm/`.

That inverse workflow now uses a pointwise observation-operator formulation based on
real DIC measurement points, rather than a legacy DIC-to-FE nodal interpolation path.

The current maintained entrypoints are:

- [script/config.py](/D:/MyProject/post_DIC/script/config.py:1)
- [script/run_preprocess_dic.py](/D:/MyProject/post_DIC/script/run_preprocess_dic.py:1)
- [script/run_plot.py](/D:/MyProject/post_DIC/script/run_plot.py:1)

## Project Structure

```text
post_DIC/
|- README.md
|- AGENTS.md
|- data/
|- docs/
|  `- user_operation_guide.md
|- lib/
|  |- dic_grid_orientation.py
|  |- dic_grid_utils.py
|  |- dic_preprocessing.py
|  |- fixed_point_reference.py
|  |- io_utils.py
|  |- processing_utils.py
|  `- rigid_body_reference.py
|- script/
|  |- config.py
|  |- run_plot.py
|  `- run_preprocess_dic.py
`- fgm_asm/
```

## Input And Output

### Input

The current preprocessing workflow typically uses:

- `data/<case>_grid.csv`
- `data/<case>_pixel.csv`

The exact file paths are configured in [script/config.py](/D:/MyProject/post_DIC/script/config.py:49):

- `DIC_EXPORT_BASENAME`

### Output

Preprocessing outputs are currently written to:

- `data/`

The intended working convention is:

- input CSV files are placed in `data/`
- preprocessing output `npy` files are written to `data/`
- preprocessing cloud plots are written to `data/`

For each configured case, the workflow writes:

- `<case_name>_x_grid.npy`
- `<case_name>_y_grid.npy`
- `<case_name>_ux_grid.npy`
- `<case_name>_uy_grid.npy`
- `<case_name>_ux.png`
- `<case_name>_uy.png`

## Core Data Convention

The loader reads the CSV by **column position**.

The maintained structured layout is:

- columns `B` to `M`: coordinates, displacement, and strain values
- column `N`: grid row index
- column `O`: grid column index

For user readability, the recommended visible names for the last two columns are:

- `grid_row`
- `grid_col`

Inside the current Python loader, those two columns are mapped directly as `grid_row` and `grid_col` in [lib/io_utils.py](/D:/MyProject/post_DIC/lib/io_utils.py:1).

Reference coordinates are restored by:

- `X_ref = X - DX`
- `Y_ref = Y - DY`

After that, the maintained preprocessing applies:

1. fixed-end strip rigid-body removal
2. fixed-point reference
3. 2D array orientation operations such as `rot180`, `fliplr`, `flipud`, or transpose

## Maintained Workflow

The maintained preprocessing workflow is:

1. Edit [script/config.py](/D:/MyProject/post_DIC/script/config.py:49)
2. Run `python script/run_preprocess_dic.py`
3. Check the generated `npy` files and `ux/uy` plots under `data/`
4. If needed, adjust orientation-related settings in `config.py` and run again

If you only want to regenerate plots, run:

```bash
python script/run_plot.py
```

## Entry Scripts

### `script/config.py`

Central user-editable configuration file.

It currently contains:

- DIC export base name
- fixed-point selection
  - use `None` to skip fixed-point displacement referencing
- fixed-end side selection
- rigid-strip width and optional top/bottom strip margin
- orientation operation list
- plot colormap setting

### `script/run_preprocess_dic.py`

This is the maintained preprocessing export entrypoint.

It:

- loads the grid CSV
- loads the pixel CSV
- rebuilds the structured 2D arrays
- removes rigid-body motion from the configured fixed-end strip
- applies fixed-point reference
- applies the configured orientation transform
- saves `npy` arrays
- saves `u_x` and `u_y` plots

### `script/run_plot.py`

Re-runs the same preprocessing logic and regenerates the displacement plots for:

- grid data
- pixel data

## Main Library Files

### `lib/dic_preprocessing.py`

Implements the current structured preprocessing pipeline.

It handles:

- loading cleaned CSV data
- rebuilding the structured grid
- rigid-body removal from a fixed-end strip
- fixed-point referencing
- 2D orientation transform
- saving `npy` arrays and plots

### `lib/dic_grid_utils.py`

Rebuilds structured 2D arrays from the exported grid-index columns.

### `lib/rigid_body_reference.py`

Implements fixed-end strip rigid-body removal, including:

- fixed-end strip selection
- rigid `tx, ty, theta` least-squares mode
- optional top/bottom strip-margin trimming

### `lib/fixed_point_reference.py`

Implements fixed-point displacement referencing. The maintained preprocessing path currently uses the grid-form API.

### `lib/dic_grid_orientation.py`

Implements grid-oriented topology transforms such as:

- `rot90`
- `rot180`
- `rot270`
- `fliplr`
- `flipud`
- transpose

### `lib/io_utils.py`

Provides CSV read/write helpers and maps the 15 exported CSV columns into the internal dataframe representation.

### `lib/processing_utils.py`

Provides raw CSV loading, cleaning, and reference-coordinate restoration used by the maintained preprocessing path.

## Notes

- The most important CSV requirement is stable column order.
- The `N/O` grid-index columns must be preserved.
- Grid data and pixel data are currently preprocessed through the same rigid-removal, fixed-point, and orientation sequence.
- The fixed-end side for rigid-body removal is now configured explicitly and is independent from the fixed-point reference corner.
- The repository still contains inverse-analysis code, but that is separate from the maintained preprocessing workflow described here.
