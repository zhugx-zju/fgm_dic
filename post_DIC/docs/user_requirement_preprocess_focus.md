# User Requirement: DIC Preprocessing Focus

This document records the user's current requirements for the maintained DIC preprocessing workflow.

## Core Goal

The project should currently focus on the **DIC data preprocessing stage only**.

The user's target is:

- make the preprocessed displacement-field distribution reasonable
- make the field trend and contour details as consistent as possible with the DIC software output
- keep the workflow aligned with the real experiment setup

At this stage, the project should **not** prioritize later inverse-analysis work.

## Real Experiment Meaning

For the current experiment:

- the **right end is fixed**
- the **left end is loaded in tension**

So any preprocessing step that uses a fixed-end region must treat:

- `right` as the physical fixed-end side

This physical meaning must not be inferred from visual similarity alone.

## Maintained Preprocessing Mainline

The user wants the preprocessing workflow split into two maintained branches:

1. `pixel` data branch
2. `grid` data branch

Both branches should follow the same conceptual preprocessing order:

1. load CSV data
2. reconstruct the 2D structured grid from the exported grid indices
3. remove rigid-body motion using the fixed-end region
4. subtract the selected fixed-point displacement reference
5. apply the 2D grid orientation transform
6. plot the fields

Additional branch-specific requirements:

- the `pixel` branch should still use pixel-style displacement data for cloud-plot visualization
- the `grid` branch should also save processed displacement arrays to `.npy`

## Required CSV Convention

The user clarified that the exported CSV grid-index columns are:

- spreadsheet column `N`: grid row index
- spreadsheet column `O`: grid column index

Recommended visible names:

- `grid_row`
- `grid_col`

The input CSV files are placed in:

- `data/`

The preprocessing outputs should also be written to:

- `data/`

## Meaning Of The Three Preprocessing Steps

The user explicitly distinguished these three ideas and does not want them mixed together:

### 1. Rigid-body-motion removal

This is the step that removes overall rigid translation and rigid rotation from the displacement field.

This step is conceptually aligned with the DIC software's "remove rigid-body displacement" behavior.

### 2. Fixed-point displacement reference

This is an additional project-side reference step.

It is **not** the same as rigid-body-motion removal.

It should happen **after** rigid-body-motion removal.

### 3. Orientation transform

This is the later 2D grid orientation/layout transform used to match the desired plotting or project coordinate convention.

It is **not** the same as rigid-body rotation removal.

It should happen **after** rigid-body-motion removal and fixed-point referencing.

## Comparison Standard

When comparing the code output against the DIC software output, the user cares primarily about:

- field trend
- contour detail
- spatial distribution consistency

The user does **not** primarily care about matching absolute reference offsets.

So the comparison standard should focus on:

- whether the field shape is consistent
- whether local high/low regions and contour trends are consistent

## Current Engineering Preference

The user wants the current rigid-body-removal logic to stay physically tied to the fixed end.

So the maintained code path should currently prefer:

- fixed-end-side-based rigid-body estimation
- `right` as the fixed-end side for the current experiment

The attempted strategy

- `tx, ty` from the right-end strip
- `theta` from the full domain

was tried and then rejected, because it distorted the field distribution.

So that strategy should **not** remain in the maintained preprocessing path.

## Current Non-Goals

The user explicitly does **not** want the current work to focus on:

- DIC-to-FE interpolation
- inverse identification algorithm changes
- later inverse workflow cleanup, except where configuration must remain compatible

Inverse parameters may remain in `script/config.py`, but they are not the main focus of the current preprocessing work.

## Practical Output Requirements

The user expects the maintained preprocessing workflow to provide:

- processed `x/y/ux/uy` arrays for grid data
- displacement cloud plots for both grid and pixel data
- geometry width/height inferred from processed grid data
- average right-edge displacement inferred from processed grid data
- auto-writeback of these inferred values into `script/config.py`

## Current Open Technical Concern

The main unresolved technical concern is:

- how to make the code-side rigid-body-removal result match the DIC software field **detail** better

At present, future debugging should prioritize:

- fixed-end region selection quality
- strip-width-ratio influence
- whether the selected right-end region contains local deformation/noise that harms the rigid-body fit

rather than introducing new global-rotation assumptions too early.
