# Next-Step Plan For DIC Data Review

## Purpose

This note records the recommended next steps for analyzing the DIC-exported data before further modifying the inverse workflow.

The current project context is already sufficient to proceed. The main remaining uncertainty is not the FE solver itself, but the exact meaning of the DIC exports and how they should be mapped into the inverse-analysis coordinate and boundary-condition convention.

One practical distinction should already be kept explicit:

- pixel-data exports are primarily for cloud-plot visualization and diagnostic comparison
- grid-data exports are the preferred source for inverse-analysis input


## Current Understanding

The current inverse workflow assumes:

- a rectangular FE domain
- left edge fixed
- right edge prescribed with positive `x` displacement
- DIC displacement data aligned into that FE convention before interpolation

The current DIC preprocessing uses:

- reference-coordinate recovery with `X_ref = X - DX`, `Y_ref = Y - DY`
- fixed-point displacement referencing
- explicit `180` degree rotation for DIC-to-FE orientation alignment

This works as a transitional workflow, but several questions about the DIC exports still need to be verified more carefully.


## Main Questions Still To Be Analyzed

### 1. Which DIC export should be treated as the formal inverse input?

The project already indicates that:

- exported grid data is more suitable as inverse input
- exported pixel data is more suitable for visualization and diagnosis

However, the user still needs a clearer interpretation of:

- what exactly the DIC software means by "grid data"
- what exactly the DIC software means by "pixel data"
- whether the pixel field is a resampled display field or a truly independent measurement field
- whether rigid-body-removal exports preserve the coordinate meaning required by the current code


### 2. What does the DIC software change when rigid-body motion is removed?

This must be clarified carefully.

The remaining uncertainty is whether the software export after rigid-body removal changes:

- only displacement values
- displacement values and internal display orientation
- the effective coordinate meaning of the exported field

This matters because the current project uses the displacement field to reconstruct reference coordinates.


### 3. Does the current `fixed-point + 180 degree rotation` workflow reflect a physical alignment or a software-specific workaround?

This question should be answered explicitly.

The current implementation likely works because it forces the DIC data into the FE solver's convention, but it is still necessary to distinguish:

- physical specimen orientation
- DIC software coordinate orientation
- FE solver coordinate orientation


### 4. How much of the specimen is truly measured?

The current workflow assumes a rectangular region that becomes the FE domain size.

But in reality:

- the measured ROI may exclude edge regions
- the full specimen length may not be directly observed
- the right-edge prescribed displacement currently extracted from DIC may not correspond to the true physical loading boundary if the ROI is incomplete

This has to be analyzed before treating the DIC-derived right-edge displacement as a strict FE boundary condition.


## Recommended Next-Step Investigation

### Step 1. Collect a minimal diagnostic export set

The user does not need to use all exports in the formal inverse workflow, but it is useful to collect them once for diagnosis.

Recommended diagnostic set:

1. raw grid-data CSV
2. rigid-body-removed grid-data CSV
3. raw pixel-data CSV
4. rigid-body-removed pixel-data CSV

This is useful because it allows direct comparison of:

- grid versus pixel sampling
- raw versus rigid-body-removed displacement
- whether coordinate columns remain unchanged
- whether the rigid-body removal acts like constant translation removal or full rigid-body alignment

Working rule during this diagnosis:

- use pixel-data exports to compare DIC software cloud plots and exported display fields
- use grid-data exports to analyze measurement coordinates, displacement values, and inverse-input suitability


### Step 2. Compare the coordinate columns across the four exports

The first concrete check should be:

- compare `X`, `Y` from raw grid data and rigid-body-removed grid data
- compare `X`, `Y` from raw pixel data and rigid-body-removed pixel data

Questions to answer:

- are `X`, `Y` identical before and after software rigid-body removal?
- do only `DX`, `DY` change?
- is the exported ROI extent identical?

If `X`, `Y` stay the same and only `DX`, `DY` change, then software rigid-body removal is easier to interpret in the current pipeline.


### Step 3. Compare raw and rigid-body-removed displacement differences

For both grid and pixel exports, compute:

- `delta_DX = DX_removed - DX_raw`
- `delta_DY = DY_removed - DY_raw`

Then inspect whether these differences are:

- nearly constant over the field
- approximately affine in space
- clearly rotation-like

This will reveal whether the software is performing:

- translation removal only
- translation plus rotation removal
- a more complicated transformation


### Step 4. Verify physical edge meaning in the exported coordinate system

This step is critical.

For the raw grid data, verify:

- which side corresponds to the physical fixed end
- which side corresponds to the physical loading end
- whether increasing `X` means moving from fixed end to loading end, or the opposite
- whether `DX > 0` corresponds to the tensile direction used by the FE solver

This should be checked directly from:

- raw displacement cloud plots
- a line profile through the specimen center
- the known test setup


### Step 5. Evaluate whether the measured ROI reaches the physical ends

The raw grid export should be used to determine:

- whether the left and right physical edges are fully present in the measured field
- whether the ROI truncates the specimen before the clamps or edge boundaries
- whether the currently extracted "right-edge average displacement" is actually a boundary displacement or only an interior ROI displacement

If the ROI does not reach the true loaded edge, then the present FE boundary-condition mapping should be treated as an approximation.


### Step 6. Decide the formal inverse-input policy

After the above checks, the project should adopt one explicit policy.

Recommended policy candidate:

- use raw grid-data CSV as the formal inverse input
- use rigid-body-removed grid-data CSV only as a diagnostic reference
- use pixel data only for cloud plots, visualization, and consistency checks

This should remain the default unless the diagnostic comparison reveals a strong reason to change it.


## Why This Plan Matters

At the moment, the main risk is not simply interpolation error. The larger risk is using a DIC export whose coordinate or displacement meaning is misunderstood.

If that happens, then:

- `fixed-point` correction may be applied on the wrong semantic basis
- `180` degree rotation may compensate for a software convention rather than a physical convention
- `X_ref = X - DX`, `Y_ref = Y - DY` may lose physical meaning
- FE boundary conditions may be mapped to the wrong measured edge

Therefore, the next priority should be data-meaning verification, not immediate solver modification.


## Recommended Practical Outcome

Once the above investigation is complete, the project should be able to state clearly:

- which export is the formal inverse input
- whether software rigid-body removal is allowed in the formal workflow
- that pixel data is for cloud-plot/display use and not the default inverse input
- that grid data is the default inverse input used for FE comparison
- what `fixed-point` means in that workflow
- what `180 degree rotation` means in that workflow
- whether the current right-edge displacement extraction is physically justified
- whether ROI-based residual masking must be implemented next


## Immediate Follow-Up

After this data-review step, the next technical implementation task should likely be one of the following:

1. add explicit ROI masking to the inverse residual
2. revise the DIC-to-FE alignment workflow so that the coordinate meaning is cleaner
3. replace full-domain nodal matching with measurement-point-based comparison

The correct order depends on the outcome of the DIC export analysis above.
