# Rigid-Body Removal Analysis For `stage_00463`

## Files Compared

The following DIC exports were compared:

- `data_analysis/stage_00463.comdat_comdata_grid.csv`
- `data_analysis/stage_00463.comdat_comdata_grid_remove.csv`
- `data_analysis/stage_00463.comdat_comdata_pixel.csv`
- `data_analysis/stage_00463.comdat_comdata_pixel_remove.csv`

The main quantitative analysis below was carried out on the grid-data pair, because grid data is the preferred inverse-analysis input and is easier to interpret as a measurement field.


## Main Finding

The DIC software's "remove rigid-body displacement" operation is not a simple subtraction of the average displacement.

It behaves like a rigid-body transformation of the current configuration:

- a nearly uniform translation
- plus a very small in-plane rotation

The software updates both:

- the current coordinates `X`, `Y`
- the displacement components `DX`, `DY`

while keeping the reconstructed reference coordinates effectively unchanged.


## Evidence

### 1. `X, Y` change together with `DX, DY`

For the compared grid exports:

- mean `X` change: about `+0.28492 mm`
- mean `DX` change: about `+0.28492 mm`
- mean `Y` change: about `+0.05430 mm`
- mean `DY` change: about `+0.05430 mm`

So the operation is not just:

```text
DX_new = DX_old - constant
DY_new = DY_old - constant
```

Instead, the software is updating the current coordinates consistently with the displacement change.


### 2. Reference coordinates remain unchanged

The following relations remain true up to numerical noise:

```text
X_remove - DX_remove ~= X_raw - DX_raw
Y_remove - DY_remove ~= Y_raw - DY_raw
```

The observed differences are on the order of `1e-8`.

This means:

- the reference configuration is effectively preserved
- the software is changing the current configuration representation
- the removed field is still internally self-consistent


### 3. Best-fit rigid transform between raw and removed coordinates

Fitting a rigid transform from the raw current coordinates to the removed current coordinates gives approximately:

- translation in `x`: `+0.28450 mm`
- translation in `y`: `+0.05429 mm`
- rotation angle: `-0.00733 deg`

The rigid-fit residual is about `1e-6 mm`, which is extremely small.

So the software operation is very well described as:

- rigid translation
- plus a tiny rigid rotation


## Why The Displacement Magnitudes Become Much Smaller

Before rigid-body removal, the grid data had approximately:

- `DX` range: `[-0.29857, -0.25474] mm`
- `DY` range: `[-0.07253, -0.04284] mm`

After rigid-body removal, the grid data became approximately:

- `DX` range: `[-0.01247, 0.02938] mm`
- `DY` range: `[-0.01430, 0.01411] mm`

This happens because the original field contains a dominant whole-body motion component.

After the software subtracts the best rigid-body motion, the remaining field only contains:

- residual deformation relative to that rigid-body motion
- local non-rigid kinematic variation

So the numbers become much smaller because the software is no longer showing the absolute motion of the specimen relative to the lab or image frame. It is showing motion relative to a co-moving rigid frame.


## Interpretation

The rigid-body-removed field should not be interpreted as:

- absolute displacement in the test-machine frame
- displacement relative to the fixed clamp

Instead, it should be interpreted as:

- displacement after removing a best-fit rigid translation/rotation
- residual deformation field in a re-referenced current frame

Therefore:

- the fixed end is not guaranteed to be zero
- the loading end is not guaranteed to equal the total actuator displacement
- the sign pattern only has meaning relative to the rigid-body-removed frame


## Implications For This Project

This result changes the interpretation of the input data in an important but useful way.

### Good news

Because:

```text
X_ref = X - DX
Y_ref = Y - DY
```

remains effectively unchanged after software rigid-body removal, the exported "remove" field is still self-consistent with the project's reference-coordinate convention.

So the rigid-body-removed data does not destroy the reference-configuration reconstruction.


### Caution

However, it still changes the physical meaning of the displacement field:

- raw data represents motion in the original software/lab frame
- removed data represents motion after subtracting a best-fit rigid-body component

So the two fields cannot be treated as equivalent for boundary-condition interpretation.


## Recommendation For Formal Inverse Input

For this project, the recommended default remains:

- use raw grid data as the formal inverse input

Reasons:

1. It preserves the original measured displacement relative to the DIC export frame.
2. It makes the FE boundary-condition meaning easier to track.
3. It avoids mixing software-side rigid-body assumptions with code-side alignment assumptions.
4. It keeps the inverse preprocessing under the project's control.

The rigid-body-removed grid data is still very valuable, but mainly as:

- a diagnostic field
- a way to inspect deformation after removing bulk motion
- a reference for future improvement of the code-side rigid-body alignment


## Practical Policy

Recommended practical policy for `stage_00463` and similar cases:

- formal inverse input: raw grid CSV
- diagnostic comparison: rigid-body-removed grid CSV
- visualization and cloud-plot diagnosis: pixel CSVs

This policy should be changed only if a later workflow revision explicitly chooses to accept software rigid-body-removed data as the formal observation field.
