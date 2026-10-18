# DIC ROI And FE Alignment Strategy For Inverse Analysis

## Problem Statement

The current inverse workflow uses DIC-measured displacement fields as input, but two structural issues must be handled explicitly:

1. The measured DIC ROI does not cover the full specimen.
2. The DIC grid and the FE mesh are not naturally aligned node-to-node.

These are normal conditions in full-field inverse problems. The solution is not to force the DIC grid and FE mesh to coincide, but to reformulate the inverse problem so that:

- DIC provides displacement observations only where measurements exist.
- FE provides a continuous mechanically admissible field over the full specimen.


## Recommended Formulation

The inverse problem should be formulated as:

- compare FE displacement and DIC displacement only inside the measured ROI
- do not invent pseudo-measurements outside the ROI
- reconstruct the full-domain field through equilibrium constraints and regularization

A suitable objective is:

```math
J(m) = \sum_{x_i \in ROI} \|u_{FE}(x_i; m) - u_{DIC}(x_i)\|^2 + \gamma R(m)
```

where:

- `x_i` are measurement locations
- `u_DIC(x_i)` are measured displacements
- `u_FE(x_i; m)` are FE displacements evaluated at the same locations
- `m` is the unknown modulus or parameter field
- `R(m)` is the regularization term

This means the measured displacement is a partial observation, not a full-domain truth field.


## Core Principles

### 1. Treat ROI As Partial Observation

The DIC ROI should only define where the data misfit is evaluated.

It should not be used to:

- extrapolate displacement to the full specimen before inversion
- fill non-measured regions with interpolated pseudo-data
- enforce artificial displacement constraints outside the ROI

Outside the ROI, the solution should be determined by:

- FE equilibrium
- boundary conditions
- constitutive assumptions
- regularization


### 2. Do Not Force Grid Coincidence

The DIC grid and FE mesh do not need to match geometrically.

What is required is:

- a shared physical coordinate system
- a consistent mapping between measurement points and the FE displacement field

The comparison should happen at common physical locations, not necessarily at common nodes.


### 3. Use Reference Coordinates

For small-deformation inverse analysis, the preferred basis is the reference configuration:

```text
X_ref = X - DX
Y_ref = Y - DY
```

This is already consistent with the current project workflow and is the right basis for:

- geometry definition
- FE domain definition
- ROI positioning
- measurement-to-model comparison


## Recommended Comparison Strategies

### Strategy A: Evaluate FE Displacement At DIC Measurement Points

This is the recommended long-term strategy.

Workflow:

1. Keep the DIC grid points as measurement points.
2. Solve the FE problem on its own mesh.
3. For each DIC point, find the FE element containing that point.
4. Use the FE shape functions of that element to evaluate the FE displacement at the DIC point.
5. Build the data residual directly at DIC point locations.

Residual form:

```math
r_i = u_{FE}(x_i) - u_{DIC}(x_i)
```

Advantages:

- no need to make DIC nodes and FE nodes coincide
- no need to interpolate DIC data onto FE nodes first
- natural treatment of partial ROI coverage
- avoids an extra interpolation layer on the measured data
- closest to the standard finite-element observation operator

Disadvantages:

- requires element search and shape-function evaluation
- requires changing the current residual assembly


### Strategy B: Interpolate DIC Displacement To FE Nodes Inside ROI

This is acceptable as a transitional implementation, but it is not the preferred final formulation.

Workflow:

1. Use DIC grid data as measured points.
2. Interpolate the DIC displacement field to FE nodes.
3. Only keep FE nodes inside the measured ROI for the data misfit.
4. Do not assign synthetic values outside the ROI.

Advantages:

- easier to implement using the current code structure
- compatible with the current `griddata`-based preprocessing path

Disadvantages:

- introduces interpolation error before inversion
- may smooth the measured field twice
- may become unstable if FE nodes are denser than the DIC grid
- may unintentionally extrapolate near ROI boundaries


## Recommendation For This Repository

### Short-Term Practical Path

Use Strategy B first, but correct the formulation in the following way:

- continue using DIC exported grid data as input
- continue using reference coordinates
- continue using fixed-point correction as an initial displacement correction
- interpolate DIC displacement only for nodes inside ROI
- build a measurement mask so that only ROI-supported FE nodes contribute to the data term
- do not interpret `U_measured` as a full-domain displacement truth

This is the smallest change from the current repository design.


### Medium-Term Preferred Path

Move to Strategy A by introducing a measurement operator:

- store measurement coordinates `x_meas, y_meas`
- store measured displacements `ux_meas, uy_meas`
- evaluate FE displacement at those measurement points
- assemble the data residual using those pointwise comparisons

This removes the need to first convert DIC measurements into a full nodal FE displacement vector.


## Alignment Requirements

Before either comparison strategy is used, DIC data and FE geometry must be aligned in the same coordinate system.

The required alignment steps are:

1. Restore reference coordinates.
2. Remove rigid-body offset.
3. Align the DIC frame and FE frame.
4. Confirm the physical specimen boundary and ROI boundary in the same basis.


## Rigid-Body Correction Recommendation

The current fixed-point subtraction is a useful first approximation, but it is weak because it only removes displacement using one reference point.

A more robust alignment should estimate at least:

- translation in `x`
- translation in `y`
- in-plane rotation `theta`

Recommended correction model:

```math
x' = R(\theta) x + t
```

where:

- `R(theta)` is the planar rotation matrix
- `t` is the translation vector

This correction should be applied to the coordinate system used for model-data comparison.


## ROI Handling Rules

The ROI should be handled explicitly through a mask or observation operator.

Recommended rules:

- only points inside the valid DIC ROI participate in the data term
- FE nodes or DIC points outside the ROI are excluded from the residual
- FE solve is still carried out on the full specimen domain
- no artificial data completion should be performed before inversion

This avoids contaminating the inverse solution with non-measured regions.


## Implications For Current Code

The current preprocessing path in `lib/dic_inverse_input.py` does the following:

1. align DIC coordinates
2. interpolate DIC displacement to FE grid locations
3. convert the interpolated field into `U_measured`

This is close to Strategy B, but currently it implicitly behaves as if a full nodal displacement field were available everywhere.

That assumption should be corrected.

Recommended changes:

- add an ROI mask or measurement mask to the prepared inverse input
- only use masked nodes in the data residual
- prevent non-ROI FE nodes from contributing to the displacement mismatch term
- later replace nodal interpolation residuals with pointwise FE evaluation at DIC measurement points


## Suggested Data Structures

### Transitional Version

For a masked nodal comparison:

```text
U_measured
measurement_mask
X_grid
Y_grid
```

where `measurement_mask` marks which FE nodes have valid DIC support.


### Preferred Version

For pointwise measurement comparison:

```text
x_meas
y_meas
ux_meas
uy_meas
measurement_mask
```

Optionally:

```text
element_id_per_point
shape_function_weights_per_point
```

to accelerate repeated FE evaluation during inversion.


## Objective Development Order

Recommended implementation order:

1. Keep DIC grid data as the only inverse input source.
2. Add explicit ROI masking to the current FE-node comparison workflow.
3. Ensure the residual is only assembled inside ROI-supported locations.
4. Replace single-point displacement correction with rigid-body alignment.
5. Upgrade the residual evaluation from FE-node comparison to DIC-point comparison.


## Final Recommendation

For this project, the technically appropriate formulation is:

- use DIC exported grid data, not pixel data
- work in reference coordinates
- treat DIC as partial observation over ROI only
- solve the FE problem on the full specimen
- compare FE and DIC only at measured locations

In the short term, use masked interpolation to FE nodes.

In the long term, switch to a true measurement-operator formulation where FE displacements are evaluated directly at DIC measurement points.
