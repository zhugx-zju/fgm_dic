"""
Python FGM (Functionally Graded Material) Solver Package

A finite element analysis package for:
1. Forward problem: Computing displacement for FGM plane stress problems
2. Inverse problem: Reconstructing modulus distribution from displacement data
   using SciPy L-BFGS-B with Tikhonov regularization
"""

from .mesh import MeshInfo, shape_fun_at_gauss, get_body_force_load, setup_boundary_conditions
from .material import MaterialInfo, generate_fgm_modulus
from .fem_forward import fem_assemble, forward_solver, compute_reaction_forces, compute_tensile_end_force
from .regularization import get_tikhonov_regularization, get_tikhonov_gradient
from .inverse_solver import (
    lbfgs_inverse_solver_scipy,
    lambda_distance,
    get_stiffness_gradient
)
from .l_curve import (
    find_optimal_gamma_lcurve,
    plot_lcurve_results
)
from .observation import (
    ObservationPackage,
    build_observation_operator,
    build_observation_package,
    compute_axis_cell_widths,
    get_fe_gauss_targets,
    interleave_components,
    load_preprocessed_dic_grids,
    load_observation_package,
    nearest_axis_indices,
    orient_structured_grid_to_fe_basis,
    prepare_observation_package_from_grids,
    sample_dic_nearest_to_targets,
    sample_dic_on_uniform_subgrid,
    save_observation_package,
)
from .workflows import (
    run_real_data_inverse_lcurve,
)
from .utils import (
    save_results
)
from .visualization import (
    plot_displacement_fields,
    plot_gradient_field,
    plot_modulus_distribution,
    plot_reconstruction_comparison,
    plot_reconstruction_results,
    plot_single_displacement_field,
    plot_iteration_history,
    visualize_forward_results
)

__version__ = '1.0.0'
__all__ = [
    # Mesh
    'MeshInfo',
    'shape_fun_at_gauss',
    'get_body_force_load',
    'setup_boundary_conditions',
    # Material
    'MaterialInfo',
    'generate_fgm_modulus',
    # Forward solver
    'fem_assemble',
    'forward_solver',
    'compute_reaction_forces',
    'compute_tensile_end_force',
    # Regularization
    'get_tikhonov_regularization',
    'get_tikhonov_gradient',
    # Inverse solver
    'lbfgs_inverse_solver_scipy',
    'lambda_distance',
    'get_stiffness_gradient',
    # L-curve
    'find_optimal_gamma_lcurve',
    'plot_lcurve_results',
    # Observation operators
    'ObservationPackage',
    'build_observation_operator',
    'build_observation_package',
    'compute_axis_cell_widths',
    'get_fe_gauss_targets',
    'interleave_components',
    'load_preprocessed_dic_grids',
    'load_observation_package',
    'nearest_axis_indices',
    'orient_structured_grid_to_fe_basis',
    'prepare_observation_package_from_grids',
    'sample_dic_nearest_to_targets',
    'sample_dic_on_uniform_subgrid',
    'save_observation_package',
    'run_real_data_inverse_lcurve',
    # Utils
    'save_results',
    # Visualization
    'plot_displacement_fields',
    'plot_gradient_field',
    'plot_modulus_distribution',
    'plot_reconstruction_comparison',
    'plot_reconstruction_results',
    'plot_single_displacement_field',
    'plot_iteration_history',
    'visualize_forward_results',
]

