"""
Tikhonov regularization module for inverse problems.
"""

import numpy as np


def get_tikhonov_regularization(mesh_info, material_info):
    """
    Calculate the Tikhonov regularization value.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object

    Returns:
        tik_reg: Tikhonov regularization value
    """
    E_vec = material_info.get_current_modulus()
    # Equivalent to E.T @ L @ E, but nonnegative even near the constant-field
    # nullspace, where cancellation would corrupt the L-curve penalty norm.
    E_ele = E_vec[mesh_info.ele_nods_id - 1]
    weights = mesh_info.gauss_abs_det_j * mesh_info.gauss_w[:, None]
    grad_x = np.einsum('gea,ea->ge', mesh_info.gauss_dNdx, E_ele)
    grad_y = np.einsum('gea,ea->ge', mesh_info.gauss_dNdy, E_ele)
    return float(np.sum(weights*(grad_x**2 + grad_y**2)))


def get_tikhonov_gradient(mesh_info, material_info):
    """
    Calculate the gradient of the Tikhonov regularization with respect to modulus.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object

    Returns:
        tik_reg_grad: Gradient vector [n_nod]
    """
    L_reg = mesh_info.get_regularization_matrix()
    E_vec = material_info.get_current_modulus()
    return 2.0 * np.asarray(L_reg @ E_vec).ravel()


def get_affine_detrended_regularization(mesh_info, material_info):
    """Integrate squared deviations of both modulus gradients from their means.

    Any affine E(x,y) has zero penalty. Both x and y gradients remain free;
    this does not constrain the inverse field to depend on x alone.
    """
    E_vec = material_info.get_current_modulus()
    E_ele = E_vec[mesh_info.ele_nods_id - 1]
    weights = mesh_info.gauss_abs_det_j * mesh_info.gauss_w[:, None]
    grad_x = np.einsum('gea,ea->ge', mesh_info.gauss_dNdx, E_ele)
    grad_y = np.einsum('gea,ea->ge', mesh_info.gauss_dNdy, E_ele)
    area = float(weights.sum())
    mean_x = float(np.sum(weights * grad_x) / area)
    mean_y = float(np.sum(weights * grad_y) / area)
    return float(np.sum(weights * ((grad_x - mean_x)**2 + (grad_y - mean_y)**2)))


def get_affine_detrended_gradient(mesh_info, material_info):
    """Derivative of the affine-detrended gradient penalty with respect to E."""
    E_vec = material_info.get_current_modulus()
    gx, gy, area = mesh_info.get_integrated_gradient_vectors()
    gradient = 2.0 * np.asarray(mesh_info.get_regularization_matrix() @ E_vec).ravel()
    gradient -= 2.0 * ((gx @ E_vec) * gx + (gy @ E_vec) * gy) / area
    return gradient
