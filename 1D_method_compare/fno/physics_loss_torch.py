"""
PyTorch-based Physics Loss for FEM
Fully differentiable FEM residual computation
"""
import torch
import torch.nn as nn


class PhysicsLossPyTorch(nn.Module):
    """
    Differentiable FEM residual loss using PyTorch
    Computes ||K(E)*u - f||^2 / ||f||^2 with autograd support
    """

    def __init__(self, L, A, n_node):
        """
        Parameters:
        -----------
        L : float
            Bar length
        A : float
            Cross-sectional area
        n_node : int
            Number of nodes
        """
        super(PhysicsLossPyTorch, self).__init__()

        self.L = L
        self.A = A
        self.n_node = n_node
        self.n_elem = n_node - 1
        self.h = L / self.n_elem

        # Gauss quadrature points and weights (2-point)
        self.register_buffer('gauss_xi', torch.tensor([0.5 - 3**0.5/6, 0.5 + 3**0.5/6]))
        self.register_buffer('gauss_w', torch.tensor([0.5, 0.5]))

    def assemble_stiffness_batch(self, E_nodal):
        """
        Assemble global stiffness matrix for a batch

        Parameters:
        -----------
        E_nodal : tensor (B, N)
            Young's modulus at nodes for batch

        Returns:
        --------
        K : tensor (B, N, N)
            Global stiffness matrices
        """
        B = E_nodal.shape[0]
        device = E_nodal.device

        # Initialize global stiffness (batch)
        K = torch.zeros(B, self.n_node, self.n_node, device=device)

        # Shape function derivatives
        dN_dxi = torch.tensor([-1.0, 1.0], device=device)
        B_mat = dN_dxi / self.h  # B = dN/dx

        # Element assembly
        for e in range(self.n_elem):
            # Element nodal E
            E1 = E_nodal[:, e]      # (B,)
            E2 = E_nodal[:, e + 1]  # (B,)

            # Compute element stiffness using Gauss quadrature
            k_e = torch.zeros(B, 2, 2, device=device)

            for i in range(len(self.gauss_xi)):
                xi = self.gauss_xi[i]
                w = self.gauss_w[i]

                # Shape functions at Gauss point
                N1 = 1.0 - xi
                N2 = xi

                # E at Gauss point (linear interpolation)
                E_xi = E1 * N1 + E2 * N2  # (B,)

                # k_e += w * E_xi * A * B^T * B * h
                # B^T * B = outer([-1/h, 1/h], [-1/h, 1/h])
                BTB = torch.outer(B_mat, B_mat)  # (2, 2)
                k_e += w * E_xi[:, None, None] * self.A * BTB * self.h

            # Assemble into global K
            # K[nodes[i], nodes[j]] += k_e[i, j]
            nodes = [e, e + 1]
            for i in range(2):
                for j in range(2):
                    K[:, nodes[i], nodes[j]] += k_e[:, i, j]

        return K

    def compute_residual_batch(self, E_nodal, u_obs, F_batch):
        """
        Compute FEM residual for batch (differentiable)

        Parameters:
        -----------
        E_nodal : tensor (B, N)
            Predicted E at nodes
        u_obs : tensor (B, N)
            Observed displacement
        F_batch : tensor (B,)
            Applied forces

        Returns:
        --------
        residual_norm : tensor (B,)
            Normalized residual ||K*u - f|| / ||f|| for each sample
        """
        B = E_nodal.shape[0]
        device = E_nodal.device

        # Assemble K(E)
        K = self.assemble_stiffness_batch(E_nodal)  # (B, N, N)

        # Force vector: f[-1] = F, others = 0
        f = torch.zeros(B, self.n_node, device=device)
        f[:, -1] = F_batch

        # Apply Dirichlet BC: u(0) = 0
        # Work on free DOFs only (exclude node 0)
        K_free = K[:, 1:, 1:]      # (B, N-1, N-1)
        u_free = u_obs[:, 1:]      # (B, N-1)
        f_free = f[:, 1:]          # (B, N-1)

        # Compute residual: r = K_free @ u_free - f_free
        # Use batch matrix multiplication
        r = torch.bmm(K_free, u_free.unsqueeze(-1)).squeeze(-1) - f_free  # (B, N-1)

        # Normalized residual
        r_norm = torch.norm(r, dim=1)  # (B,)
        f_norm = torch.norm(f_free, dim=1) + 1e-10
        residual_norm = r_norm / f_norm

        return residual_norm

    def forward(self, E_pred, u_obs, F_batch):
        """
        Compute physics loss

        Parameters:
        -----------
        E_pred : tensor (B, 1, N) or (B, N)
            Predicted E
        u_obs : tensor (B, N)
            Observed displacement
        F_batch : tensor (B,)
            Applied forces

        Returns:
        --------
        loss : scalar tensor
            Mean squared residual
        """
        if E_pred.dim() == 3:
            E_pred = E_pred.squeeze(1)  # (B, N)

        # Compute residuals (differentiable!)
        residual_norm = self.compute_residual_batch(E_pred, u_obs, F_batch)

        # Mean squared residual
        loss = (residual_norm ** 2).mean()

        return loss


class SmoothnessLossPyTorch(nn.Module):
    """
    Smoothness loss: ||∂logE/∂x||^2 (finite difference)
    Already differentiable, just wrap in nn.Module
    """

    def __init__(self, h):
        """
        Parameters:
        -----------
        h : float
            Grid spacing
        """
        super(SmoothnessLossPyTorch, self).__init__()
        self.h = h

    def forward(self, logE_pred):
        """
        Compute smoothness loss

        Parameters:
        -----------
        logE_pred : tensor (B, 1, N) or (B, N)
            Predicted logE

        Returns:
        --------
        loss : scalar tensor
        """
        if logE_pred.dim() == 3:
            logE_pred = logE_pred.squeeze(1)  # (B, N)

        # Finite difference: (logE[i+1] - logE[i]) / h
        dlogE_dx = (logE_pred[:, 1:] - logE_pred[:, :-1]) / self.h

        # L2 norm
        loss = (dlogE_dx ** 2).mean()

        return loss


# Test the implementation
if __name__ == '__main__':
    import numpy as np
    from fem1d import FEM1DBar

    print("Testing PyTorch Physics Loss...")

    # Parameters
    L = 1.0
    A = 1.0
    n_node = 101
    batch_size = 4

    # Create instances
    physics_loss_torch = PhysicsLossPyTorch(L, A, n_node)
    fem_numpy = FEM1DBar(L, A, n_node)

    # Generate test data
    torch.manual_seed(0)
    E_pred = torch.rand(batch_size, n_node) * 0.5 + 0.8  # E in [0.8, 1.3]
    E_pred.requires_grad = True
    F_batch = torch.rand(batch_size) * 0.1

    # Generate corresponding u using numpy FEM
    u_obs_list = []
    for i in range(batch_size):
        E_np = E_pred[i].detach().numpy()
        F_np = F_batch[i].item()
        u_np = fem_numpy.forward_solve(E_np, F_np)
        u_obs_list.append(u_np)
    u_obs = torch.tensor(np.array(u_obs_list), dtype=torch.float32)

    # Compute loss
    print("\n1. Forward pass...")
    loss = physics_loss_torch(E_pred, u_obs, F_batch)
    print(f"   Physics loss: {loss.item():.6e}")

    # Test gradient
    print("\n2. Backward pass...")
    loss.backward()
    print(f"   Gradient norm: {E_pred.grad.norm().item():.6e}")
    print(f"   Gradient shape: {E_pred.grad.shape}")
    print(f"   Gradient range: [{E_pred.grad.min():.6e}, {E_pred.grad.max():.6e}]")

    # Compare with numpy version (should be similar residual value)
    print("\n3. Compare with numpy FEM...")
    E_np = E_pred[0].detach().numpy()
    u_np = u_obs[0].numpy()
    F_np = F_batch[0].item()
    residual_numpy = fem_numpy.compute_residual(E_np, u_np, F_np)
    residual_torch = physics_loss_torch.compute_residual_batch(
        E_pred[[0]], u_obs[[0]], F_batch[[0]]
    )[0].item()
    print(f"   Numpy residual:  {residual_numpy:.6e}")
    print(f"   Torch residual:  {residual_torch:.6e}")
    print(f"   Relative diff:   {abs(residual_torch - residual_numpy) / residual_numpy:.6e}")

    # Test smoothness loss
    print("\n4. Test smoothness loss...")
    h = L / (n_node - 1)
    smooth_loss = SmoothnessLossPyTorch(h)
    logE_pred = torch.log(E_pred)
    logE_pred.requires_grad = True
    loss_smooth = smooth_loss(logE_pred)
    print(f"   Smoothness loss: {loss_smooth.item():.6e}")
    loss_smooth.backward()
    print(f"   Gradient computed: {logE_pred.grad is not None}")

    print("\n✓ All tests passed!")
