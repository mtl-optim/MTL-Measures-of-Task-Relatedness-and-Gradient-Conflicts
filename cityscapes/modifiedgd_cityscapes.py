import torch
import numpy as np

# Stores condition numbers globally (can be used across multiple function calls)
condition_numbers = []

def modifiedgd(grads, eps=1e-5):
    """
    ModifiedGD for 2-task gradient conflict resolution.
    
    Args:
        grads (torch.Tensor): A tensor of shape (d, 2), where d is the number of parameters
                              and each column is a gradient vector from one task.
        eps (float): Small constant to check for numerical stability during SVD.

    Returns:
        torch.Tensor: A combined gradient vector of shape (d,) that balances the two input gradients.
    """
    g1 = grads[:, 0]  # Gradient from task 1
    g2 = grads[:, 1]  # Gradient from task 2

    grad_vec = grads.t()  # Shape becomes (2, d) for task-wise operations

    # Compute L2 norms of the gradients
    norm_g1 = torch.norm(g1, 2)
    norm_g2 = torch.norm(g2, 2)
    tensor_norms = [norm_g1, norm_g2]
    min_norm = min(tensor_norms)

    # Stack the gradients for SVD computation
    pair = torch.stack([g1, g2])  # Shape: (2, d)
    U, singular_values, Vh = torch.linalg.svd(pair, full_matrices=False)

    # Compute condition number of the gradient matrix
    if singular_values.min() <= eps:
        cond_num = -1
    else:
        cond_num = singular_values.max() / singular_values.min()
        condition_numbers.append(cond_num.cpu())

    # If condition number is not too high (well-conditioned case), average gradients directly
    if len(condition_numbers) == 0 or cond_num == -1 or cond_num < np.percentile(condition_numbers, 70):
        g = grad_vec.mean(dim=0)
        return g
    else:
        # Scale gradients to have the same norm (based on the minimum)
        scaled_grads = []
        for grad, norm in zip(grad_vec, tensor_norms):
            if norm > 0:
                scaling_factor = min_norm / norm
                scaled_grads.append(grad * scaling_factor)
            else:
                scaled_grads.append(grad)
        avg_grad = torch.stack(scaled_grads).mean(dim=0)
        return avg_grad

# Example usage
if __name__ == "__main__":
    # Two example gradient vectors for a model with 5 parameters
    g1 = torch.tensor([0.2, -0.1, 0.4, -0.3, 0.5])
    g2 = torch.tensor([0.5, -0.2, 0.1, -0.4, 0.2])
    
    # Stack into shape (d, 2)
    grads = torch.stack([g1, g2], dim=1)

    combined_grad = modifiedgd(grads)
    print("Combined Gradient:", combined_grad)