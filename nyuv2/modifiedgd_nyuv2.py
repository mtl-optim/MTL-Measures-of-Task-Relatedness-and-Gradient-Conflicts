import torch
import numpy as np
import itertools

# Global container for condition numbers per gradient pair
# Should be initialized to have one list per pair: (0,1), (0,2), (1,2)
condition_numbers = [[], [], []]

def modifiedgd(grads, eps=1e-5):
    """
    ModifiedGD for 3-task gradient conflict resolution.

    Args:
        grads (torch.Tensor): A tensor of shape (d, 3), where d is the number of parameters
                              and each column is a gradient vector from one task.
        eps (float): Small threshold to avoid division by near-zero singular values.

    Returns:
        torch.Tensor: A combined gradient vector of shape (d,) that resolves gradient conflicts.
    """
    g1 = grads[:, 0]
    g2 = grads[:, 1]
    g3 = grads[:, 2]

    grad_vec = grads.t()  # Shape: (3, d)

    # Compute L2 norms
    norm_g1 = torch.norm(g1, 2)
    norm_g2 = torch.norm(g2, 2)
    norm_g3 = torch.norm(g3, 2)
    tensor_norms = [norm_g1, norm_g2, norm_g3]
    min_norm = min(tensor_norms)

    # Check condition numbers between all pairs
    num_conflicts = 0
    idx_of_pair = 0

    for i, j in itertools.combinations(range(grads.shape[1]), 2):
        pair = torch.stack([grads[:, i], grads[:, j]], dim=0)  # Shape: (2, d)
        U, singular_values, Vh = torch.linalg.svd(pair, full_matrices=False)

        if singular_values.min() <= eps:
            cond_num = -1
        else:
            cond_num = singular_values.max() / singular_values.min()
            condition_numbers[idx_of_pair].append(cond_num.cpu())

        # Count pair as conflicting if condition number is high
        if (len(condition_numbers[idx_of_pair]) != 0 and cond_num != -1 and 
            cond_num > np.percentile(condition_numbers[idx_of_pair], 70)):
            num_conflicts += 1

        idx_of_pair += 1

    # If fewer than 3 conflicts, average gradients directly
    if num_conflicts < 3:
        g = grad_vec.mean(dim=0)
        return g
    else:
        # Scale all gradients to same norm, then average
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
    # Three example gradients of a model with 5 parameters
    g1 = torch.tensor([0.2, -0.1, 0.4, -0.3, 0.5])
    g2 = torch.tensor([0.5, -0.2, 0.1, -0.4, 0.2])
    g3 = torch.tensor([-0.3, 0.4, -0.2, 0.3, -0.1])
    
    grads = torch.stack([g1, g2, g3], dim=1)  # Shape: (5, 3)

    combined_grad = modifiedgd(grads)
    print("Combined Gradient:", combined_grad)
