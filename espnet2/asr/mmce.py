import torch

def calibration_unbiased_loss(predicted_probs, correct_labels):
    """Function to compute MMCE_m loss in PyTorch, with probabilities as input."""
    
    # Get predicted class labels (argmax)
    pred_labels = torch.argmax(predicted_probs, dim=1)
    
    # Get the maximum probability for each sample (confidence score)
    predicted_probs_max = torch.max(predicted_probs, dim=1).values
    
    # Create mask for correctly predicted labels (1 if correct, 0 otherwise)
    correct_mask = (pred_labels == correct_labels).float()
    
    # Compute the difference between correct predictions and predicted probabilities
    c_minus_r = correct_mask - predicted_probs_max
    
    # Create the dot product term (outer product of c_minus_r)
    dot_product = torch.matmul(c_minus_r.unsqueeze(1), c_minus_r.unsqueeze(0))
    
    # Tile the predicted probabilities
    prob_tiled = predicted_probs_max.unsqueeze(1).repeat(1, predicted_probs_max.size(0)).unsqueeze(2)
    
    # Create probability pairs by concatenating tiled probabilities
    prob_pairs = torch.cat([prob_tiled, prob_tiled.transpose(0, 1)], dim=2)
    
    # Define kernel function
    def kernel(matrix):
        return torch.exp(-torch.abs(matrix[:, :, 0] - matrix[:, :, 1]) / (2 * 0.2))
    
    # Apply the kernel function to probability pairs
    kernel_prob_pairs = kernel(prob_pairs)
    
    # Compute numerator as dot product multiplied by kernel_prob_pairs
    numerator = dot_product * kernel_prob_pairs
    
    # Return the final MMCE_m loss value (normalizing by the number of samples squared)
    return torch.sum(numerator) / (correct_mask.size(0) ** 2)

def calibration_mmce_w_loss(predicted_probs, correct_labels):
    """Function to compute MMCE_w loss in PyTorch, with probabilities as input."""
    
    # Get predicted class labels (argmax)
    predicted_labels = torch.argmax(predicted_probs, dim=1)
    
    # Get maximum probability for each sample
    predicted_probs_max = torch.max(predicted_probs, dim=1).values
    
    # Create mask for correct predictions (1 if correct, 0 otherwise)
    correct_mask = (predicted_labels == correct_labels).float()
    
    sigma = 0.2
    
    def torch_kernel(matrix):
        """Kernel was taken to be a Laplacian kernel with sigma = 0.2."""
        return torch.exp(-torch.abs(matrix[:, :, 0] - matrix[:, :, 1]) / (2 * sigma))
    
    # Compute the number of correct and incorrect predictions
    k = torch.sum(correct_mask).int()
    k_p = torch.sum(1.0 - correct_mask).int()
    
    # Handle cases where there are no correct or incorrect predictions
    cond_k = 1 if k > 0 else 0
    cond_k_p = 1 if k_p > 0 else 0
    k = max(k, 1) * cond_k * cond_k_p + (1 - cond_k * cond_k_p) * 2
    k_p = max(k_p, 1) * cond_k_p * cond_k + ((1 - cond_k_p * cond_k) * (correct_mask.shape[0] - 2))
    
    # Select top k correct and top k_p incorrect probabilities
    correct_prob = torch.topk(predicted_probs_max * correct_mask, k)[0]
    incorrect_prob = torch.topk(predicted_probs_max * (1 - correct_mask), k_p)[0]
    
    # Define function to get pairs
    def get_pairs(tensor1, tensor2):
        correct_prob_tiled = tensor1.unsqueeze(1).repeat(1, tensor1.size(0)).unsqueeze(2)
        incorrect_prob_tiled = tensor2.unsqueeze(1).repeat(1, tensor2.size(0)).unsqueeze(2)
        
        correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.transpose(0, 1)], dim=2)
        incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.transpose(0, 1)], dim=2)
        
        correct_prob_tiled_1 = tensor1.unsqueeze(1).repeat(1, tensor2.size(0)).unsqueeze(2)
        incorrect_prob_tiled_1 = tensor2.unsqueeze(1).repeat(1, tensor1.size(0)).unsqueeze(2)
        
        correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.transpose(0, 1)], dim=2)
        
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs
    
    # Get correct, incorrect, and correct-incorrect probability pairs
    correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs = get_pairs(correct_prob, incorrect_prob)
    
    # Apply Laplacian kernel to the pairs
    correct_kernel = torch_kernel(correct_prob_pairs)
    incorrect_kernel = torch_kernel(incorrect_prob_pairs)
    correct_incorrect_kernel = torch_kernel(correct_incorrect_pairs)
    
    # Compute sampling weights and values for correct and incorrect predictions
    sampling_weights_correct = torch.matmul((1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0))
    sampling_weights_incorrect = torch.matmul(incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0))
    sampling_correct_incorrect = torch.matmul((1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0))
    
    def get_out_tensor(kernel, weights):
        return kernel * weights

    correct_correct_vals = get_out_tensor(correct_kernel, sampling_weights_correct)
    incorrect_incorrect_vals = get_out_tensor(incorrect_kernel, sampling_weights_incorrect)
    correct_incorrect_vals = get_out_tensor(correct_incorrect_kernel, sampling_correct_incorrect)
    
    # Denominator for normalizing the loss
    correct_denom = torch.sum(1.0 - correct_prob)
    incorrect_denom = torch.sum(incorrect_prob)
    
    m = torch.sum(correct_mask)
    n = torch.sum(1.0 - correct_mask)
    
    # Compute MMD error with the normalization
    mmd_error = (1.0 / (m * m + 1e-5)) * torch.sum(correct_correct_vals)
    mmd_error += (1.0 / (n * n + 1e-5)) * torch.sum(incorrect_incorrect_vals)
    mmd_error -= (2.0 / (m * n + 1e-5)) * torch.sum(correct_incorrect_vals)
    
    # Final MMCE_w loss
    return torch.max(torch.sqrt(mmd_error + 1e-10), torch.tensor(0.0)) * cond_k * cond_k_p
