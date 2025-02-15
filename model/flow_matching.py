import torch
import torch.nn as nn
from utils.data_utils import MATRIX_PAD

def zero_center_func(x, node_mask):
    N = node_mask.sum()
    mean = torch.sum(x) / N
    x = x - mean * node_mask
    return x

class ConditionalFlowMatcher(nn.Module):
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, args):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : float
        """
        super().__init__()
        self.args = args
        self.device = args.device
        self.sigma = args.sigma
        self.dim = args.emb_dim

    def zero_centered_noise(self, size, node_mask_batch):
        rand = torch.randn(size).to(self.device)
        x_batch = rand * node_mask_batch
        map_zero_center = torch.vmap(zero_center_func) # map on multiple batch
        return map_zero_center(x_batch, node_mask_batch).masked_fill(~(node_mask_batch.bool()), 1e-19)

    def sample_be_matrix(self, matrix):
        node_mask = (matrix[:, :, 0] != MATRIX_PAD)
        masks = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).long()

        noise = self.zero_centered_noise(masks.shape, masks) # (n, n, b, d)
        noise = 0.5 * (noise + noise.transpose(1, 2))
        matrix = matrix + noise * self.sigma
        
        return matrix

    def sample_conditional_pt(self, x0, x1, t):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        return self.sample_be_matrix(mu_t)

    def compute_conditional_vector_field(self, x0, x1):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return x1 - x0