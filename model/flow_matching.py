import torch
import torch.nn as nn
from utils.data_utils import MATRIX_PAD

class DiscreteFlowMatcher(nn.Module):
    """
    Implements Discrete Flow Matching on the Mass-Conserving Manifold.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.eps = 1e-4

    def sample_conditional_pt(self, x0, arrows, arrow_lens, t):
        """
        Generates the intermediate state x_t given reactants x_0 and the mechanistic arrows.
        Same as before.
        """
        B, N, _ = x0.shape
        xt = x0.clone()
        
        t_expanded = t.view(B, 1)

        src_u = arrows[:, :, 0].long()
        src_v = arrows[:, :, 1].long()
        sink_u = arrows[:, :, 2].long()
        sink_v = arrows[:, :, 3].long()
        n_total = arrows[:, :, 4]

        probs = t_expanded.expand_as(n_total)
        k_jumped = torch.binomial(n_total, probs)

        batch_indices = torch.arange(B, device=self.device).view(B, 1).expand(B, arrows.size(1))
        mask = torch.arange(arrows.size(1), device=self.device).expand(B, arrows.size(1)) < arrow_lens.unsqueeze(1)
        
        b_idx = batch_indices[mask]
        s_u_idx = src_u[mask]
        s_v_idx = src_v[mask]
        k_u_idx = sink_u[mask]
        k_v_idx = sink_v[mask]
        vals = k_jumped[mask]

        xt.index_put_((b_idx, s_u_idx, s_v_idx), -vals, accumulate=True)
        non_diag_src = (s_u_idx != s_v_idx)
        xt.index_put_((b_idx[non_diag_src], s_v_idx[non_diag_src], s_u_idx[non_diag_src]), -vals[non_diag_src], accumulate=True)
        
        xt.index_put_((b_idx, k_u_idx, k_v_idx), vals, accumulate=True)
        non_diag_sink = (k_u_idx != k_v_idx)
        xt.index_put_((b_idx[non_diag_sink], k_v_idx[non_diag_sink], k_u_idx[non_diag_sink]), vals[non_diag_sink], accumulate=True)

        n_remaining = n_total - k_jumped
        denom = 1.0 - t_expanded + self.eps
        target_rates_all = n_remaining / denom
        
        return xt, target_rates_all, mask

    def compute_loss(self, pred_props, target_rates, arrows, arrow_mask, matrix_masks):
        """
        Calculates NLL loss using the decomposed Source/Sink propensities.
        matrix_masks: (B, N, N) where True/1 indicates a padded (invalid) entry.
        """
        log_s_prop, log_t_prop = pred_props
        B, N, _ = log_s_prop.shape

        neg_inf = -1e9
        masked_s = log_s_prop.masked_fill(matrix_masks.bool(), neg_inf)
        masked_t = log_t_prop.masked_fill(matrix_masks.bool(), neg_inf)

        src_u = arrows[:, :, 0].long()
        src_v = arrows[:, :, 1].long()
        sink_u = arrows[:, :, 2].long()
        sink_v = arrows[:, :, 3].long()
        batch_idx = torch.arange(B, device=self.device).view(B, 1).expand_as(src_u)

        log_p_arrows = log_s_prop[batch_idx, src_u, src_v] + log_t_prop[batch_idx, sink_u, sink_v]

        log_Z_s = torch.logsumexp(masked_s.view(B, -1), dim=1, keepdim=True)
        log_Z_t = torch.logsumexp(masked_t.view(B, -1), dim=1, keepdim=True)

        nll_all = -(log_p_arrows - log_Z_s - log_Z_t)

        if arrow_mask.any():
            loss_active = (nll_all[arrow_mask] * target_rates[arrow_mask]).mean()
        else:
            loss_active = torch.tensor(0.0, device=self.device)

        valid_entries = (~matrix_masks.bool()).float()
        num_valid = valid_entries.sum() + 1e-6
        loss_reg = ((log_s_prop**2) * valid_entries).sum() / num_valid + \
                   ((log_t_prop**2) * valid_entries).sum() / num_valid
        
        return loss_active + 0.01 * loss_reg
