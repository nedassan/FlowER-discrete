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

    def compute_loss(self, pred_props, target_rates, arrows, arrow_mask):
        log_s_prop, log_t_prop = pred_props
        B = log_s_prop.size(0)
        
        src_u = arrows[:, :, 0].long()
        src_v = arrows[:, :, 1].long()
        sink_u = arrows[:, :, 2].long()
        sink_v = arrows[:, :, 3].long()

        batch_idx = torch.arange(B, device=self.device).view(B, 1).expand_as(src_u)
        
        log_pred_rates = log_s_prop[batch_idx, src_u, src_v] + log_t_prop[batch_idx, sink_u, sink_v]
        pred_rates = torch.exp(log_pred_rates)

        if arrow_mask.any():
            loss_active = ((pred_rates[arrow_mask] - target_rates[arrow_mask])**2).mean()
        else:
            loss_active = torch.tensor(0.0, device=self.device)

        loss_bg = (log_s_prop**2).mean() + (log_t_prop**2).mean()
        
        return loss_active + 0.01 * loss_bg
