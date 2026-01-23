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
        self.eps = 1e-5

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
        target_rates_all = torch.clamp(target_rates_all, max = 100.0)
        
        return xt, target_rates_all, mask

    def compute_loss(self, pred_props, target_rates, arrows, arrow_lens, matrix_masks):
        source_props, sink_props, stop_logits = pred_props
        B, N, _, _ = source_props.shape

        ignore_mask = matrix_masks.view(B, -1).eq(0) 
        valid_entries = ~ignore_mask

        src_flat = source_props[..., 1].view(B, -1) 
        snk_flat = sink_props[..., 1].view(B, -1)

        src_masked = src_flat.masked_fill(ignore_mask, -1e12)
        snk_masked = snk_flat.masked_fill(ignore_mask, -1e12)

        src_stop = stop_logits[:, :1]
        snk_stop = stop_logits[:, 1:]

        full_src_logits = torch.cat([src_masked, src_stop], dim=1)  
        full_snk_logits = torch.cat([snk_masked, snk_stop], dim=1) 

        log_p_src = torch.log_softmax(full_src_logits, dim=-1) 
        log_p_snk = torch.log_softmax(full_snk_logits, dim=-1)

        is_termination = (arrow_lens == 0)
        
        loss_active = torch.tensor(0.0, device=self.device)
        loss_term = torch.tensor(0.0, device=self.device)

        if (~is_termination).any():
            src_u, src_v = arrows[:, :, 0].long(), arrows[:, :, 1].long()
            snk_u, snk_v = arrows[:, :, 2].long(), arrows[:, :, 3].long()

            gt_src_idx = (src_u * N + src_v).clamp(0, N * N - 1)
            gt_snk_idx = (snk_u * N + snk_v).clamp(0, N * N - 1)

            gt_log_p_src = torch.gather(log_p_src, 1, gt_src_idx)
            gt_log_p_snk = torch.gather(log_p_snk, 1, gt_snk_idx)
            
            nll_active = -(gt_log_p_src + gt_log_p_snk)

            active_arrow_mask = torch.arange(arrows.size(1), device=self.device).expand(B, -1) < arrow_lens.unsqueeze(1)
            
            if active_arrow_mask.any():
                weights = torch.log1p(torch.clamp(target_rates, max=100.0))
                loss_active = (nll_active[active_arrow_mask] * weights[active_arrow_mask]).sum() / (active_arrow_mask.sum() + 1e-9)

        if is_termination.any():
            term_log_p_src = log_p_src[is_termination, -1]
            term_log_p_snk = log_p_snk[is_termination, -1]
            
            nll_term = -(term_log_p_src + term_log_p_snk)

            loss_term = nll_term.mean()

        reg_loss = 1e-4 * (src_flat[valid_entries]**2 + snk_flat[valid_entries]**2).mean()

        total_loss = loss_active + loss_term + reg_loss
        
        return total_loss, loss_active, loss_term
