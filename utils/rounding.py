import torch
from typing import Optional

def saferound_tensor(
    x: torch.Tensor,
    places: int,
    strategy: str = "difference",
    topline: Optional[float] = None
) -> torch.Tensor:
    """
    Round a tensor elementwise to `places` decimal places, adjusting
    a minimal number of entries so that the total sum is exactly preserved.

    Args:
        x (torch.Tensor): input tensor of floats.
        places (int): number of decimal places to round to.
        strategy (str): one of {"difference","largest","smallest"}:
            - "difference": pick the entries with largest fractional parts first.
            - "largest"  : pick the largest values first.
            - "smallest" : pick the smallest values first.
        topline (float, optional): if given, override the target sum
            with `topline`. Otherwise target is x.sum().

    Returns:
        torch.Tensor: same shape as `x`, rounded to `places`, but whose
        sum exactly equals the rounded(original_sum, places).
    """
    assert isinstance(places, int), "places must be integer"
    assert strategy in ("difference","largest","smallest"), f"Unknown strategy {strategy}"

    # Flatten for simplicity
    orig = x.view(-1).to(dtype=torch.float64)
    N = orig.numel()

    # Determine the exact sum we need to hit
    total = topline if topline is not None else orig.sum().item()
    scale = 10 ** places
    target_int = int(round(total * scale))

    # Scale and take floor/ceil
    scaled = orig * scale
    low = torch.floor(scaled).to(torch.int64)   # integer floors
    high = torch.ceil(scaled).to(torch.int64)   # integer ceils
    sum_low = int(low.sum().item())
    residual = target_int - sum_low             # how many +1’s we need

    if residual != 0:
        # Depending on strategy, create a sort key
        if strategy == "difference":
            # fractional part, descending
            frac = (scaled - low).cpu()
            _, indices = torch.sort(frac, descending=True)
        elif strategy == "largest":
            # values descending
            _, indices = torch.sort(orig.cpu(), descending=True)
        else:  # "smallest"
            # values ascending
            _, indices = torch.sort(orig.cpu(), descending=False)

        # Pick exactly `abs(residual)` indices
        k = min(abs(residual), N)
        chosen = indices[:k]

        # Apply the +1 or -1
        if residual > 0:
            low[chosen] += 1
        else:
            # In the very rare case sum_low > target_int, we go back down
            low[chosen] -= 1

    # Convert back to float decimals
    rounded_flat = low.to(torch.float64).mul_(1.0 / scale)
    # reshape back
    return rounded_flat.view_as(x).to(dtype=x.dtype)

