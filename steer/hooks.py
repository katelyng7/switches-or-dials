"""Position-aware activation steering.

Positions are split into two disjoint sets, with P the (padded) prompt length:

  prompt positions      0 .. P-2
  generation positions  P-1, P, P+1, ...   (position P-2+g predicts reply token g)

The last prompt token counts as generation position 1, because its residual is
what the first reply token is sampled from. Modes:

  all       every position
  prompt    prompt positions only
  response  every generation position
  first_k   generation positions 1..k        (first_k with k=inf == response)
  after_k   generation positions k+1, ...    (after_k with k=0  == response)
  none      no steering

The hook works with and without a KV cache: it counts how many positions it has
seen, so it knows the absolute position of each column it is given.
"""
import torch

MODES = ("all", "prompt", "response", "first_k", "after_k", "none")


class PositionSteerer:
    def __init__(self, block, direction, scale, coeffs, mode="response", k=0, prompt_len=None):
        """
        block      module whose output is the residual stream to steer
        direction  (d,) steering vector; normalised to unit length here
        scale      median residual norm at this layer
        coeffs     float or (B,) tensor of strengths a, one per batch row
        prompt_len P; if None it is taken from the first forward call
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.block, self.mode, self.k = block, mode, int(k)
        self.unit = direction / direction.norm()
        self.scale = float(scale)
        self.coeffs = coeffs
        self._fixed_prompt_len = prompt_len
        self._handle = None
        self.reset()

    def reset(self):
        self.seen = 0
        self.prompt_len = self._fixed_prompt_len

    def _mask(self, T, device):
        pos = torch.arange(self.seen, self.seen + T, device=device)
        first_gen = self.prompt_len - 1
        if self.mode == "all":
            return torch.ones(T, dtype=torch.bool, device=device)
        if self.mode == "none":
            return torch.zeros(T, dtype=torch.bool, device=device)
        if self.mode == "prompt":
            return pos < first_gen
        if self.mode == "response":
            return pos >= first_gen
        if self.mode == "first_k":
            return (pos >= first_gen) & (pos < first_gen + self.k)
        return pos >= first_gen + self.k  # after_k

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        B, T, _ = h.shape
        if self.prompt_len is None:
            self.prompt_len = T
        mask = self._mask(T, h.device)
        self.seen += T
        if not mask.any():
            return output
        coeffs = torch.as_tensor(self.coeffs, device=h.device, dtype=torch.float32)
        coeffs = coeffs.expand(B) if coeffs.ndim == 0 else coeffs
        if coeffs.shape[0] != B:
            raise ValueError(f"{coeffs.shape[0]} coeffs for a batch of {B}")
        delta = (coeffs * self.scale)[:, None] * self.unit.to(h.device, torch.float32)[None, :]
        delta = delta[:, None, :] * mask[None, :, None]
        h = h + delta.to(h.dtype)
        if isinstance(output, (tuple, list)):
            return (h, *output[1:])
        return h

    def __enter__(self):
        self.reset()
        self._handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
