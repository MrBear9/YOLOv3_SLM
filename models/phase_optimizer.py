"""Adam phase updates with an additional task-driven modulation direction.

For a phase p, dL/d(log g) = <dL/dp, p - mean(p)>. One scalar Adam
update coordinates the contrast of a whole phase layer; ordinary Adam still
learns its spatial structure. The gain is immediately folded into p, so model
state dictionaries, simulation and hardware export keep their existing format.
This uses backpropagated task gradients, not a forward-only phase estimator.
"""

import math

import torch


class _PhaseGainMixin:
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        directions = []
        for group in self.param_groups:
            gain_lr = float(group.get("phase_gain_lr", 0.0))
            reference_lr = float(group.get("phase_gain_reference_lr", group["lr"]))
            if gain_lr <= 0 or reference_lr <= 0 or group["lr"] <= 0:
                continue
            for name, phase in zip(group.get("phase_gain_names", ()), group["params"]):
                if name is None or phase.grad is None:
                    continue
                # DDP has already averaged these gradients. Every rank computes
                # the same scalar direction; no extra collective or forward pass.
                centered = phase.detach().float() - phase.detach().float().mean()
                derivative = (phase.grad.detach().float() * centered).sum()
                directions.append((group, phase, derivative, gain_lr * group["lr"] / reference_lr))

        result = super().step()
        for group, phase, derivative, gain_lr in directions:
            state = self.state[phase]  # ordinary Adam state is initialized now
            if "phase_gain_log" not in state:
                state["phase_gain_log"] = derivative.new_zeros(())
                state["phase_gain_m"] = derivative.new_zeros(())
                state["phase_gain_v"] = derivative.new_zeros(())
                state["phase_gain_step"] = 0
            state["phase_gain_step"] += 1
            step = state["phase_gain_step"]
            beta1, beta2 = group["betas"]
            if group.get("maximize", False):
                derivative = -derivative
            state["phase_gain_m"].lerp_(derivative, 1 - beta1)
            state["phase_gain_v"].mul_(beta2).addcmul_(derivative, derivative, value=1 - beta2)
            direction = (state["phase_gain_m"] / (1 - beta1 ** step)) / (
                (state["phase_gain_v"] / (1 - beta2 ** step)).sqrt() + group["eps"]
            )
            previous = state["phase_gain_log"]
            # Bounds apply to accumulated gain in this optimizer stage, not to
            # pixel phase values. Gains may decrease if the task loss prefers it.
            updated = (previous - gain_lr * direction).clamp(math.log(0.5), math.log(2.0))
            ratio = (updated - previous).exp()
            mean = phase.mean()
            phase.copy_(mean + ratio.to(phase.dtype) * (phase - mean))
            state["phase_gain_log"] = updated
        return loss if closure is not None else result

    def phase_gain_statistics(self):
        """Cumulative amplitude multiplier; pixel updates are measured separately."""
        stats = {}
        for group in self.param_groups:
            for name, phase in zip(group.get("phase_gain_names", ()), group["params"]):
                if name is None:
                    continue
                value = self.state.get(phase, {}).get("phase_gain_log")
                stats[f"{name}_modulation_gain"] = 1.0 if value is None else float(value.exp().item())
        return stats


class PhaseGainAdam(_PhaseGainMixin, torch.optim.Adam):
    """Spatial Adam plus a coordinated phase-amplitude update."""


class PhaseGainAdamW(_PhaseGainMixin, torch.optim.AdamW):
    """AdamW variant; phase groups should retain zero weight decay."""
