"""Cosine phase variables and layer-specific adaptation; no learned loss weights."""
import math
import copy

import torch


class PhaseAdaptation:
    """Per-SLM exploration, trust radius, bandwidth and cooldown from phase-only mAP feedback.

    Structural weights and scales are constants, never learned. This policy is
    optimizer history: model rollback must NOT erase failed-search observations.
    """
    def __init__(self, count, device, initial_grid=4, max_grid=8, sigma=.01,
                 min_sigma=.001, max_sigma=.02, trust=.03, max_trust=.05,
                 expand_after=2, failure_patience=2, cooldown=2, restart_after=4):
        self.weights=torch.tensor([.45,.25,.15,.15],device=device)
        self.max_grid,self.min_sigma,self.max_sigma=max_grid,min_sigma,max_sigma
        self.max_trust,self.expand_after=max_trust,expand_after
        self.failure_patience,self.cooldown=failure_patience,cooldown
        self.restart_after=restart_after
        self.initial_sigma,self.initial_trust=sigma,trust
        self.cursor=0
        self.layers=[dict(grid=initial_grid,sigma=sigma,trust=trust,successes=0,
                          failures=0,cool_until=0,last_gain=None,stagnation=0,restarts=0) for _ in range(count)]

    def objective(self, values, transfer):
        return values[0]+transfer*(self.weights*values[1:]).sum()

    def select(self, cycle):
        for offset in range(len(self.layers)):
            index=(self.cursor+offset)%len(self.layers)
            if self.layers[index]['cool_until']<=cycle:
                self.cursor=(index+1)%len(self.layers)
                return index
        return None

    def feedback(self, index, cycle, success, gain=None):
        layer=self.layers[index]
        layer['last_gain']=gain
        layer['stagnation']=0 if success else layer.get('stagnation',0)+1
        if success:
            layer['failures']=0
            layer['successes']+=1
            layer['sigma']=min(self.max_sigma,layer['sigma']*1.15)
            layer['trust']=min(self.max_trust,max(layer['sigma'],layer['trust']*1.1))
            if layer['successes']>=self.expand_after:
                layer['grid']=min(self.max_grid,layer['grid']+2)
                layer['successes']=0
        elif success is False:
            layer['failures']+=1
            layer['sigma']=max(self.min_sigma,layer['sigma']*.7)
            layer['trust']=max(layer['sigma'],layer['trust']*.8)
            if layer['failures']>=self.failure_patience:
                layer['cool_until']=cycle+self.cooldown+1
                layer['failures']=0

        # Restart exploration without erasing learned phases.
        if layer['stagnation']>=self.restart_after:
            layer['grid']=min(self.max_grid,layer['grid']+2)
            layer['sigma']=min(self.max_sigma,self.initial_sigma)
            layer['trust']=min(self.max_trust,self.initial_trust)
            layer['stagnation']=0
            layer['restarts']=layer.get('restarts',0)+1

    def state_dict(self):
        return copy.deepcopy(dict(layers=self.layers,cursor=self.cursor))

    def load_state_dict(self, state):
        if len(state['layers'])!=len(self.layers):
            raise ValueError('Phase policy layer count differs')
        self.layers=copy.deepcopy(state['layers'])
        self.cursor=state['cursor']


class CosinePhaseSpace:
    """Fixed RMS-one, non-DC cosine basis; dense phases remain export compatible."""

    def __init__(self, student, grid=8):
        self.layers = [layer for _, layer in student.all_slm_layers()]
        if not self.layers or grid < 2:
            raise ValueError("Need at least one SLM and grid >= 2")
        self.base = []
        self.basis = []
        self.grid = grid
        for layer in self.layers:
            if layer._phase_param_mode != "direct_sgd" or layer.mode != "phase":
                raise ValueError("Forward search requires direct_sgd, phase-only SLMs")
            base = layer.phase_raw.detach().clone()
            h, w = base.shape[-2:]
            if grid > min(h, w):
                raise ValueError("Cosine grid must not exceed the phase resolution")
            def basis(length):
                x = (torch.arange(length, device=base.device, dtype=base.dtype) + .5) / length
                b = torch.cos(math.pi * torch.arange(grid, device=base.device)[:, None] * x)
                b[1:] *= math.sqrt(2.)
                return b
            self.base.append(base)
            self.basis.append((basis(h), basis(w)))
        self.coefficients = self.base[0].new_zeros(len(self.layers), grid * grid - 1)

    def residual(self, index, coefficients):
        # Remove the global phase gauge: a constant phase cannot change intensity.
        matrix = torch.cat((coefficients.new_zeros(1), coefficients)).reshape(self.grid, self.grid)
        by, bx = self.basis[index]
        return (by.T @ matrix @ bx)[None, None]

    @torch.no_grad()
    def apply(self, coefficients=None):
        coefficients = self.coefficients if coefficients is None else coefficients
        for i, layer in enumerate(self.layers):
            layer.phase_raw.copy_(self.base[i] + self.residual(i, coefficients[i]))

    def state_dict(self):
        return {"grid": self.grid, "base": self.base, "coefficients": self.coefficients}

    def load_state_dict(self, state):
        if state["grid"] != self.grid or state["coefficients"].shape != self.coefficients.shape:
            raise ValueError("Resume phase space does not match")
        for dest, source in zip(self.base, state["base"]):
            dest.copy_(source)
        self.coefficients.copy_(state["coefficients"])
        self.apply()


def paired_lookahead(space, original, candidate, capture, restore, adapt, measure, compare):
    """Equal-budget detector trials; restore all mutable training state on exit.

    capture/restore must include detector, optimizer, phase and per-rank RNG.
    Only phase selection survives; trial detector updates are discarded.
    """
    common=capture()
    results=[]
    try:
        for coefficients in (original,candidate):
            restore(common)
            space.apply(coefficients)
            with torch.enable_grad():
                adapt()
            with torch.no_grad():
                results.append(measure())
        return compare(*results)
    finally:
        restore(common)
