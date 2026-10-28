"""Dense local-target phase learning for the adaptive entry point ONLY.

Paper-inspired numerical adaptation, not model-free optical hardware training.
The camera/digital objective supplies an output-field derivative. Each optical
layer sees a detached local target; no training graph spans optical layers.
Feedback uses known ASM/sampling adjoints and independently stored calibrated
pupils. No cosine basis, search, finite-difference labels or regression kernels.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.no_grad()
def asm_adjoint(prop, error):
    """Adjoint of fixed ASM, including padding/cropping and supersampling."""
    k = prop.supersample_factor
    h, w = prop.fine_resolution
    if k > 1:
        error = error.repeat_interleave(k, -2).repeat_interleave(k, -1) / (k*k)
    if prop._linear_conv:
        ph, pw = prop.H.shape[-2:]
        top, left = (ph-h+1-h%2)//2, (pw-w+1-w%2)//2
        error = F.pad(error, (left, pw-w-left, top, ph-h-top))
    spectrum = torch.fft.fft2(torch.fft.ifftshift(error, dim=(-2,-1)), norm='ortho')
    output = torch.fft.fftshift(torch.fft.ifft2(spectrum * prop.H.conj(), norm='ortho'), dim=(-2,-1))
    if prop._linear_conv:
        top, left = (ph-h+h%2)//2, (pw-w+w%2)//2
        output = output[..., top:top+h, left:left+w]
    if k > 1:
        h, w = prop.input_resolution
        output = output.reshape(*output.shape[:-2], h, k, w, k).sum(dim=(-3,-1))
    return output


@torch.no_grad()
def synchronize_module(module):
    """Synchronize real/complex state without an autograd communication graph."""
    if dist.is_initialized():
        for tensor in list(module.parameters()) + list(module.buffers()):
            view = torch.view_as_real(tensor) if tensor.is_complex() else tensor
            packed = view.contiguous()
            dist.broadcast(packed, 0)
            view.copy_(packed)


def require_finite(tensors, message):
    flag = torch.stack([torch.isfinite(x).all() for x in tensors]).all().to(torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise FloatingPointError(message)


def circular(value):
    return torch.atan2(value.sin(), value.cos())


class LocalPhaseLearner:
    """Full-pixel Adam, local optical VJPs, and physical-grid error pupils."""
    def __init__(self, student, *, lr=1e-4, beta=.9, mirror_rate=.1):
        self.student = student
        self.layers = [layer for _, layer in student.all_slm_layers()]
        if any(layer._phase_param_mode != 'direct_sgd' or layer.mode != 'phase' for layer in self.layers):
            raise ValueError('Local learning requires direct_sgd phase-only layers')
        self.parameters = [layer.phase_raw for layer in self.layers]
        self.base = [p.detach().clone() for p in self.parameters]
        self.beta, self.mirror_rate = beta, mirror_rate
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        self.theta = [None] * len(self.layers)
        self.calibration_count = 0
        self.updates = 0
        self.calibrate()

    def modulation(self, index, field):
        phase = self.layers[index].simulation_phase()
        geometry = self.student.geometry
        if geometry is not None:
            return geometry.modulate(field, phase, index)
        return field * torch.complex(phase.cos(), phase.sin())

    @torch.no_grad()
    def pupil(self, index):
        shape = getattr(self.student, f'prop{index+1}').input_resolution
        one = torch.ones((1,1,*shape), device=self.parameters[index].device, dtype=torch.complex64)
        return self.modulation(index, one)

    @torch.no_grad()
    def calibrate(self):
        """Ideal simulated modulator-plane field calibration, not phase copying.

        In hardware this needs complex fields before/after each SLM, not just
        the final intensity camera. Fixed propagation/sampling remain known.
        A uniform unit probe suffices for this noiseless phase-only simulation.
        """
        for index in range(1, len(self.layers)):
            measured = self.pupil(index)
            self.theta[index] = torch.angle(measured).detach()
        self.calibration_count += 1

    @torch.no_grad()
    def forward_fields(self, intensity):
        geometry = self.student.geometry
        if geometry is not None:
            field = geometry.incident_field(intensity, self.student.config.OPTICAL_FIELD_EPS)
        else:
            amplitude = (intensity.clamp_min(0) + self.student.config.OPTICAL_FIELD_EPS).sqrt()
            field = torch.complex(amplitude, torch.zeros_like(amplitude))
        incoming, fields = [], []
        for index in range(len(self.layers)):
            if index and geometry is not None:
                field = geometry.transfers[index-1](field)
            incoming.append(field.detach())
            field = getattr(self.student, f'prop{index+1}')(self.modulation(index, field))
            fields.append(field.detach())
        return incoming, fields

    def feature(self, field):
        intensity = self.student.geometry.intensity(field) if self.student.geometry is not None else field.abs().square()
        return self.student._postprocess_intensity(intensity)

    def local_gradient(self, index, incoming, error):
        # The graph includes ONLY phase-grid interpolation and multiplication.
        # Propagation is a fixed explicitly evaluated adjoint, not autograd.
        pupil_error = asm_adjoint(getattr(self.student, f'prop{index+1}'), error)
        parameter = self.parameters[index]
        enabled = parameter.requires_grad
        try:
            with torch.enable_grad():
                parameter.requires_grad_(True)
                actual = self.modulation(index, incoming.detach())
                return torch.autograd.grad(actual, parameter, pupil_error.detach())[0].detach()
        finally:
            parameter.requires_grad_(enabled)

    def feedback(self, index, error):
        with torch.no_grad():
            theta = self.theta[index]
            pupil = torch.complex(theta.cos(), theta.sin())
            geometry = self.student.geometry
            if geometry is not None:
                pupil = pupil * getattr(geometry, f'panel_mask_{index}')
            result = asm_adjoint(getattr(self.student, f'prop{index+1}'), error) * pupil.conj()
        if geometry is not None:
            # Transpose of the FIXED forward coordinate sampler; reverse
            # interpolation has a different scale and is not a substitute.
            with torch.enable_grad():
                x = torch.zeros((error.shape[0],1,*geometry.shapes[index-1]),
                    device=error.device, dtype=error.dtype, requires_grad=True)
                result = torch.autograd.grad(geometry.transfers[index-1](x), x, result)[0].detach()
        return result

    def gradients(self, intensity, objective):
        incoming, fields = self.forward_fields(intensity)
        terminal = fields[-1].detach().requires_grad_(True)
        loss = objective(self.feature(terminal))
        require_finite([loss], 'Non-finite camera/detector objective')
        error = torch.autograd.grad(loss, terminal)[0].detach()
        gradients = [None] * len(self.layers)
        for index in reversed(range(len(self.layers))):
            # U_target = U - error; d(0.5||U-U_target||^2)/dU = error.
            # Evaluate that residual directly to avoid cancellation at small LR.
            gradients[index] = self.local_gradient(index, incoming[index], error)
            if index:
                error = self.beta * self.feedback(index, error)
        require_finite(gradients, 'Non-finite local phase gradient')
        return loss.detach(), gradients

    def step(self, intensity, objective):
        loss, gradients = self.gradients(intensity, objective)
        if dist.is_initialized():
            for gradient in gradients:
                dist.all_reduce(gradient)
                gradient.div_(dist.get_world_size())
            dist.all_reduce(loss)
            loss.div_(dist.get_world_size())
        require_finite(gradients, 'Non-finite reduced local gradient')
        old_pupils = [None] + [torch.angle(self.pupil(i)) for i in range(1,len(self.layers))]
        previous = [p.detach().clone() for p in self.parameters]
        self.optimizer.zero_grad(set_to_none=True)
        for p, gradient in zip(self.parameters, gradients):
            p.grad = gradient
        self.optimizer.step()
        require_finite(self.parameters, 'Non-finite phase after Adam update')
        with torch.no_grad():
            for index in range(1,len(self.layers)):
                delta = circular(torch.angle(self.pupil(index)) - old_pupils[index])
                self.theta[index].add_(self.mirror_rate * delta)
            self.updates += 1
            return dict(loss=float(loss), updates=self.updates,
                phase_step_rms=[float((p-b).square().mean().sqrt()) for p,b in zip(self.parameters,previous)],
                gradient_rms=[float(g.square().mean().sqrt()) for g in gradients],
                phase_rms_from_start=[float(circular(p-b).square().mean().sqrt()) for p,b in zip(self.parameters,self.base)],
                feedback_mismatch_rms=[float(circular(self.theta[i]-torch.angle(self.pupil(i))).square().mean().sqrt())
                    for i in range(1,len(self.layers))])

    def diagnose(self, intensity):
        """Mandatory startup check against a full-chain reference, never training."""
        flags = [p.requires_grad for p in self.parameters]
        for p in self.parameters: p.requires_grad_(True)
        try:
            # A spatially varying objective avoids the phase-invariant total
            # power degeneracy. Uses configured readout and normalization.
            output = self.student(intensity)
            weight = torch.linspace(.2,1,output.numel(),device=output.device).reshape_as(output)
            objective = lambda feature: (feature.square()*weight).mean()
            reference = torch.autograd.grad(objective(output), self.parameters)
            loss, local = self.gradients(intensity, objective)
            rows = []
            for index, (left,right) in enumerate(zip(local,reference)):
                # The recursive target deliberately damps earlier layers by beta.
                expected = right * self.beta**(len(self.layers)-1-index)
                relative = float((left-expected).norm()/expected.norm().clamp_min(1e-12))
                rows.append(dict(layer=index+1, relative_error=relative,
                    reference_norm=float(expected.norm())))
            if not all(torch.isfinite(torch.tensor(r['relative_error'])) and r['relative_error']<.005 for r in rows):
                raise RuntimeError(f'Local optical mapping failed startup check: {rows}')
            return dict(passed=True,layers=rows,scope='Known-model reference; not used as feedback training labels')
        finally:
            for p, flag in zip(self.parameters, flags): p.requires_grad_(flag)

    def state_dict(self):
        return dict(theta=self.theta,base=self.base,optimizer=self.optimizer.state_dict(),
            beta=self.beta,mirror_rate=self.mirror_rate,calibration_count=self.calibration_count,updates=self.updates)

    def load_state_dict(self, state):
        if len(state['theta'])!=len(self.theta) or state['beta']!=self.beta or state['mirror_rate']!=self.mirror_rate:
            raise ValueError('Incompatible local phase learning state')
        for index in range(1,len(self.theta)):
            if state['theta'][index].shape!=self.theta[index].shape:
                raise ValueError('Feedback grid changed')
        device = self.parameters[0].device
        self.theta = [None if t is None else t.to(device).clone() for t in state['theta']]
        self.base = [p.to(device).clone() for p in state['base']]
        self.optimizer.load_state_dict(state['optimizer'])
        self.calibration_count, self.updates = state['calibration_count'],state['updates']
