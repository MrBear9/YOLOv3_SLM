"""Numerical FFM with coordinate-mapped forward error propagation.

This is a simulation implementation, NOT a validated physical model-free system.

Optical propagation is differentiated explicitly. Autograd is used only for the
digital output objective and the local hardware-grid phase mapping. A separate
full-autograd reference is used by the mandatory startup diagnostic.
"""
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from models.SLM.forward_learning import CosinePhaseSpace
from models.SLM.phase_search import confirmation


def adjoint_propagate(prop, error):
    """Exact adjoint of C F^-1 H F P, including asymmetric odd-size padding."""
    h,w=prop.input_resolution
    if prop._linear_conv:
        ph,pw=prop.H.shape[-2:]
        top=(ph-h+1-h%2)//2;left=(pw-w+1-w%2)//2
        error=F.pad(error,(left,pw-w-left,top,ph-h-top))
    spectrum=torch.fft.fft2(torch.fft.ifftshift(error,dim=(-2,-1)),norm='ortho')
    out=torch.fft.fftshift(torch.fft.ifft2(spectrum*prop.H.conj(),norm='ortho'),dim=(-2,-1))
    if prop._linear_conv:
        ph,pw=prop.H.shape[-2:]
        top=(ph-h+h%2)//2;left=(pw-w+w%2)//2
        out=out[...,top:top+h,left:left+w]
    return out


def mapped_forward_error(prop,error):
    """For this centered convolution: W^H e = conj(J W J conj(e)).

    J reverses both spatial axes. The startup check verifies this identity
    against autograd for the configured discretization and local phase map.
    """
    return prop(error.conj().flip((-2,-1))).flip((-2,-1)).conj()


def phase_gradient(student, gray, objective, same_forward=False, mapped_forward=True):
    layers=[l for _,l in student.all_slm_layers()]
    if any(l.mode!='phase' or l._phase_param_mode!='direct_sgd' for l in layers):
        raise ValueError('Numerical adjoint requires direct_sgd phase-only layers')
    with torch.no_grad():
        amplitude=(gray.clamp(min=0)+student.config.OPTICAL_FIELD_EPS).sqrt()
        field=torch.complex(amplitude,torch.zeros_like(amplitude))
        fields=[];phases=[]
        for i,layer in enumerate(layers,1):
            phase=layer.simulation_phase()
            modulation=torch.complex(phase.cos(),phase.sin())
            field=field*modulation
            fields.append(field);phases.append(modulation)
            field=getattr(student,f'prop{i}')(field)
        intensity=(field.real.square()+field.imag.square()).detach()
    with torch.enable_grad():
        intensity.requires_grad_(True)
        loss=objective(student._postprocess_intensity(intensity))
        q,=torch.autograd.grad(loss,intensity)
    gradients=[]
    with torch.no_grad():
        error=2*q*field
    for index in reversed(range(len(layers))):
        prop=getattr(student,f'prop{index+1}')
        with torch.no_grad():
            error=(prop(error.conj()).conj() if same_forward else
                   mapped_forward_error(prop,error) if mapped_forward else adjoint_propagate(prop,error))
            local=(error.conj()*(1j*fields[index])).real.sum(0,keepdim=True)
        layer=layers[index]
        enabled=layer.phase_raw.requires_grad
        try:
            with torch.enable_grad():
                layer.phase_raw.requires_grad_(True)
                mapped=layer.simulation_phase()
                raw,=torch.autograd.grad(mapped,layer.phase_raw,local)
                gradients.append(raw.detach())
        finally:
            layer.phase_raw.requires_grad_(enabled)
        with torch.no_grad():error=error*phases[index].conj()
    return loss.detach(),list(reversed(gradients))


class DensePhaseSpace:
    """Full raw-pixel residuals scaled so coefficient L2 equals phase RMS."""
    def __init__(self,student,grid=32):
        self.layers=[l for _,l in student.all_slm_layers()]
        self.base=[l.phase_raw.detach().clone() for l in self.layers]
        self.grid=grid
        if len({tuple(x.shape) for x in self.base})!=1:raise ValueError('Unequal phase shapes')
        self.coefficients=self.base[0].new_zeros(len(self.layers),self.base[0].numel())

    @torch.no_grad()
    def apply(self,coefficients=None):
        coefficients=self.coefficients if coefficients is None else coefficients
        for i,l in enumerate(self.layers):
            l.phase_raw.copy_(self.base[i]+coefficients[i].reshape_as(self.base[i])*math.sqrt(self.base[i].numel()))

    def state_dict(self):
        return dict(kind='dense',base=self.base,coefficients=self.coefficients,grid=self.grid)

    def load_state_dict(self,state):
        if state.get('kind')!='dense' or state['coefficients'].shape!=self.coefficients.shape:
            raise ValueError('Incompatible dense phase state')
        for a,b in zip(self.base,state['base']):a.copy_(b)
        self.coefficients.copy_(state['coefficients']);self.apply()


def diagnose(student,gray,objective,grid=32,tolerance=.005,component_objectives=None):
    """Read-only diagnostic: full gradient, projection and same-forward identity."""
    layers=[l for _,l in student.all_slm_layers()]
    flags=[l.phase_raw.requires_grad for l in layers]
    try:
        with torch.enable_grad():
            for l in layers:l.phase_raw.requires_grad_(True)
            reference_loss=objective(student(gray))
            reference=torch.autograd.grad(reference_loss,[l.phase_raw for l in layers])
        _,manual=phase_gradient(student,gray,objective,mapped_forward=False)
        _,mapped=phase_gradient(student,gray,objective)
        _,ffm=phase_gradient(student,gray,objective,same_forward=True)
        basis=CosinePhaseSpace(student,grid)
        rows=[]
        for i,(g,m,f) in enumerate(zip(reference,manual,ffm)):
            norm=g.norm().clamp_min(1e-20)
            by,bx=basis.basis[i];plane=g.squeeze()
            coefficients=by@plane@bx.T/plane.numel()
            coefficients[0,0]=0
            projection=by.T@coefficients@bx
            rows.append(dict(layer=i+1,gradient_norm=g.norm().item(),
                adjoint_relative_error=((g-m).norm()/norm).item(),
                same_forward_relative_error=((g-f).norm()/norm).item(),
                mapped_forward_relative_error=((g-mapped[i]).norm()/norm).item(),
                projection_energy_fraction=(projection.square().sum()/g.square().sum().clamp_min(1e-30)).item(),
                cosine=F.cosine_similarity(g.flatten(),m.flatten(),dim=0).item()))
        component_report={}
        if component_objectives:
            component_gradients={}
            for name,function in component_objectives.items():
                _,component_gradients[name]=phase_gradient(student,gray,function)
            for i in range(len(layers)):
                entries={name:float(gs[i].norm()) for name,gs in component_gradients.items()}
                if 'detection' in component_gradients and 'feature' in component_gradients:
                    entries['detection_feature_cosine']=float(F.cosine_similarity(
                        component_gradients['detection'][i].flatten(),component_gradients['feature'][i].flatten(),dim=0))
                component_report[f'layer_{i+1}']=entries
        # Directional finite difference over the entire optical/digital path.
        originals=[l.phase_raw.detach().clone() for l in layers]
        scale=max(float(g.square().mean().sqrt()) for g in reference)+1e-20
        directions=[g.detach()/scale for g in reference]
        predicted=sum(float((g*d).sum()) for g,d in zip(reference,directions))
        differences=[]
        try:
            for epsilon in (.001,.0003,.0001):
                losses=[]
                for sign in (-1,1):
                    with torch.no_grad():
                        for l,b,d in zip(layers,originals,directions):l.phase_raw.copy_(b+sign*epsilon*d)
                        losses.append(float(objective(student(gray))))
                measured=(losses[1]-losses[0])/(2*epsilon)
                differences.append(dict(epsilon=epsilon,measured=measured,predicted=predicted,
                    relative_error=abs(measured-predicted)/max(abs(predicted),1e-12)))
        finally:
            with torch.no_grad():
                for l,b in zip(layers,originals):l.phase_raw.copy_(b)
        return dict(backend='numerical_FFM_with_spatial_reversal_not_physical_validation',layers=rows,finite_difference=differences,component_gradients=component_report,
            passed=all(math.isfinite(r['adjoint_relative_error']) and r['adjoint_relative_error']<tolerance for r in rows)
                and math.isfinite(predicted) and predicted>1e-12
                and all(r['mapped_forward_relative_error']<tolerance for r in rows),
            task_finite_difference_consistent=min(x['relative_error'] for x in differences)<.15,
            finite_difference_note='Task finite differences may differ from autograd with detached assignment targets or nonsmooth normalization; reported separately, not used to relax optical gradient checks.',
            same_forward_validated=all(r['same_forward_relative_error']<tolerance for r in rows))
    finally:
        for l,flag in zip(layers,flags):l.phase_raw.requires_grad_(flag)


def gradient_step(space,controller,student,bank,objective_factory,measure,*,device,world,
                  cycle,radius,transfer,relative_gain,z_score,steps=5,mapped_forward=True):
    """Dense task gradient, per-layer RMS step and search-only backtracking.

    One winning candidate is checked once on held-out data. No detector updates
    or fitting on confirmation samples. No per-pixel optimizer state to corrupt
    on rollback; layer radii are the persistent adaptation state.
    """
    gradients=[torch.zeros_like(b) for b in space.base]
    for batch in bank:
        _,gs=phase_gradient(student,batch[0].to(device),objective_factory(batch),mapped_forward=mapped_forward)
        for total,g in zip(gradients,gs):total.add_(g/len(bank))
    for g in gradients:
        if dist.is_initialized():dist.all_reduce(g);g.div_(world)
    original=space.coefficients.clone()
    diagnostics=[dict(layer=i+1,rms=float(g.square().mean().sqrt())) for i,g in enumerate(gradients)]
    available=[i for i,l in enumerate(controller.layers) if l['cool_until']<=cycle]
    if not available:available=list(range(len(gradients)))
    selected=(cycle//4)%len(gradients) if cycle%4==0 else max(available,key=lambda i:diagnostics[i]['rms'])
    direction=gradients[selected].flatten()
    norm=direction.norm()
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite phase gradient')
    direction=direction/norm.clamp_min(1e-20)
    scores=[];accepted=False;check=None
    try:
        with torch.no_grad():
            baseline=controller.objective(measure('search').mean(0),transfer).item()
            best=baseline;winner=original.clone()
            for j in range(steps):
                step=controller.layers[selected]['trust']*(.5**j)
                candidate=original.clone();candidate[selected]-=step*direction
                candidate[selected]*=min(1.,float(radius*(1-1e-6)/candidate[selected].norm().clamp_min(1e-20)))
                space.apply(candidate)
                score=controller.objective(measure('search').mean(0),transfer).item()
                scores.append(dict(step_rms=step,objective=score))
                if math.isfinite(score) and score<best:best=score;winner=candidate.clone()
            if best<baseline:
                space.apply(original);before=measure('confirm')
                space.apply(winner);after=measure('confirm')
                accepted,check=confirmation(before,after,controller,transfer,relative_gain,z_score)
            if accepted:space.coefficients.copy_(winner)
        return dict(accepted=accepted,layer=selected+1,backend='numerical_FFM' if mapped_forward else 'numerical_adjoint',
            start_objective=baseline,best_objective=best,confirmation=check,
            gradient_layers=diagnostics,line_search=scores,paired_adaptation=None,
            proposal_rms=(winner-original).norm(dim=1).tolist(),
            phase_rms_from_base=space.coefficients.norm(dim=1).tolist())
    finally:space.apply()
