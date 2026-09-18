"""Bounded, transactional phase search; all objectives use forward measurements.

CMA-ES uses the upstream pycma ask/tell implementation. Covariance is deliberately
restarted between blocks because the detector and calibration images can change.
"""
import math
import numpy as np
import torch
import torch.distributed as dist


def broadcast(value):
    if dist.is_initialized():
        dist.broadcast(value, 0)
    return value


def leader():
    return not dist.is_initialized() or dist.get_rank() == 0


def to_ball(z, radius):
    radius *= 1-1e-6
    norm = z.norm(dim=-1, keepdim=True)
    return radius * z * (norm.tanh() / norm.clamp_min(1e-12))


def confirmation(before, after, controller, transfer, relative_gain, z_score):
    """Paired batch uncertainty heuristic, not a formal generalization guarantee."""
    result = {}
    passed = True
    for name, b, a in (
        ("detection", before[:, 0], after[:, 0]),
        ("objective", before[:, 0] + transfer*(before[:, 1:]*controller.weights).sum(1),
         after[:, 0] + transfer*(after[:, 1:]*controller.weights).sum(1)),
    ):
        delta = a-b
        se = delta.std(unbiased=True)/math.sqrt(len(delta)) if len(delta)>1 else delta.new_tensor(0.)
        upper = delta.mean() + z_score*se
        threshold = -relative_gain*b.mean().abs()
        ok = bool(torch.isfinite(delta).all() and upper < threshold)
        result[name] = {"before": b.mean().item(), "after": a.mean().item(),
                        "delta": delta.mean().item(), "standard_error": se.item(),
                        "upper": upper.item(), "threshold": threshold.item(), "passed": ok}
        passed = passed and ok
    return passed, result


def evidence_status(details, z_score=.5):
    """Three-way heuristic; an inconclusive test is not proof of degradation."""
    if details is None:
        return 'no_proposal'
    if all(row['passed'] for row in details.values()):
        return 'improving'
    if any(row['delta']-z_score*row['standard_error']>0 for row in details.values()):
        return 'degrading'
    return 'uncertain'


@torch.no_grad()
def probe_subspace(space, controller, measure, *, seed, cycle, radius, transfer,
                   directions=4, epsilon=.02, exploration_every=4, layer_attempts=None):
    """Rank layer/band coordinate responses using search-only paired probes."""
    original=space.coefficients.clone()
    rng=torch.Generator().manual_seed(seed)
    xy=torch.arange(space.grid,device=original.device)
    frequency=torch.maximum(xy[:,None],xy[None,:]).flatten()[1:]
    rows=[]
    try:
        for layer in range(len(space.layers)):
            edge=min(space.grid,max(4,controller.layers[layer]['grid']))
            bands=[(0,max(2,edge//2)),(max(2,edge//2),edge),(edge,min(space.grid,edge+4))]
            for low,high in bands:
                indices=((frequency>=low)&(frequency<high)).nonzero().flatten()
                if not indices.numel(): continue
                order=torch.randperm(indices.numel(),generator=rng)[:directions].to(indices.device)
                for index in indices[order].tolist():
                    measurements=[]
                    for sign in (-1,1):
                        candidate=original.clone();candidate[layer,index]+=sign*epsilon
                        norm=candidate[layer].norm()
                        candidate[layer]*=min(1.,float(radius*(1-1e-6)/norm.clamp_min(1e-12)))
                        space.apply(candidate)
                        values=measure('probe')
                        if not torch.isfinite(values).all(): raise FloatingPointError('Nonfinite probe')
                        measurements.append(values[:,0]+transfer*(values[:,1:]*controller.weights).sum(1))
                    delta=measurements[1]-measurements[0]
                    se=delta.std(unbiased=True)/math.sqrt(len(delta)) if len(delta)>1 else delta.new_tensor(0.)
                    score=max(0.,float(delta.mean().abs()-.5*se))
                    rows.append(dict(layer=layer,index=index,band=[low,high],score=score,
                                     slope=float(delta.mean()/(2*epsilon))))
        eligible=[l for l in range(len(space.layers)) if controller.layers[l]['cool_until']<=cycle]
        if not eligible: eligible=list(range(len(space.layers)))
        if layer_attempts is not None:
            # Balance maintenance coverage before spending more on a sensitive layer.
            least=min(layer_attempts[l] for l in eligible)
            eligible=[l for l in eligible if layer_attempts[l]==least]
            selected=max(eligible,key=lambda l:sum(r['score'] for r in rows if r['layer']==l))
        elif cycle%exploration_every==0:
            selected=(cycle//exploration_every)%len(space.layers)
        else:
            selected=max(eligible,key=lambda l:sum(r['score'] for r in rows if r['layer']==l))
        ranked=sorted((r for r in rows if r['layer']==selected),key=lambda r:r['score'],reverse=True)
        return selected,[r['index'] for r in ranked[:max(1,len(ranked)//2)]],rows
    finally:
        space.apply(original)


@torch.no_grad()
def search_block(space, controller, measure, *, layer_index=0, active_grid=4,
                 generations=6, population=16, radius=.15, local_radius=.03,
                 sigma_rms=.01, transfer=.02, seed=42, relative_gain=.0002,
                 z_score=.5, progress=None, max_dimensions=32, guided_indices=None, guided_slopes=None, pair_check=None,
                 return_proposal=False, expanded_confirmation=False):
    """Search one SLM's active frequency band; other SLMs remain bitwise fixed.

    Only the search pool ranks candidates. The single winning candidate is then
    checked once against independent batches. No loss weights are updated here.
    """
    if not 0<=layer_index<len(space.layers) or not 2<=active_grid<=space.grid:
        raise ValueError('Invalid layer or active frequency grid')
    if population<4 or generations<1 or max_dimensions<1 or min(radius,local_radius,sigma_rms)<=0:
        raise ValueError('Invalid search budget or radii')
    original=space.coefficients.clone()
    if original[layer_index].norm()>=radius:
        raise ValueError('Starting phase is outside the search radius')
    xy=torch.arange(space.grid,device=original.device)
    mask=((xy[:,None]<active_grid)&(xy[None,:]<active_grid)).flatten()[1:]
    # Bounded coordinate blocks keep CMA covariance tractable at larger grids.
    indices=mask.nonzero().flatten()
    generator=torch.Generator(device='cpu').manual_seed(seed)
    if indices.numel()>max_dimensions:
        order=torch.randperm(indices.numel(),generator=generator)[:max_dimensions].to(indices.device)
        indices=indices[order]
    if guided_indices:
        guided=torch.tensor(guided_indices,device=indices.device,dtype=torch.long)
        indices=torch.cat((guided,indices[~torch.isin(indices,guided)]))[:max_dimensions]
    z0=original.new_zeros(indices.numel())
    sigma=sigma_rms/(local_radius*math.sqrt(indices.numel()))
    if guided_slopes:
        scale=max(abs(v) for v in guided_slopes.values())+1e-12
        for j,index in enumerate(indices.tolist()):
            z0[j]=-.5*sigma*guided_slopes.get(index,0.)/scale
    evaluations={'search':0,'confirm':0,'confirm_extra':0}
    rows=[]

    def decode(z):
        candidate=original.clone()
        delta=to_ball(z[None],local_radius)[0]
        candidate[layer_index,indices]+=delta
        # Convex projection preserves the local bound around the feasible start.
        norm=candidate[layer_index].norm()
        candidate[layer_index]*=min(1.,float(radius*(1-1e-6)/norm.clamp_min(1e-12)))
        return candidate

    def evaluate(a,split):
        space.apply(a)
        values=measure(split)
        evaluations[split]+=1
        if values.ndim!=2 or values.shape[1]!=5 or not torch.isfinite(values).all():
            raise FloatingPointError('Expected finite batchwise detection and structure losses')
        return values

    def feasible(a):
        return (a[layer_index]-original[layer_index]).norm().item()<=local_radius*(1+1e-5)

    try:
        start_score=controller.objective(evaluate(original,'search').mean(0),transfer).item()
        best_score,best_a=start_score,original.clone()
        if leader():
            import cma
            rng=np.random.RandomState(seed)
            es=cma.CMAEvolutionStrategy(z0.cpu().double().numpy(),sigma,
                {'popsize':population,'verbose':-9,'verb_log':0,'randn':rng.randn,'seed':seed})
        for generation in range(generations):
            candidates=z0.new_empty((population,z0.numel()))
            if leader():
                raw=es.ask()
                candidates.copy_(torch.as_tensor(np.asarray(raw),device=z0.device,dtype=z0.dtype))
            broadcast(candidates)
            scores=[]
            valid=0
            for z in candidates:
                a=decode(z)
                if not feasible(a):
                    score=1e12+1e9*(a[layer_index]-original[layer_index]).norm().item()
                else:
                    score=controller.objective(evaluate(a,'search').mean(0),transfer).item()
                    valid+=1
                    if score<best_score: best_score,best_a=score,a.clone()
                scores.append(score)
            if leader(): es.tell(raw,scores)
            row=dict(generation=generation+1,valid_candidates=valid,best_objective=best_score)
            rows.append(row)
            if progress: progress(row)
        center=torch.empty_like(z0)
        if leader(): center.copy_(torch.as_tensor(es.mean,device=z0.device,dtype=z0.dtype))
        a=decode(broadcast(center))
        if feasible(a):
            score=controller.objective(evaluate(a,'search').mean(0),transfer).item()
            if score<best_score: best_score,best_a=score,a.clone()
        accepted,check=False,None
        if best_score<start_score:
            before=evaluate(original,'confirm')
            after=evaluate(best_a,'confirm')
            accepted,check=confirmation(before,after,controller,transfer,relative_gain,z_score)
        initial_check=check
        if expanded_confirmation and evidence_status(check,z_score)=='uncertain':
            # Extra images are disjoint from the first confirmation stage.
            extra_before=evaluate(original,'confirm_extra')
            extra_after=evaluate(best_a,'confirm_extra')
            accepted,check=confirmation(torch.cat((before,extra_before)),torch.cat((after,extra_after)),
                                        controller,transfer,relative_gain,z_score)
        paired=None
        if best_score<start_score and pair_check is not None:
            accepted,paired=pair_check(original,best_a)
        if accepted and not return_proposal: space.coefficients.copy_(best_a)
        result=dict(accepted=accepted,layer=layer_index+1,active_grid=active_grid,
                    active_dimensions=z0.numel(),coordinate_indices=indices.tolist(),sigma_rms=sigma_rms,block_radius=local_radius,
                    start_objective=start_score,best_objective=best_score,confirmation=check,
                    initial_confirmation=initial_check,evidence=evidence_status(check,z_score),
                    evaluations=evaluations,generations=rows,paired_adaptation=paired,
                    phase_rms_from_base=space.coefficients.norm(dim=1).tolist(),
                    proposal_rms=(best_a-original).norm(dim=1).tolist())
        if return_proposal:
            result['_proposal']=best_a.clone() if best_score<start_score else None
        return result
    except Exception:
        space.coefficients.copy_(original)
        raise
    finally:
        space.apply()


