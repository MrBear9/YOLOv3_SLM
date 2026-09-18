"""Layer-adaptive forward phase learning with fixed structural supervision."""
import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.forward_learning import CosinePhaseSpace, PhaseAdaptation, paired_lookahead
from models.SLM.phase_search import search_block, probe_subspace, confirmation
from models.SLM.search_diagnostics import save_diagnostics
from models.SLM.forward_distributed import initialize, sync_buffers, sync_initial_state, mean_measurement, capture_rng, restore_rng
from models.SLM.slm_train_setup import setup_training
from models.SLM.slm_utils import prepare_batch
from models.SLM.dataset_slm import slm_collate_fn
from models.SLM.utils_slm import save_detector_best
from models.SLM.evaluation_slm import evaluate_slm_detector, save_slm_detection_visualization
from models.yolov8.feature_adapter import prepare_slm_detector_feature

REVISION = "phase_maintenance_v6"
DEFAULTS = dict(data="data/military/data.yaml", device="cuda:0", cycles=24,
    batch_size=4, workers=4, seed=42, grid=16, method="cmaes",
    population=12, generations=4, search_batches=64, confirm_batches=32,
    phase_radius=.15, block_radius=.03, sigma_rms=.01, initial_grid=4,
    transfer=.02, max_sigma=.02, max_block_radius=.05,
    expand_after=2, layer_cooldown=2,
    confirm_relative_gain=.0002, confirm_z=.5,
    detector_epochs=0, detector_lr=1e-6, detector_weight_decay=0.,
    bn_mode="frozen", patience=8, min_map_gain=.0002,
    rollback_drop=.002, phase_patience=2, min_sigma=.001,
    max_train_batches=0, visualize_every=1, warmup_cycles=0, restart_after=3, max_search_dimensions=32,
    rollback_patience=3, rollback_lr_factor=1.0, probe_batches=4,
    probe_directions=3, probe_epsilon=.02, adapt_batches=8,
    maintenance_batches=32, min_layer_attempts=6, confirm_multiplier=3,
    diagnostic_budget=4, exploration_cycles=3, exploration_drop=.002, coverage_per_grid=2)
SCRATCH_DEFAULTS = dict(cycles=100, grid=32, min_sigma=.01, detector_lr=1e-4, bn_mode="train", detector_epochs=1,
    population=16, generations=6, search_batches=32, confirm_batches=16, restart_after=4,
    warmup_cycles=10, patience=20, phase_radius=1., block_radius=.1,
    sigma_rms=.03, max_sigma=.06, max_block_radius=.2, rollback_drop=.01)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher")
    p.add_argument("--init-pair")
    p.add_argument("--resume")
    p.add_argument("--restart-stopped", action="store_true")
    p.add_argument("--output", required=True)
    for key, value in DEFAULTS.items():
        p.add_argument("--"+key.replace("_", "-"), type=type(value), default=None)
    args = p.parse_args()
    if args.resume and args.init_pair:
        p.error("Use --init-pair for a new run, or --resume")
    saved = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if saved and saved.get("forward_revision") != REVISION:
        p.error("Old runner checkpoints cannot resume here; use --init-pair and a new output")
    scratch = not saved and not args.init_pair
    for key, default in DEFAULTS.items():
        if getattr(args,key) is None:
            value = saved["arguments"].get(key, default) if saved else SCRATCH_DEFAULTS.get(key, default) if scratch else default
            setattr(args,key, value)
    args.training_mode = saved["arguments"].get("training_mode", "finetune") if saved else "scratch" if scratch else "finetune"
    if args.restart_stopped and not saved:
        p.error("--restart-stopped requires --resume")
    if saved:
        args.teacher = args.teacher or saved["arguments"]["teacher"]
        for key in DEFAULTS:
            if key not in {"cycles", "workers", "device", "visualize_every", "patience", "min_map_gain", "search_batches"} and getattr(args,key)!=saved["arguments"].get(key, DEFAULTS[key]):
                p.error(f"Resume changed --{key.replace('_','-')}")
    if not args.teacher:
        p.error("Supply --teacher for frozen structural supervision (also required for scratch)")
    if args.training_mode=="scratch" and (args.detector_epochs<1 or args.warmup_cycles<1 or args.cycles<=args.warmup_cycles):
        p.error("Scratch requires detector-epochs>=1 and 1<=warmup-cycles<cycles")
    for key in ("cycles","batch_size","grid","population","generations","search_batches","confirm_batches","initial_grid","expand_after","patience","phase_patience","restart_after","max_search_dimensions","rollback_patience","probe_batches","probe_directions","adapt_batches","maintenance_batches","min_layer_attempts"):
        if getattr(args,key)<1: p.error(key+" must be positive")
    if args.probe_batches>args.search_batches or args.adapt_batches>args.search_batches:
        p.error("probe-batches and adapt-batches must not exceed search-batches")
    if args.grid<2 or args.population<4 or args.population%2 or not 2<=args.initial_grid<=args.grid:
        p.error("2<=initial-grid<=grid, even population>=4")
    for key in ("phase_radius","block_radius","sigma_rms","detector_lr","min_sigma","max_sigma","max_block_radius","probe_epsilon"):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0: p.error(key+" must be positive finite")
    for key in ("warmup_cycles","workers","detector_epochs","max_train_batches","visualize_every","transfer","confirm_relative_gain","confirm_z","min_map_gain","rollback_drop","detector_weight_decay","layer_cooldown"):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<0: p.error(key+" must be nonnegative finite")
    if not 0<args.rollback_lr_factor<=1: p.error("rollback-lr-factor must be in (0,1]")
    if args.method not in {"cmaes","frozen"} or args.bn_mode not in {"frozen","train"}:
        p.error("method=cmaes|frozen, bn-mode=frozen|train")
    if args.method=="frozen" and args.detector_epochs==0: p.error("Frozen control needs detector training")
    if args.confirm_multiplier<2 or args.exploration_cycles<1 or args.coverage_per_grid<1 or args.diagnostic_budget<0:
        p.error("confirm-multiplier>=2, exploration-cycles>=1, coverage-per-grid>=1, diagnostic-budget>=0")
    if not math.isfinite(args.exploration_drop) or args.exploration_drop<0:
        p.error("exploration-drop must be finite and nonnegative")
    if args.training_mode=='finetune' and args.method=='cmaes' and args.detector_epochs:
        p.error("Phase maintenance uses a frozen detector. Run a separate --method frozen --detector-epochs 1 control for detector adaptation.")
    if not args.min_sigma<=args.sigma_rms<=args.block_radius<=args.phase_radius:
        p.error("Require min-sigma <= sigma-rms <= block-radius <= phase-radius")
    if not args.sigma_rms<=args.max_sigma<=args.max_block_radius<=args.phase_radius or args.block_radius>args.max_block_radius:
        p.error("Require sigma-rms<=max-sigma<=max-block-radius<=phase-radius and block-radius<=max-block-radius")
    if args.method=="cmaes":
        try:
            import cma
        except ImportError:
            p.error("Install the tested backend: python -m pip install cma==4.4.4")
        if cma.__version__!="4.4.4": p.error("Use tested cma==4.4.4 for reproducible ask/tell behavior")
    return args, saved


def maintenance_improved(candidate, baseline, min_gain):
    """A neutral candidate is not a successful deployment transaction."""
    return math.isfinite(candidate) and candidate > baseline + min_gain


def restore_pair(student, detector, checkpoint):
    if checkpoint.get("phase_parameterization", "direct_sgd") != "direct_sgd":
        raise ValueError("Initialize from a direct_sgd pair; pyramid conversion must be explicit")
    geometry = checkpoint.get("optical_geometry")
    if geometry is None:
        raise ValueError("Initialization needs checkpoint optical_geometry metadata")
    if tuple(geometry["dmd_resolution"]) != tuple(Config.RESOLUTION):
        raise ValueError("Checkpoint DMD resolution differs")
    for i, layer in enumerate(geometry["slm_layers"], 1):
        if (layer["effective_sampling_pitch_m"] != Config.sampling_pitch(i)
                or tuple(layer["active_shape"]) != tuple(Config.slm_active_shape(i))):
            raise ValueError(f"Checkpoint optical geometry differs at SLM {i}")
    student.load_state_dict(checkpoint["student_state_dict"], strict=True)
    detector.load_state_dict(checkpoint["detector_state_dict"], strict=True)
    student.enable_norm = bool(checkpoint.get("student_enable_norm", student.enable_norm))
    Config.STUDENT_NORM_MODE = checkpoint.get("student_norm_mode", Config.STUDENT_NORM_MODE)



def cpu_copy(obj):
    if torch.is_tensor(obj): return obj.detach().cpu().clone()
    if isinstance(obj,dict): return {k:cpu_copy(v) for k,v in obj.items()}
    if isinstance(obj,list): return [cpu_copy(v) for v in obj]
    if isinstance(obj,tuple): return tuple(cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def snapshot(detector, space, optimizer):
    return cpu_copy(dict(detector=detector.state_dict(), coefficients=space.coefficients,
                         optimizer=optimizer.state_dict()))


def restore_snapshot(state, detector, space, optimizer):
    detector.load_state_dict(state["detector"])
    space.coefficients.copy_(state["coefficients"])
    space.apply()
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))


def finish_maintenance_trial(metrics, state, best_state, detector, space, optimizer,
                             *, min_gain, initial_lr, adapted):
    """Keep only an improving pair; rejected trials restore parameters and moments."""
    if maintenance_improved(metrics["map50"],state["best_map50"],min_gain):
        return True
    restore_snapshot(best_state,detector,space,optimizer)
    state["current_map50"]=state["best_map50"]
    state["current_detection_loss"]=best_state["validation_detection_loss"]
    if 'metrics' in best_state:
        state['current_metrics']=copy.deepcopy(best_state['metrics'])
    if adapted:
        state["lr"]=max(initial_lr*.1,state["lr"]*.5)
    return False


def coverage_grids(initial, maximum):
    grids=[initial]
    while grids[-1]<maximum:
        grids.append(min(maximum,grids[-1]*2))
    return grids


def next_coverage_grid(coverage, grids, repeats):
    return next((g for g in grids if coverage.get(str(g),0)<repeats),grids[-1])


def exploration_decision(metrics, evidence, confirmation_details, best_map, min_gain, max_drop):
    """Validation judges deployment; bounded exploration may retain weak loss gains."""
    if metrics is None or not math.isfinite(metrics['map50']) or evidence=='degrading':
        return 'reject'
    if maintenance_improved(metrics['map50'],best_map,min_gain):
        return 'commit'
    improving_means=confirmation_details is not None and all(r['delta']<0 for r in confirmation_details.values())
    if improving_means and metrics['map50']>=best_map-max_drop:
        return 'explore'
    return 'reject'


def calibration_split(dataset, count, seed):
    """Deduplicate repeated train entries before disjoint image-level splitting."""
    entries = list({os.path.normcase(os.path.realpath(x["image_path"])):x for x in dataset.entries}.values())
    order = torch.randperm(len(entries), generator=torch.Generator().manual_seed(seed)).tolist()
    if count>=len(entries): raise ValueError("Confirmation pool leaves no detector training images")
    train = copy.copy(dataset)
    train.entries = [entries[i] for i in order[count:]]
    confirm = copy.copy(dataset)
    confirm.entries = [entries[i] for i in order[:count]]
    confirm.canvas_scale_range = None
    return train, confirm


def data_fingerprint(train, confirm, val):
    digest = hashlib.sha256()
    for group, dataset in (("train",train),("confirm",confirm),("val",val)):
        digest.update(group.encode())
        for item in dataset.entries:
            for field in ("image_path","label_path"):
                path = Path(item[field])
                digest.update(str(path.resolve()).encode())
                if path.exists():
                    stat = path.stat()
                    digest.update(str(stat.st_size).encode())
                    if field=="label_path": digest.update(path.read_bytes())
    return digest.hexdigest()


def make_bank(dataset, batches, batch_size, rank, world, seed, teacher, device, offset=0):
    needed = (batches+offset)*batch_size*world
    if len(dataset)<needed:
        raise ValueError(f"Need {needed} calibration images, only {len(dataset)} available; lower calibration batch counts explicitly")
    clean = copy.copy(dataset)
    clean.canvas_scale_range = None
    ids = torch.randperm(len(clean), generator=torch.Generator().manual_seed(seed))[offset*batch_size*world:needed].tolist()
    bank = []
    with torch.no_grad():
        for j in range(batches):
            start = j*batch_size*world+rank*batch_size
            batch = slm_collate_fn([clean[i] for i in ids[start:start+batch_size]])
            gray,rgb,targets = prepare_batch(batch,device)
            bank.append((gray.cpu(), teacher(rgb).detach().cpu(), [x.cpu() for x in targets]))
    return bank, [clean.entries[i]["image_path"] for i in ids]


def main():
    args, saved = arguments()
    device,rank,world = initialize(args.device)
    main_rank = rank==0
    out = Path(args.output).resolve()
    if args.resume and Path(args.resume).resolve().parent!=out:
        raise ValueError("Resume into the original output directory")
    occupied = torch.tensor(int(out.exists() and any(out.iterdir()) and not args.resume),device=device)
    if world>1: dist.all_reduce(occupied,op=dist.ReduceOp.MAX)
    if occupied.item(): raise FileExistsError("Choose a new output directory")
    Config.YAML_PATH,Config.OUTPUT_DIR = args.data,str(out)
    Config.TEACHER_DETECTOR_CHECKPOINT = args.teacher
    Config.DEVICE,Config.GPU_IDS = str(device),([device.index] if device.type=="cuda" else [])
    Config.BATCH_SIZE,Config.NUM_WORKERS,Config.TRAIN_SEED = args.batch_size,args.workers,args.seed
    Config.SLM_PHASE_PARAM_MODE,Config.SLM_INIT_MODE = "direct_sgd","random"
    Config.SLM_MULTI_HEAD_ENABLED,Config.STUDENT_NORM_SCHEDULE = False,"always"
    ctx = setup_training(main_rank,world>1,forward_only=True, detector_from_teacher=args.training_mode!="scratch")
    student,detector,teacher = ctx["student_raw"],ctx["detector_raw"],ctx["teacher"]
    student.requires_grad_(False).eval()
    teacher.requires_grad_(False).eval()
    if ctx["val_loader"] is None: raise ValueError("Validation split required")
    checkpoint = saved or (torch.load(args.init_pair,map_location="cpu",weights_only=True) if args.init_pair else None)
    if checkpoint is not None and checkpoint.get("teacher_detector_sha256",ctx["teacher_checkpoint_sha256"])!=ctx["teacher_checkpoint_sha256"]:
        raise ValueError("Checkpoint teacher identity differs from --teacher")
    if checkpoint is not None:
        restore_pair(student,detector,checkpoint)
    space = CosinePhaseSpace(student,args.grid)
    controller = PhaseAdaptation(len(space.layers), device, initial_grid=args.initial_grid, max_grid=args.grid,
        sigma=args.sigma_rms, min_sigma=args.min_sigma, max_sigma=args.max_sigma,
        trust=args.block_radius, max_trust=args.max_block_radius, expand_after=args.expand_after,
        failure_patience=args.phase_patience, cooldown=args.layer_cooldown, restart_after=args.restart_after)
    optimizer = torch.optim.AdamW(detector.parameters(),lr=args.detector_lr,weight_decay=args.detector_weight_decay)
    source_loader = ctx["train_loader"]
    confirm_multiplier=args.confirm_multiplier if args.training_mode=='finetune' and args.method=='cmaes' else 1
    train,confirm = calibration_split(source_loader.dataset,args.confirm_batches*confirm_multiplier*args.batch_size*world,args.seed+9001)
    if len(train)<max(args.search_batches*args.batch_size*world,args.batch_size*world):
        raise ValueError("Insufficient unique train images after confirmation split")
    val_paths = {os.path.normcase(os.path.realpath(x['image_path'])) for x in ctx['val_loader'].dataset.entries}
    if any(os.path.normcase(os.path.realpath(x['image_path'])) in val_paths for x in train.entries+confirm.entries):
        raise ValueError("Train and validation image paths overlap")
    fingerprint = data_fingerprint(train,confirm,ctx['val_loader'].dataset)
    sampler = DistributedSampler(train,num_replicas=world,rank=rank,seed=args.seed,drop_last=True) if world>1 else None
    loader = DataLoader(train,batch_size=args.batch_size,sampler=sampler,shuffle=sampler is None,
                        num_workers=args.workers,drop_last=True,collate_fn=slm_collate_fn,
                        generator=torch.Generator().manual_seed(args.seed+rank),
                        worker_init_fn=source_loader.worker_init_fn,pin_memory=device.type=="cuda")
    if not len(loader): raise ValueError("Empty training loader")
    ctx["train_loader"],ctx["train_sampler"] = loader,sampler
    counts = dict(optical_images=0,search_batch_evaluations=0,teacher_images=0)
    state = dict(cycle=0,best_map50=-1.,current_map50=-1.,significant_best=-1.,stale=0,
                 layer_attempts=[0]*len(space.layers),
                 coverage=[{} for _ in space.layers],diagnostics_used=0,exploration_age=0,
                 phase_enabled=args.method!="frozen",
                 lr=args.detector_lr,stopped=False, bad_detector_cycles=0)
    best_state = None
    if saved:
        if saved["world_size"]!=world or saved["teacher_detector_sha256"]!=ctx["teacher_checkpoint_sha256"]:
            raise ValueError("Resume process count or teacher differs")
        if saved["data_fingerprint"]!=fingerprint: raise ValueError("Resume dataset changed")
        space.load_state_dict(saved["phase_space"])
        controller.load_state_dict(saved["controller"])
        optimizer.load_state_dict(saved["optimizer"])
        state,counts,best_state = saved["run_state"],saved["counts"],saved["best_state"]
        restore_rng(saved["rng_by_rank"][rank],loader,device)
        if state["stopped"] and not args.restart_stopped:
            raise ValueError("Stopped run: use --restart-stopped explicitly to extend the experiment")
        if args.restart_stopped:
            state.update(stopped=False, stale=0, significant_best=state["best_map50"])
    if state["cycle"]>=args.cycles: raise ValueError("cycles must exceed resumed cycle")
    # Explicitly synchronize parameters after loading a pair, including CPU init.
    sync_initial_state(detector, space.base+[space.coefficients])
    space.apply()
    out.mkdir(parents=True,exist_ok=True)

    def write_json(name,value):
        if main_rank: (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")

    def metadata():
        return dict(forward_revision=REVISION,arguments=vars(args),world_size=world,
            teacher_detector_sha256=ctx["teacher_checkpoint_sha256"],data_fingerprint=fingerprint,
            student_norm_mode=Config.STUDENT_NORM_MODE,selection_metric="validation_map50",
            best_map50=state["best_map50"],val_map50=state["current_map50"],
            loss=state.get("current_detection_loss"))

    def save(name,extra=None):
        if main_rank:
            temp=out/(name+".tmp")
            save_detector_best(detector,str(temp),state["cycle"],0.,
                extra={**metadata(),**(extra or {})},student=student,config=Config)
            os.replace(temp,out/name)

    def validate():
        sync_buffers(detector)
        rng = capture_rng(loader,device)
        losses,metrics = evaluate_slm_detector(Config,teacher,student,detector,ctx["val_loader"],
            ctx["detection_criterion"],ctx["feature_criterion"],device,"joint_fit",
            response_detector=ctx["reference_detector"])
        restore_rng(rng[rank],loader,device)
        counts["optical_images"]+=len(ctx["val_loader"].dataset)
        if not math.isfinite(metrics["map50"]): raise FloatingPointError("Nonfinite validation mAP")
        metrics["validation_detection_loss"]=losses["detection"]
        return metrics

    def consider(metrics):
        nonlocal best_state
        state["current_map50"]=metrics["map50"]
        state["current_detection_loss"]=metrics["validation_detection_loss"]
        state["current_metrics"]=copy.deepcopy(metrics)
        if metrics["map50"]>state["best_map50"]:
            state["best_map50"]=metrics["map50"]
            best_state=snapshot(detector,space,optimizer)
            best_state["validation_detection_loss"]=state["current_detection_loss"]
            best_state["metrics"]=copy.deepcopy(metrics)
            save("detector_best.pth")

    if not saved:
        initial=validate()
        consider(initial)
        state["significant_best"]=state["best_map50"]
        write_json("initial_metrics.json",initial)
    write_json("search_config.json",{**vars(args),"world_size":world,"phase_dimensions":space.coefficients.numel(),
        "fixed_structure_weights":controller.weights.tolist(),
        "student_norm_mode":Config.STUDENT_NORM_MODE,"student_enable_norm":student.enable_norm,
        "data_fingerprint":fingerprint,"teacher_sha256":ctx["teacher_checkpoint_sha256"],
        "train_images":len(train),"confirmation_images":len(confirm),
        "torch_version":str(torch.__version__),
        "optical_config":{k:repr(getattr(Config,k)) for k in dir(Config) if k.isupper()},
        "scope":"forward-only phase exploration with separate deployed best; scratch/frozen control uses detector backward; validation mAP50 selection"})
    write_json("confirmation_images.json",[x["image_path"] for x in confirm.entries])
    if not saved:
        rng=capture_rng(loader,device)
        save("training_last.pth",dict(phase_space=space.state_dict(),controller=controller.state_dict(),
            optimizer=optimizer.state_dict(),run_state=state,counts=counts,best_state=best_state,rng_by_rank=rng))

    for cycle in range(state["cycle"]+1,args.cycles+1):
        started=time.monotonic()
        state["cycle"]=cycle
        search_result,phase_metrics,rollback=None,None,False
        warming_up = cycle<=args.warmup_cycles
        layer_index=0 if state["phase_enabled"] and not warming_up else None
        phase_start_map=state["current_map50"]
        phase_rollback=False
        maintenance=args.training_mode=="finetune"
        committed=False
        candidate_evaluated=False
        exploration_kept=False
        previous_metrics=copy.deepcopy(state.get('current_metrics'))
        if maintenance and state['exploration_age']:
            state['exploration_age']+=1
        grids=coverage_grids(args.initial_grid,args.grid)
        if main_rank: print(f"Cycle {cycle}/{args.cycles}: method={args.method}, phase_enabled={state['phase_enabled']}, layer={None if layer_index is None else layer_index+1}, best={state['best_map50']:.6f}",flush=True)
        if layer_index is not None:
            phase_start=snapshot(detector,space,optimizer)
            phase_start_loss=state.get("current_detection_loss")
            detector.eval()
            sync_buffers(detector)
            search_bank,paths=make_bank(train,args.search_batches,args.batch_size,rank,world,args.seed+cycle*101,teacher,device)
            confirm_bank,_=make_bank(confirm,args.confirm_batches,args.batch_size,rank,world,args.seed+9002,teacher,device)
            counts["teacher_images"]+=(args.search_batches+args.confirm_batches)*args.batch_size*world
            write_json(f"search_images_cycle_{cycle:03d}.json",paths)
            banks={"search":search_bank,"confirm":confirm_bank,"probe":search_bank[:args.probe_batches]}
            @torch.no_grad()
            def measure(split):
                if split=='confirm_extra' and split not in banks:
                    extra_batches=args.confirm_batches*(args.confirm_multiplier-1)
                    banks[split],_=make_bank(confirm,extra_batches,args.batch_size,rank,world,
                        args.seed+9002,teacher,device,offset=args.confirm_batches)
                    counts['teacher_images']+=extra_batches*args.batch_size*world
                values=[]
                for gray,tfeature,targets in banks[split]:
                    feature=student(gray.to(device))
                    predictions=detector(prepare_slm_detector_feature(Config,feature).to(memory_format=torch.contiguous_format))
                    targets=[x.to(device) for x in targets]
                    detection,_=ctx["detection_criterion"](predictions,targets)
                    components=ctx["feature_criterion"](feature,tfeature.to(device),student,targets=targets,return_components=True)
                    values.append(mean_measurement(torch.cat((detection.reshape(1),components))))
                    counts["optical_images"]+=gray.shape[0]*world
                    counts["search_batch_evaluations"]+=1
                return torch.stack(values)
            def progress(row):
                if main_rank: print(f"  generation {row['generation']}/{args.generations}, best objective={row['best_objective']:.6f}, feasible={row['valid_candidates']}",flush=True)
            if maintenance:
                for index,layer in enumerate(controller.layers):
                    target_grid=next_coverage_grid(state['coverage'][index],grids,args.coverage_per_grid)
                    if target_grid!=layer['grid']:
                        layer['sigma']=args.sigma_rms
                        layer['trust']=args.block_radius
                    layer['grid']=target_grid
            layer_index,guided_indices,probe_rows=probe_subspace(space,controller,measure,
                seed=args.seed+cycle*917,cycle=cycle,radius=args.phase_radius,transfer=args.transfer,
                directions=args.probe_directions,epsilon=args.probe_epsilon,
                layer_attempts=state["layer_attempts"] if maintenance else None)
            if main_rank: print(f"  forward probes selected SLM {layer_index+1}",flush=True)

            def pair_check(original,candidate):
                def capture():
                    return snapshot(detector,space,optimizer),capture_rng(loader,device)
                def restore(common):
                    model_state,rng=common
                    restore_snapshot(model_state,detector,space,optimizer)
                    restore_rng(rng[rank],loader,device)
                    optimizer.zero_grad(set_to_none=True)
                    detector.eval()
                def adapt():
                    for gray,_,targets in search_bank[:args.adapt_batches]:
                        ctx['detector'].train()
                        if args.bn_mode=='frozen':
                            for module in detector.modules():
                                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
                        for group in optimizer.param_groups:
                            group['lr']=state['lr']*(.2+.8*.5*(1+math.cos(math.pi*(cycle-1)/max(1,args.cycles))))
                        with torch.no_grad(): feature=student(gray.to(device))
                        optimizer.zero_grad(set_to_none=True)
                        prediction=ctx['detector'](prepare_slm_detector_feature(Config,feature).to(memory_format=torch.contiguous_format))
                        loss,_=ctx['detection_criterion'](prediction,[x.to(device) for x in targets])
                        finite=torch.isfinite(loss).to(torch.int32)
                        if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                        if not finite.item(): raise FloatingPointError('Nonfinite paired adaptation loss')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(detector.parameters(),10.,error_if_nonfinite=True)
                        optimizer.step()
                        counts['optical_images']+=gray.shape[0]*world
                def measured():
                    detector.eval();sync_buffers(detector)
                    return measure('confirm')
                def compare(before,after):
                    accepted,details=confirmation(before,after,controller,args.transfer,
                                                  args.confirm_relative_gain,args.confirm_z)
                    return accepted,dict(batches_per_branch=args.adapt_batches,confirmation=details)
                return paired_lookahead(space,original,candidate,capture,restore,adapt,measured,compare)

            search_result=search_block(space,controller,measure,layer_index=layer_index,
                active_grid=controller.layers[layer_index]["grid"],
                generations=args.generations,population=args.population,radius=args.phase_radius,
                local_radius=controller.layers[layer_index]["trust"],
                sigma_rms=controller.layers[layer_index]["sigma"],transfer=args.transfer,
                seed=args.seed+cycle*1009,
                relative_gain=args.confirm_relative_gain,z_score=args.confirm_z, max_dimensions=args.max_search_dimensions,progress=progress,
                guided_indices=guided_indices,
                guided_slopes={r["index"]:r["slope"] for r in probe_rows if r["layer"]==layer_index},
                pair_check=None if maintenance else pair_check,
                return_proposal=maintenance,expanded_confirmation=maintenance)
            search_result["forward_probes"]=probe_rows
            del banks,search_bank,confirm_bank
            proposal=search_result.pop('_proposal',None)
            if maintenance:
                diagnostic=search_result['evidence']=='degrading' and state['diagnostics_used']<args.diagnostic_budget
                candidate_evaluated=proposal is not None and (search_result['evidence']!='degrading' or diagnostic)
                if candidate_evaluated:
                    if diagnostic: state['diagnostics_used']+=1
                    space.coefficients.copy_(proposal)
                    space.apply()
                search_result['diagnostic_only']=bool(diagnostic and candidate_evaluated)
            if candidate_evaluated or (not maintenance and search_result["accepted"]):
                phase_metrics=validate()
                if not maintenance:
                    consider(phase_metrics)
                if not maintenance and phase_metrics["map50"] < phase_start_map-args.rollback_drop:
                    restore_snapshot(phase_start,detector,space,optimizer)
                    state["current_map50"]=phase_start_map
                    state["current_detection_loss"]=phase_start_loss
                    phase_rollback=True
            phase_gain=(phase_metrics["map50"]-phase_start_map) if phase_metrics else None
            if not maintenance:
                controller.feedback(layer_index,cycle,
                    success=bool(search_result["accepted"] and not phase_rollback and phase_gain is not None and phase_gain>args.min_map_gain), gain=phase_gain)
            search_result.update(phase_gain=phase_gain,phase_rollback=phase_rollback,
                                 layer_state=copy.deepcopy(controller.layers[layer_index]))
            if main_rank and not maintenance:
                with (out/"phase_search.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps({"cycle":cycle,**search_result})+"\n")

        # Alternate: phase is now fixed for the entire detector training segment.
        train_losses=[]
        run_detector = not maintenance or args.method=="frozen"
        detector_epochs=args.detector_epochs if run_detector else 0
        for local_epoch in range(detector_epochs):
            if sampler: sampler.set_epoch((cycle-1)*args.detector_epochs+local_epoch)
            # Decay within a finite fine-tuning budget; rollback can reduce it further.
            lr=state["lr"]*(.2+.8*.5*(1+math.cos(math.pi*(cycle-1)/max(1,args.cycles))))
            for group in optimizer.param_groups: group["lr"]=lr
            for batch_index,batch in enumerate(tqdm(loader,desc=f"Detector cycle {cycle}",disable=not main_rank)):
                if maintenance and batch_index>=args.maintenance_batches: break
                if args.max_train_batches and batch_index>=args.max_train_batches: break
                ctx["detector"].train()
                if args.bn_mode=="frozen":
                    for module in detector.modules():
                        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
                gray,_,targets=prepare_batch(batch,device)
                with torch.no_grad(): feature=student(gray)
                counts["optical_images"]+=gray.shape[0]*world
                optimizer.zero_grad(set_to_none=True)
                loss,_=ctx["detection_criterion"](ctx["detector"](prepare_slm_detector_feature(Config,feature).to(memory_format=torch.contiguous_format)),targets)
                finite=torch.isfinite(loss).to(torch.int32)
                if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item(): raise FloatingPointError("Nonfinite detector loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(detector.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                train_losses.append(loss.item())
        if maintenance and not detector_epochs:
            final_metrics=phase_metrics if candidate_evaluated else previous_metrics
        else:
            final_metrics=validate() if detector_epochs or phase_metrics is None or phase_rollback else phase_metrics
        if maintenance and args.method=='cmaes':
            decision=exploration_decision(phase_metrics,search_result['evidence'],search_result['confirmation'],
                state['best_map50'],args.min_map_gain,args.exploration_drop) if search_result else 'reject'
            committed=decision=='commit'
            if committed:
                consider(final_metrics)
                state['exploration_age']=0
            elif decision=='explore':
                state['current_map50']=final_metrics['map50']
                state['current_detection_loss']=final_metrics['validation_detection_loss']
                state['current_metrics']=copy.deepcopy(final_metrics)
                state['exploration_age']=max(1,state['exploration_age'])
                exploration_kept=True
            else:
                if search_result:
                    restore_snapshot(phase_start,detector,space,optimizer)
                rollback=candidate_evaluated
                phase_rollback=candidate_evaluated
            # A provisional path has a finite budget, including failed attempts.
            if not committed and (state['exploration_age']>=args.exploration_cycles or cycle==args.cycles):
                restore_snapshot(best_state,detector,space,optimizer)
                state['current_map50']=state['best_map50']
                state['current_detection_loss']=best_state['validation_detection_loss']
                state['current_metrics']=copy.deepcopy(best_state['metrics'])
                state['exploration_age']=0
                exploration_kept=False
                rollback=True
                phase_rollback=candidate_evaluated
            if search_result is not None:
                state['layer_attempts'][layer_index]+=1
                key=str(search_result['active_grid'])
                state['coverage'][layer_index][key]=state['coverage'][layer_index].get(key,0)+1
                neutral=(search_result['evidence']=='uncertain' or decision=='explore') and not committed
                controller.feedback(layer_index,cycle,success=True if committed else None if neutral else False,
                    gain=phase_gain)
                search_result.update(committed=committed,decision=decision,exploration_kept=exploration_kept,
                    phase_rollback=phase_rollback,candidate_map50=phase_metrics['map50'] if candidate_evaluated else None,
                    deployed_map50=state['best_map50'],working_map50=state['current_map50'],
                    exploration_age=state['exploration_age'],phase_rms_from_base=space.coefficients.norm(dim=1).tolist(),
                    layer_state=copy.deepcopy(controller.layers[layer_index]))
                if main_rank:
                    with (out/'phase_search.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps({'cycle':cycle,**search_result})+'\n')
        elif maintenance:
            committed=finish_maintenance_trial(final_metrics,state,best_state,detector,space,optimizer,
                min_gain=args.min_map_gain,initial_lr=args.detector_lr,adapted=bool(detector_epochs))
            if committed:
                consider(final_metrics)
            else:
                rollback=True
                phase_rollback=bool(search_result and search_result["accepted"])
        else:
            consider(final_metrics)
        # Best is a full paired transaction, including optimizer and BN buffers.
        degraded=not maintenance and not warming_up and final_metrics["map50"] < state["best_map50"]-args.rollback_drop
        state['bad_detector_cycles']=state.get('bad_detector_cycles',0)+1 if degraded else 0
        if state['bad_detector_cycles']>=args.rollback_patience:
            restore_snapshot(best_state,detector,space,optimizer)
            state["current_map50"]=state["best_map50"]
            state["current_detection_loss"]=best_state.get("validation_detection_loss")
            state["lr"]*=args.rollback_lr_factor
            state["bad_detector_cycles"]=0
            rollback=True
        significant=state["best_map50"]>state["significant_best"]+args.min_map_gain
        if significant or warming_up:
            state["significant_best"]=state["best_map50"]
            state["stale"]=0
        else: state["stale"]+=1
        coverage_ready=not maintenance or args.method=="frozen" or (min(state['layer_attempts'])>=args.min_layer_attempts and all(
            coverage.get(str(g),0)>=args.coverage_per_grid for coverage in state['coverage'] for g in grids))
        state["stopped"]=state["stale"]>=args.patience and coverage_ready and state['exploration_age']==0
        summary=torch.tensor([sum(train_losses),len(train_losses)],dtype=torch.float64,device=device)
        if world>1: dist.all_reduce(summary)
        row={"cycle":cycle,"stage":"detector_warmup" if warming_up else "alternating","phase_metrics":phase_metrics,"metrics":final_metrics,
             "committed":committed if maintenance else None,"candidate_map50":final_metrics["map50"] if candidate_evaluated or detector_epochs or not maintenance else None,
             "deployed_map50":state['best_map50'] if maintenance else state["current_map50"],"layer_attempts":list(state["layer_attempts"]),
             "exploration_kept":exploration_kept,"exploration_age":state['exploration_age'],"coverage":copy.deepcopy(state['coverage']),
             "current_map50":state["current_map50"],"best_map50":state["best_map50"],
             "rollback":rollback,"phase_accepted":search_result["accepted"] if search_result else None,
             "phase_enabled":state["phase_enabled"],"layer_adaptation":copy.deepcopy(controller.layers),
             "selected_layer":None if layer_index is None else layer_index+1,
             "phase_rollback":phase_rollback,"stale_cycles":state["stale"],
             "detector_loss":(summary[0]/summary[1]).item() if summary[1] else None,
             "phase_rms_from_base":space.coefficients.norm(dim=1).tolist(),
             "counts":dict(counts),"seconds":time.monotonic()-started}
        rng=capture_rng(loader,device)
        save("training_last.pth",dict(phase_space=space.state_dict(),controller=controller.state_dict(),
            optimizer=optimizer.state_dict(),run_state=state,counts=counts,best_state=best_state,rng_by_rank=rng))
        if main_rank:
            with (out/"cycles.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        if args.visualize_every and cycle%args.visualize_every==0:
            if main_rank:
                save_slm_detection_visualization(Config,cycle,teacher,student,detector,ctx["val_loader"].dataset,
                    str(out/"visualizations"),prefix="current",device=device)
                save_diagnostics(out,space,cycle)
            if world>1: dist.barrier()
        if world>1: dist.barrier()
        if state["stopped"]:
            if main_rank: print("Stopped: validation gain did not exceed min-map-gain within patience. Best pair retained.",flush=True)
            break
    write_json("result.json",dict(best_map50=state["best_map50"],completed_cycles=state["cycle"],
        early_stopped=state["stopped"],counts=counts,best_checkpoint=str(out/"detector_best.pth")))
    if ctx["tensorboard_writer"]: ctx["tensorboard_writer"].close()


if __name__=="__main__":
    try: main()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
