"""Dense local-target adaptation. Only this entry uses the new optical learner."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from models.SLM.physical_defaults import add_optics_argument, save_optics_snapshot
from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.forward_distributed import initialize, capture_rng, restore_rng
from models.SLM.local_phase_learning import LocalPhaseLearner, synchronize_module, require_finite
from models.SLM.slm_train_setup import setup_training
from models.SLM.slm_utils import prepare_batch
from models.SLM.utils_slm import save_detector_best, synchronize_detector_buffers
from models.SLM.evaluation_slm import evaluate_slm_detector, save_slm_detection_visualization
from models.SLM.search_diagnostics import save_diagnostics
from models.SLM.adaptive_snapshots import AdaptiveSnapshots
from models.yolov8.feature_adapter import prepare_slm_detector_feature

REVISION = 'dense_local_target_v1'
DEFAULTS = dict(data='data/military/data.yaml',device='cuda:0',cycles=60,
    batch_size=4,workers=4,seed=42,phase_lr=1e-4,phase_batches=64,
    beta=.9,mirror_rate=.1,calibrate_every=50,phase_init='random',
    detector_epochs=0,detector_lr=1e-5,warmup_cycles=0,bn_mode='frozen',
    max_train_batches=0,visualize_every=5,patience=0,lr_patience=3,min_phase_lr=1e-6,
    paper_every=5,paper_samples=2,paper_pool=32)
SCRATCH = dict(cycles=100,phase_lr=.003,phase_batches=0,detector_epochs=1,
    detector_lr=1e-4,warmup_cycles=5,bn_mode='train')


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    add_optics_argument(parser)
    parser.add_argument('--teacher')
    parser.add_argument('--init-pair')
    parser.add_argument('--resume')
    parser.add_argument('--num-layers',type=int)
    parser.add_argument('--output',required=True)
    for key,value in DEFAULTS.items():
        parser.add_argument('--'+key.replace('_','-'),type=type(value),default=None)
    args=parser.parse_args()
    if args.init_pair and args.resume: parser.error('Use --init-pair OR --resume')
    saved=torch.load(args.resume,map_location='cpu',weights_only=True) if args.resume else None
    if saved and saved.get('forward_revision')!=REVISION:
        parser.error('Old search checkpoints cannot resume. Use paired weights with --init-pair and a new output.')
    args.training_mode=saved['arguments']['training_mode'] if saved else ('finetune' if args.init_pair else 'scratch')
    for key,default in DEFAULTS.items():
        if getattr(args,key) is None:
            setattr(args,key,saved['arguments'].get(key,default) if saved else SCRATCH.get(key,default) if args.training_mode=='scratch' else default)
    if saved:
        args.teacher=args.teacher or saved['arguments']['teacher']
        args.num_layers=args.num_layers or saved['num_layers']
        for key in DEFAULTS:
            if key not in {'cycles','workers','device','visualize_every','patience','paper_every','paper_samples','paper_pool'} and getattr(args,key)!=saved['arguments'][key]:
                parser.error(f'Resume changed --{key.replace("_","-")}')
    if not args.teacher: parser.error('--teacher is required to identify the matching architecture/checkpoint')
    for key in ('teacher','init_pair','resume','data'):
        value=getattr(args,key)
        if value and not Path(value).is_file(): parser.error(f'Missing {key}: {value}')
    for key in ('cycles','batch_size','lr_patience','paper_pool'):
        if getattr(args,key)<1: parser.error(f'{key} must be positive')
    for key in ('workers','phase_batches','detector_epochs','warmup_cycles','max_train_batches','visualize_every','patience','calibrate_every','paper_every','paper_samples'):
        if getattr(args,key)<0: parser.error(f'{key} must be nonnegative')
    for key in ('phase_lr','detector_lr','min_phase_lr'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0: parser.error(f'{key} must be positive finite')
    if not 0<args.beta<=1 or not 0<=args.mirror_rate<=1: parser.error('0<beta<=1 and 0<=mirror-rate<=1')
    if args.phase_init not in ('zero','random') or args.bn_mode not in ('train','frozen'): parser.error('phase-init=zero|random; bn-mode=train|frozen')
    if args.num_layers is not None and args.num_layers not in (1,2,3,4): parser.error('num-layers must be 1..4')
    if args.min_phase_lr>args.phase_lr: parser.error('min-phase-lr must not exceed phase-lr')
    if args.paper_samples>args.paper_pool: parser.error('paper-samples must not exceed paper-pool')
    if args.warmup_cycles>=args.cycles: parser.error('warmup-cycles must be less than cycles')
    if (args.training_mode=='scratch' or args.warmup_cycles) and args.detector_epochs<1:
        parser.error('Scratch/warmup requires detector-epochs >= 1')
    return args,saved


def cpu_copy(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    if isinstance(value,Path): return str(value)
    return copy.deepcopy(value)


def restore_pair(student,detector,checkpoint):
    if checkpoint.get('phase_parameterization','direct_sgd')!='direct_sgd':
        raise ValueError('Use a direct_sgd phase-only paired checkpoint')
    student.load_state_dict(checkpoint['student_state_dict'],strict=True)
    detector.load_state_dict(checkpoint['detector_state_dict'],strict=True)
    student.enable_norm=bool(checkpoint.get('student_enable_norm',student.enable_norm))
    Config.STUDENT_NORM_MODE=checkpoint.get('student_norm_mode',Config.STUDENT_NORM_MODE)


def dataset_identity(ctx):
    digest=hashlib.sha256()
    for name in ('train_loader','val_loader'):
        digest.update(name.encode())
        for entry in ctx[name].dataset.entries:
            for key in ('image_path','label_path'):
                path=Path(entry[key]).resolve()
                digest.update(str(path).encode())
                if path.exists():
                    digest.update(str(path.stat().st_size).encode())
                    if key=='label_path': digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    args,saved=arguments()
    device,rank,world=initialize(args.device)
    is_main=rank==0
    out=Path(args.output).resolve()
    if saved and Path(args.resume).resolve().parent!=out: raise ValueError('Resume into the original output directory')
    occupied=torch.tensor(int(out.exists() and any(out.iterdir()) and not saved),device=device)
    if world>1:dist.all_reduce(occupied,op=dist.ReduceOp.MAX)
    if occupied.item(): raise FileExistsError('Use a new output directory')
    Config.YAML_PATH,Config.OUTPUT_DIR=args.data,str(out)
    Config.TEACHER_DETECTOR_CHECKPOINT=args.teacher
    Config.DEVICE,Config.GPU_IDS=str(device),([device.index] if device.type=='cuda' else [])
    Config.BATCH_SIZE,Config.NUM_WORKERS,Config.TRAIN_SEED=args.batch_size,args.workers,args.seed
    Config.SLM_PHASE_PARAM_MODE,Config.SLM_INIT_MODE='direct_sgd',args.phase_init
    if args.phase_init=='zero': Config.SLM_INIT_NOISE_STD=0.
    Config.SLM_MULTI_HEAD_ENABLED,Config.STUDENT_NORM_SCHEDULE=False,'always'
    Config.ENABLE_CHANNELS_LAST=False
    # Recreate workers from the saved loader generator at epoch boundaries.
    Config.PERSISTENT_WORKERS=False
    if args.num_layers: Config.configure_num_layers(args.num_layers)
    ctx=setup_training(is_main,world>1,forward_only=True,detector_from_teacher=False)
    student,detector,teacher=ctx['student_raw'],ctx['detector_raw'],ctx['teacher']
    if ctx['val_loader'] is None or not len(ctx['val_loader'].dataset): raise ValueError('A nonempty validation split is required')
    if len(ctx['train_loader'])<1: raise ValueError('No complete training batch on this rank')
    student.requires_grad_(False).eval()
    checkpoint=saved or (torch.load(args.init_pair,map_location='cpu',weights_only=True) if args.init_pair else None)
    if checkpoint is not None:
        if checkpoint.get('teacher_detector_sha256',ctx['teacher_checkpoint_sha256'])!=ctx['teacher_checkpoint_sha256']:
            raise ValueError('The pair and --teacher have different teacher identities')
        restore_pair(student,detector,checkpoint)
    synchronize_module(student);synchronize_module(detector)
    learner=LocalPhaseLearner(student,lr=args.phase_lr,beta=args.beta,mirror_rate=args.mirror_rate)
    detector_optimizer=torch.optim.AdamW(detector.parameters(),lr=args.detector_lr,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(learner.optimizer,mode='min',factor=.5,
        patience=args.lr_patience,min_lr=args.min_phase_lr)
    fingerprint=dataset_identity(ctx)
    if saved:
        if saved['world_size']!=world or saved['dataset_identity']!=fingerprint:
            raise ValueError('Resume requires the same world size and dataset; otherwise use --init-pair')
        learner.load_state_dict(saved['local_learner'])
        detector_optimizer.load_state_dict(saved['detector_optimizer'])
        scheduler.load_state_dict(saved['phase_scheduler'])
    feedback_before=[None if x is None else x.clone() for x in learner.theta]
    count_before=learner.calibration_count
    learner.calibrate()
    gray=ctx['train_loader'].dataset[0]['gray_tensor'][None].to(device)
    diagnostic=learner.diagnose(gray)
    learner.theta=feedback_before;learner.calibration_count=count_before
    state=saved['run_state'] if saved else dict(cycle=0,best_map50=-1.,best_loss=float('inf'),stale=0)
    if saved: restore_rng(saved['rng_by_rank'][rank],ctx['train_loader'],device)
    snapshots = AdaptiveSnapshots(out,Config,ctx['val_loader'].dataset,learner,detector,
        ctx['detection_criterion'],samples=args.paper_samples,pool=args.paper_pool,seed=args.seed) if is_main and args.paper_samples and args.paper_every else None

    def snapshot_action(action):
        # Rank-zero exports never call learner.gradients/step or distributed validation.
        # Synchronize here so other GPUs cannot run ahead into gradient collectives.
        error=None; result=None
        if is_main:
            try: result=action()
            except Exception as exc: error=exc
        failed=torch.tensor(int(error is not None),device=device)
        if world>1: dist.all_reduce(failed,op=dist.ReduceOp.MAX)
        if failed.item():
            if error is not None: raise error
            raise RuntimeError('Rank-zero paper snapshot failed; inspect its traceback')
        return result

    class EvaluationConfig(Config):
        @classmethod
        def get_stage_loss_weights(cls,stage_name):
            return dict(feature=0.,detection=1.,response=0.,phase_regularization=0.)

    def validate():
        detector.eval();synchronize_detector_buffers(detector)
        losses,metrics=evaluate_slm_detector(EvaluationConfig,teacher,student,detector,ctx['val_loader'],
            ctx['detection_criterion'],ctx['feature_criterion'],device,'dense_local')
        require_finite([torch.tensor([losses['detection'],metrics['map50']],device=device)],'Non-finite validation')
        metrics['validation_detection_loss']=losses['detection']
        return metrics

    def write_json(name,value):
        if is_main: (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

    def save_pair(name,metrics,extra=None):
        if not is_main:return
        payload=dict(forward_revision=REVISION,arguments=vars(args),world_size=world,dataset_identity=fingerprint,
            teacher_detector_sha256=ctx['teacher_checkpoint_sha256'],student_norm_mode=Config.STUDENT_NORM_MODE,
            student_norm_schedule=Config.STUDENT_NORM_SCHEDULE,architecture_revision='contextdw_v1',
            detector_head_type=Config.DETECTOR_HEAD_TYPE,light_base_ch=Config.YOLO_LIGHT_BASE_CH,
            val_map50=metrics['map50'],metrics=metrics,selection_metric='detector_val_map50',
            training_scope='Dense local-target numerical adaptation with model-assisted calibrated feedback')
        if extra: payload.update(extra)
        temp=out/(name+'.tmp')
        save_detector_best(detector,temp,state['cycle'],metrics['validation_detection_loss'],
            extra=cpu_copy(payload),student=student,config=Config)
        os.replace(temp,out/name)

    if is_main:
        save_optics_snapshot(out,num_layers=Config.NUM_LAYERS)
        write_json('arguments.json',cpu_copy(vars(args)))
        write_json('local_diagnostic.json',diagnostic)
        print(f'{REVISION}: {args.training_mode}, {len(learner.layers)} SLM layers; full-pixel local targets, calibrated feedback; no phase search.',flush=True)
    if not saved:
        metrics=validate()
        state.update(best_map50=metrics['map50'],best_loss=metrics['validation_detection_loss'])
        write_json('initial_metrics.json',metrics)
        save_pair('detector_best.pth',metrics)

    for cycle in range(state['cycle']+1,args.cycles+1):
        started=time.perf_counter();state['cycle']=cycle
        warming=cycle<=args.warmup_cycles
        phase_rows=[];detector_sum=0.;detector_count=0
        detector.requires_grad_(False).eval()
        record_phase = bool(args.paper_samples and args.paper_every and not warming and
            (cycle==args.warmup_cycles+1 or cycle%args.paper_every==0 or cycle==args.cycles))
        before_snapshot = snapshot_action(lambda: snapshots.capture(local_signals=True)) if record_phase else None
        if not warming:
            if ctx['train_sampler']: ctx['train_sampler'].set_epoch(cycle*1000)
            for index,batch in enumerate(tqdm(ctx['train_loader'],desc=f'Local phase {cycle}/{args.cycles}',disable=not is_main)):
                if args.phase_batches and index>=args.phase_batches: break
                if args.calibrate_every and learner.updates and learner.updates%args.calibrate_every==0:
                    learner.calibrate()
                gray,_,targets=prepare_batch(batch,device)
                def objective(feature):
                    outputs=detector(prepare_slm_detector_feature(Config,feature))
                    return ctx['detection_criterion'](outputs,targets)[0]
                row=learner.step(gray,objective)
                phase_rows.append(row)
                if is_main:
                    with (out/'phase_steps.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(cycle=cycle,batch=index,**row))+'\n')
        if record_phase:
            snapshot_action(lambda: snapshots.save(cycle,before_snapshot))
        if args.detector_epochs:
            detector.requires_grad_(True)
            for epoch in range(args.detector_epochs):
                if ctx['train_sampler']: ctx['train_sampler'].set_epoch(cycle*1000+epoch+1)
                ctx['detector'].train()
                if args.bn_mode=='frozen':
                    for module in detector.modules():
                        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
                for index,batch in enumerate(tqdm(ctx['train_loader'],desc=f'Detector {cycle}/{args.cycles}',disable=not is_main)):
                    if args.max_train_batches and index>=args.max_train_batches: break
                    gray,_,targets=prepare_batch(batch,device)
                    with torch.no_grad(): feature=student(gray)
                    detector_optimizer.zero_grad(set_to_none=True)
                    outputs=ctx['detector'](prepare_slm_detector_feature(Config,feature).contiguous())
                    loss=ctx['detection_criterion'](outputs,targets)[0]
                    require_finite([loss],'Non-finite detector loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(detector.parameters(),10.,error_if_nonfinite=True)
                    detector_optimizer.step();detector_sum+=float(loss.detach());detector_count+=1
            synchronize_detector_buffers(detector)
        detector.requires_grad_(False).eval()
        metrics=validate()
        improved=(metrics['map50']>state['best_map50'] or
            (metrics['map50']==state['best_map50'] and metrics['validation_detection_loss']<state['best_loss']))
        if improved:
            state.update(best_map50=metrics['map50'],best_loss=metrics['validation_detection_loss'],stale=0)
            save_pair('detector_best.pth',metrics)
        elif not warming: state['stale']+=1
        if phase_rows: scheduler.step(metrics['validation_detection_loss'])
        sums=torch.tensor([detector_sum,detector_count],dtype=torch.float64,device=device)
        if world>1:dist.all_reduce(sums)
        row=dict(cycle=cycle,stage='detector_warmup' if warming else 'dense_local',metrics=metrics,
            best_map50=state['best_map50'],phase_updates=len(phase_rows),
            phase=phase_rows[-1] if phase_rows else None,calibration_count=learner.calibration_count,
            detector_loss=float(sums[0]/sums[1]) if sums[1]>0 else None,
            phase_lr=learner.optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-started)
        if is_main:
            with (out/'cycles.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
            writer=ctx['tensorboard_writer']
            if writer:
                writer.add_scalar('validation/map50',metrics['map50'],cycle)
                writer.add_scalar('validation/detection_loss',metrics['validation_detection_loss'],cycle)
                writer.add_scalar('phase/learning_rate',row['phase_lr'],cycle)
                if phase_rows:
                    for i,rms in enumerate(phase_rows[-1]['phase_rms_from_start'],1): writer.add_scalar(f'phase/slm{i}_change_rms',rms,cycle)
            if args.visualize_every and cycle%args.visualize_every==0:
                save_slm_detection_visualization(Config,cycle,teacher,student,detector,ctx['val_loader'].dataset,
                    str(out/'visualizations'),prefix='current',device=device)
                save_diagnostics(out,learner,cycle)
            print(json.dumps(row),flush=True)
        if world>1:dist.barrier()
        rng=capture_rng(ctx['train_loader'],device)
        save_pair('forward_last.pth',metrics,dict(local_learner=learner.state_dict(),
            detector_optimizer=detector_optimizer.state_dict(),phase_scheduler=scheduler.state_dict(),
            run_state=state,rng_by_rank=rng))
        if world>1:dist.barrier()
        if args.patience and state['stale']>=args.patience: break
    write_json('result.json',dict(**state,revision=REVISION,best_checkpoint=str(out/'detector_best.pth'),
        resume_checkpoint=str(out/'forward_last.pth'),updates=learner.updates))
    if ctx['tensorboard_writer']:ctx['tensorboard_writer'].close()


if __name__=='__main__':
    try:main()
    finally:
        if dist.is_initialized():dist.destroy_process_group()
