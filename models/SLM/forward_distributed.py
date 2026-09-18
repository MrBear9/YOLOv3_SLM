"""Distributed coordination for one shared forward-search experiment."""
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


def initialize(device):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    selected = torch.device(device)
    if selected.type == "cuda":
        # torchrun owns local device assignment; --device controls only single-process runs.
        selected = torch.device("cuda", local_rank if world_size > 1 else (selected.index or 0))
        torch.cuda.set_device(selected)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if selected.type == "cuda" else "gloo",
            init_method="env://", timeout=timedelta(minutes=30),
        )
    return selected, dist.get_rank() if dist.is_initialized() else 0, world_size


@torch.no_grad()
def sync_initial_state(module, phase_tensors=()):
    """Copy rank 0's loaded state without recording communication in autograd.

    DDP handles subsequent gradient reductions; initialization is not a
    differentiable operation. Preserve parameter objects and their strides.
    """
    if dist.is_initialized():
        for tensor in list(module.parameters()) + list(module.buffers()) + list(phase_tensors):
            dist.broadcast(tensor, src=0)


@torch.no_grad()
def mean_measurement(values):
    """Equal-size train shards define the mean of per-rank loss vectors."""
    if dist.is_initialized():
        dist.all_reduce(values)
        values.div_(dist.get_world_size())
    return values


@torch.no_grad()
def sync_buffers(module):
    # Ordinary DDP BN updates are local. Use rank 0's buffers on every rank
    # during phase measurements, validation and checkpoint selection.
    if dist.is_initialized():
        for buffer in module.buffers():
            dist.broadcast(buffer, src=0)


def capture_rng(loader, device):
    ns = np.random.get_state()
    state = {"torch_rng": torch.get_rng_state(),
             "cuda_rng_local": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
             "python_rng": random.getstate(),
             "numpy_rng": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
             "loader_rng": loader.generator.get_state()}
    if not dist.is_initialized():
        return [state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states


def restore_rng(state, loader, device):
    torch.set_rng_state(state["torch_rng"].cpu())
    if device.type == "cuda":
        if state.get("cuda_rng_local") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_local"].cpu(), device)
        elif state.get("cuda_rng"):
            # Compatibility with the original one-process training checkpoint.
            torch.cuda.set_rng_state(state["cuda_rng"][0].cpu(), device)
    random.setstate(state["python_rng"])
    ns = state["numpy_rng"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    loader.generator.set_state(state["loader_rng"].cpu())
