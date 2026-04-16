"""File-based barrier and stop-signal for synchronizing two independent training jobs."""

import os
import time

import torch
import torch.distributed as dist


def wait_for_partner(mode: str, model_name: str, partner_name: str,
                     barrier_dir: str, rank: int, timeout: float = 300):
    """Block until the partner job is also ready, so both start training together.

    Only effective in pair mode. In solo mode, returns immediately.
    Only rank 0 participates in file-based signaling; other ranks wait via dist.barrier().
    """
    if mode != "pair" or not barrier_dir:
        return

    if rank == 0:
        # Signal this job is ready
        sentinel = os.path.join(barrier_dir, f"{model_name}.ready")
        with open(sentinel, "w") as f:
            f.write(str(os.getpid()))

        # Wait for partner
        partner_sentinel = os.path.join(barrier_dir, f"{partner_name}.ready")
        t0 = time.time()
        while not os.path.exists(partner_sentinel):
            if time.time() - t0 > timeout:
                print(f"[BARRIER] WARNING: timed out waiting for {partner_name} after {timeout}s")
                break
            time.sleep(0.05)

        print(f"[BARRIER] {model_name} and {partner_name} synchronized, starting training")

    # All ranks wait for rank 0
    dist.barrier()


def signal_done(barrier_dir: str, model_name: str, rank: int):
    """Primary job calls this after finishing all steps to tell the interferer to stop."""
    if not barrier_dir:
        return
    if rank == 0:
        done_path = os.path.join(barrier_dir, f"{model_name}.done")
        with open(done_path, "w") as f:
            f.write("done")
        print(f"[BARRIER] {model_name} signaled done")


def should_stop(barrier_dir: str, partner_name: str, rank: int) -> bool:
    """Interferer job checks this each step. Returns True when primary is done.

    Only rank 0 checks the file; result is broadcast to all ranks via dist.barrier
    would be too expensive per-step, so we just let rank 0 decide and broadcast.
    """
    if not barrier_dir:
        return False
    if rank == 0:
        done_path = os.path.join(barrier_dir, f"{partner_name}.done")
        stop = os.path.exists(done_path)
    else:
        stop = False

    # Broadcast rank 0's decision to all ranks
    stop_tensor = torch.tensor([1 if stop else 0], device="cuda")
    dist.broadcast(stop_tensor, src=0)
    return stop_tensor.item() == 1


