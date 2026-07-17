# NOTICE: Modified from QED-Nano (github.com/CMU-AIRe/QED-Nano @ 02a4699), Apache-2.0.
# Change: pass InitProcessGroupKwargs(timeout=4h) to the Accelerator so a transient
# actor hiccup can't trip the 1800s NCCL c10d watchdog and kill finetune.
# See ../../../UPSTREAM.md.
import logging
from datetime import timedelta

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

logger = logging.getLogger(__name__)

# step_scheduler_with_optimizer=False prevents the scheduler
# from being stepped multiple times in the multi-gpu setting.
# (The default behavior in AcceleratedScheduler when split_batches=False is to
#   step() "num_processes" times, because they expect the lr schedule to
#   depend on processed samples/epochs, not completed_steps)

# Raise the collective (NCCL) watchdog timeout well above the PyTorch default of
# 30 min. In async RL the finetune ranks can legitimately wait a long time for
# the actor to refill the batch queue (e.g. after the actor drops oversized
# rollouts or briefly stalls); the default 1800 s timeout would otherwise abort
# the whole job. This is a fault-detection patience knob only and does not
# affect training results.
_PROCESS_GROUP_TIMEOUT = timedelta(hours=4)

_accelerator = None


def get_accelerator():
    global _accelerator
    if _accelerator is None:
        _accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=_PROCESS_GROUP_TIMEOUT)],
        )
    return _accelerator
