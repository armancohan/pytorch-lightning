import sys

import torch.multiprocessing

import torch.multiprocessing.spawn

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as xloader
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True
import time

from pytorch_lightning import _logger as log
from torch.multiprocessing import _prctl_pr_set_pdeathsig
import signal
import threading
from six import iteritems, itervalues
import time


# Multiprocessing contexts are introduced at Python 3.4
_supports_context = sys.version_info >= (3, 4)

def _python_version_check():
    if not _supports_context:
        raise RuntimeError("Requires python 3.4 or higher to use "
                           "torch.multiprocessing.spawn and "
                           "torch.multiprocessing.ProcessContext helper "
                           "to launch multiple processes. If you are using "
                           "this for distributed training and have a lower "
                           "version of python, please use "
                           "torch.distributed.launch instead.")

def _wrap(fn, i, args, error_queue):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)

# Hack: modify pytorch's start_process to support delaying starting processes
def start_delayed_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn', delay=0):
    _python_version_check()
    mp = torch.multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)
        if delay > 0:
            log.info(f"Sleeping thread for {delay} seconds")
            time.sleep(delay)

    context = torch.multiprocessing.ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def delayed_spawn(fn,
                  args=(),
                  nprocs=None,
                  join=True,
                  daemon=False,
                  start_method='spawn',
                  delay=0):
  """Enables multi processing based replication.
  Args:
    fn (callable): The function to be called for each device which takes part of
      the replication. The function will be called with a first argument being
      the global index of the process within the replication, followed by the
      arguments passed in `args`.
    args (tuple): The arguments for `fn`.
      Default: Empty tuple
    nprocs (int): The number of processes/devices for the replication. At the
      moment, if specified, can be either 1 or the maximum number of devices.
    join (bool): Whether the call should block waiting for the completion of the
      processes which have being spawned.
      Default: True
    daemon (bool): Whether the processes being spawned should have the `daemon`
      flag set (see Python multi-processing API).
      Default: False
    start_method (string): The Python `multiprocessing` process creation method.
      Default: `spawn`
    delay (int): Delay thread starting for `delay` seconds
  Returns:
    The same object returned by the `torch.multiprocessing.spawn` API. If
    `nprocs` is 1 the `fn` function will be called directly, and the API will
    not return.
  """
  if not xmp._is_xla_config():
    # If this is not an XLA setup, jump to normal multi-processing.
    return xmp._run_direct(fn, args, nprocs, join, daemon, start_method)

  pf_cfg = xmp._pre_fork_setup(nprocs)
  if pf_cfg.num_devices == 1:
    xmp._start_fn(0, pf_cfg, fn, args)
  else:
    return start_delayed_processes(
        xmp._mp_start_fn,
        args=(pf_cfg, fn, args),
        nprocs=pf_cfg.num_devices,
        join=join,
        daemon=daemon,
        start_method=start_method,
        delay=delay)



class DelayedParallelLoader(xloader.ParallelLoader):
  """Wraps an existing PyTorch DataLoader with background data upload.
  Args:
    loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    devices (`torch.device`...): The list of devices where the data has to be
      sent. The i-th sample returned by the `loader` will be sent to `devices[i
      % len(devices)]`.
    batchdim (int, optional): The dimension which is holding the batch size.
      Default: 0
    fixed_batch_size (bool, optional): Ensures that all the batch sizes sent to
      the devices are of the same size. The original `loader` iteration stops as
      soon as a not matching batch size is found.
      Default: False
    loader_prefetch_size (int, optional): The max capacity of the queue used by
      the thread which is reading samples from the `loader`, to be processed by
      the worker threads which upload data to the devices.
      Default: 8
    device_prefetch_size (int, optional): The max size of the per-device queues,
      where the worker threads deposit tensors which have already been sent to
      devices.
      Default: 4
  """

  def __init__(self,
               loader,
               devices,
               batchdim=0,
               fixed_batch_size=False,
               loader_prefetch_size=8,
               device_prefetch_size=4,
               delay=0):
    self._loader = loader
    self._devices = [torch.device(x) for x in devices]
    self._batchdim = batchdim
    self._fixed_batch_size = fixed_batch_size
    self._per_device_samples = len(loader) // len(devices)
    self._done = False
    self._queues = dict()
    for device in self._devices:
      self._queues[device] = xloader.PerDeviceQueue(device, loader_prefetch_size,
                                            device_prefetch_size)
    thread = threading.Thread(target=self._loader_worker)
    thread.daemon = True
    thread.start()
    for dqueue in itervalues(self._queues):
      thread = threading.Thread(target=self._worker, args=(dqueue,))
      thread.daemon = True
      thread.start()
      if delay > 0:
        time.sleep(delay)