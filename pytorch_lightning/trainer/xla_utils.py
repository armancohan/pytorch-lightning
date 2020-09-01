import multiprocessing
import multiprocessing.connection
import torch.multiprocessing
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True
import time

from pytorch_lightning import _logger as log


# Hack: modify pytorch's start_process to support delaying starting processes
def start_delayed_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn', delay=0):
    torch.multiprocessing._python_version_check()
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=torch.multiprocessing._wrap,
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