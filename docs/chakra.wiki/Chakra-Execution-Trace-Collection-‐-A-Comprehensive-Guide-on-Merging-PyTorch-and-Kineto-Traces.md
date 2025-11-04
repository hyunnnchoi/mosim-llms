**Authors**: Saeed Rashidi (HPE), Joongun Park (Georgia Tech), Abhilash Kolluri (NVIDIA), Taekyung Heo (NVIDIA)

## 1. Introduction
This document outlines the process of collecting and simulating Chakra execution traces for performance projection and design space exploration using a simulator. This document covers the collection of PyTorch execution traces (ET) and Kineto traces, their linker, and the subsequent conversion into Chakra execution traces, a standardized format that encapsulates both CPU and GPU operation information.

## 2. Overview of Trace Collection and Simulation Methodology
Chakra execution traces and the related toolchains enable the simulation of execution traces on a simulator. The figure below illustrates how the end-to-end flow works. The process begins by collecting traces from a PyTorch model. There are two types of traces collected from PyTorch: PyTorch ET and Kineto trace. We need to collect two different types of traces because each trace type covers aspects that the other cannot. While PyTorch ETs focus on CPU operators with explicit dependencies between them, Kineto traces encode GPU operators with their start and end times. To understand the differences between further, please refer to the table below, which highlights their differences and roles. After collecting these traces, we use a merger tool (`chakra_trace_link`) to merge them into a single execution trace, known as PyTorch ET+. This format essentially follows the PyTorch ET schema but also encodes GPU operators and their dependencies. Subsequently, these traces are converted into the Chakra schema using the converter (`chakra_converter`). Finally, you can use any Chakra-compatible simulator, with ASTRA-sim currently serving as a reference implementation.

<p align="center">
  <img width="505" a lt="Screenshot 2024-01-03 at 4 10 32 PM" src="https://github.com/mlcommons/chakra/assets/7621438/67228699-cec5-4a4d-b03e-e76647a80ce8">
</p>

| Trace Data Category | PyTorch ET | Kineto Trace |
|---------------------|------------|--------------|
| **Event Timestamps** | No | Yes |
| **Host Events** | Yes | Yes |
| **Device (GPU) Events** | No | Yes |
| **Operator Inputs** | Yes | Partial |
| **Operator Outputs** | Yes | No |
| **Events Hierarchy** | Explicit (call stack) | Implicit (time-based) |
| **Operator Schema** | Yes | No |
| **Data Dependencies** | Yes | No |
| **Comms Data** | Yes | No |

## 3. From Raw Traces to Chakra: A Step-by-Step Conversion Guide
This section offers a comprehensive guide on collecting traces and converting them into Chakra traces, with a specific focus on simultaneous collection methods for PyTorch execution traces and Kineto traces. For clarity, the collection process for each trace type will be explained individually before detailing the simultaneous collection method. Please note, the procedures described here have been tested and are confirmed to work with PyTorch version 2.1.2. 

### Collecting PyTorch Execution Traces
You can collect PyTorch execution traces from a PyTorch model's execution. This is achieved by using the [ExecutionTraceObserver](https://github.com/pytorch/pytorch/blob/main/torch/csrc/profiler/standalone/execution_trace_observer.cpp) implemented in PyTorch. The process involves instantiating the observer, registering a callback, and initiating profiling. Although you have the flexibility to collect as many execution traces as desired, for training jobs, profiling a single iteration is advisable for optimal results. To gather these traces, set up the observer and control the start and stop of the profiling. Below is a scripting example for profiling execution traces:
```
from torch.profiler import _ExperimentalConfig, ExecutionTraceObserver

et = ExecutionTraceObserver()
et.register_callback("pytorch_et.json")
et.start()
...
et.stop()
et.unregister_callback()
```
An implementation example of the ExecutionTraceObserver can be found in [the param benchmark code](https://github.com/facebookresearch/param/blob/main/train/compute/python/pytorch/run_benchmark.py), which illustrates how to collect execution traces from PyTorch.

### Collecting Kineto Traces
Next, it's essential to collect Kineto traces, which shed light on the GPU operators within the model. You can collect Kineto traces with `torch.profiler.profile`. When using `torch.profiler.profile`, it's important to supply the correct arguments to ensure accurate collection of Kineto traces. Additionally, ensure that prof.step() is called at the end of each iteration. The process includes a warm-up phase, during which the profiler begins tracing but discards the results, followed by an active tracing phase where the profiler traces and records data. Further details can be found in [the PyTorch manual](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs).
```
import torch

def trace_handler(prof):
    prof.export_chrome_trace("./kineto_trace.json")

def main():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=1),
        record_shapes=True,
        on_trace_ready=trace_handler,
    ) as prof:
        ...
        prof.step()
```

### Simultaneous Collection of PyTorch Execution and Kineto Traces
To ensure that traces are linked in the following steps, it's essential to collect PyTorch execution traces and Kineto traces simultaneously during model execution. This approach ensures that the traces align perfectly in terms of timing and events. To achieve this, integrate both the ExecutionTraceObserver and Kineto profiling within the same epoch. Here's an adapted example demonstrating this method:
```
import torch
from torch.profiler import ExecutionTraceObserver, profile

def trace_handler(prof):
    prof.export_chrome_trace("kineto_trace.json")

def main():
    et = ExecutionTraceObserver()
    et.register_callback("pytorch_et.json")
    et.start()

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
        on_trace_ready=trace_handler
    ) as prof:
        for epoch in ...:
            ...
            if epoch == 5:
                et.stop()
            if epoch == 4:
                et.start()
            ...
            prof.step()

    et.stop()
    et.unregister_callback()
```

### Installing Chakra and Param
Next, we need to install the Chakra and Param.
```
$ git clone --recurse-submodules git@github.com:mlcommons/chakra.git
$ cd chakra
$ pip3 install .
```

```
$ git clone git@github.com:facebookresearch/param.git
$ cd param/et_replay
$ git checkout 7b19f586dd8b267333114992833a0d7e0d601630
$ pip install .
```

### Merging Traces with `chakra_trace_link`
Next, you will need to merge the PyTorch execution trace with the Kineto trace. To accomplish this, utilize `chakra_trace_link`. This tool facilitates the merging of a PyTorch ET and a Kineto trace into a single, unified PyTorch ET+. It is important to note that this merging process must be performed for each pair of PyTorch execution trace and Kineto trace. The commands below guide you through this process:
```
$ chakra_trace_link --chakra-host-trace /path/to/chakra_host_trace --chakra-device-trace /path/to/chakra_device_trace --output-file /path/to/chakra_host_device_trace.json
```

### Converting to Chakra Execution Trace with `chakra_converter`
Next, the merged PyTorch execution trace plus trace is converted into the Chakra execution trace, making it suitable for simulation and analysis.
```
$ chakra_converter PyTorch --input /path/to/chakra_host_device_trace.json --output /path/to/chakra_trace
```

## 4. Practical Example: Trace Collection for ResNet-50
### Collecting PyTorch Execution Traces
We recommend to use PyTorch 2.5.0 or later version to avoid compatibility issue.

In this section, we demonstrate how to collect Chakra execution traces using a straightforward example of ResNet-50. First, we start by implementing a function for ResNet-50 in PyTorch. 
```
#!/usr/bin/env python3

import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall
from torch.autograd.profiler import KinetoStepTracker, profile as _profile
from torch.autograd.profiler_legacy import profile as _profile_legacy
from torch.profiler import (
    _utils,
    DeviceType,
    ExecutionTraceObserver,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch.profiler._pattern_matcher import (
    Conv2dBiasFollowedByBatchNorm2dPattern,
    ExtraCUDACopyPattern,
    ForLoopIndexingPattern,
    FP32MatMulPattern,
    GradNotSetToNonePattern,
    MatMulDimInFP16Pattern,
    NamePattern,
    OptimizerSingleTensorPattern,
    Pattern,
    report_all_anti_patterns,
    SynchronizedDataLoaderPattern,
)
```
This section imports the necessary libraries. `PyTorch` and `TorchVision` provide model definition, training routines, and dataset utilities. `DistributedDataParallel` is used for multi-GPU training, while torch.profiler components such as `ExecutionTraceObserver` and `supported_activities` handle the collection of Execution Trace (ET) and Kineto traces. `_ExperimentalConfig` enables CUDA synchronization for more accurate GPU timestamps.


```

def example(rank, use_gpu=True):
    # Register Execution Trace Observer
    eg_file = "./host_" + str(rank) + ".json"
    kineto_file = "./device_" + str(g_rank)+".json"

    # Define global variable for custom trace_handler
    global g_rank
    g_rank = rank
```

The example function contains the main training routine that will be executed by each process. `eg_file` and `kineto_file` store the path where the traces will be saved for the given rank.

```
    if use_gpu:
        torch.cuda.set_device(rank)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(rank)
        model.cuda()
        cudnn.benchmark = True
        model = DDP(model, device_ids=[rank])
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = DDP(model)

    # Use gradient compression to reduce communication
    # model.register_comm_hook(None, default.fp16_compress_hook)
    # or
    # state = powerSGD_hook.PowerSGDState(process_group=None,matrix_approximation_rank=1,start_powerSGD_iter=2)
    # model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)
```

If GPU usage is enabled, the script sets the current CUDA device based on the rank, loads a pretrained ResNet-50 model, moves it to the appropriate device, and wraps the model in `DistributedDataParallel`. 

```
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler,
                                              shuffle=False, num_workers=4)

    if use_gpu:
        criterion = nn.CrossEntropyLoss().to(rank)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
```

This block prepares the CIFAR-10 training dataset. Images are resized to 256 pixels, center-cropped to 224 pixels, and converted to tensors. A DistributedSampler ensures that each process works on a unique subset of the dataset. The DataLoader uses a batch size of 32 and multiple worker threads for data loading. The loss function is set to cross-entropy, moved to the GPU if applicable. 

```
    print(supported_activities()) # Should Include CUDA
    with torch.profiler.profile(
        activities=supported_activities(),
        schedule=torch.profiler.schedule(
            wait=0, warmup=0, active=1, repeat=1
        ),
        with_stack=False,
        execution_trace_observer=(
            ExecutionTraceObserver().register_callback(eg_file)
        ),
        experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
        record_shapes=True
    ) as p:
        for step, data in enumerate(trainloader, 0):
            print("step:{}".format(step))
            if use_gpu:
                inputs, labels = data[0].to(rank), data[1].to(rank)
            else:
                inputs, labels = data[0], data[1]

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p.step()

            # Changed termination condition
            if step + 1 >= 1:
                break
        p.export_chrome_trace(kineto_file)
```

This is where profiling starts. Both CPU and GPU activities are captured using `supported_activities()`. The schedule specifies no wait, no warmup, and a single active step, ensuring that the ET (`eg_file`) and Kineto (`kineto_file`) traces correspond to the exact same execution window for correct Chakra trace linking. The ET observer is registered to save the host-side trace to `eg_file`, while the Kineto profiler will later export the device-side trace to `kineto_file`. After training ends, the profiler exports the Kineto device trace to kineto_file in Chrome Trace format. At this point, each rank will have both a host trace and a device trace file saved in the ./result directory.

```
def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2 # Two GPUs
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, example))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

The init_process function sets up the distributed environment. It configures the master address and port, initializes the NCCL process group for multi-GPU communication, and calls the given training function.

```
$ cd ~/
$ python -m venv venv
$ source venv/bin/activate
$ pip install numpy torch
$ python resnet-50.py
```

After running the program, you will find `host_[rank].json` in your working directory. 
When you open it, you can find the following json data.

```
{
  "schema": "1.1.1-chakra.0.0.4", "pid": 2784631, "time": "2025-04-11 13:43:51", "start_ts": 2188364098,
  "nodes": [
    {
      "id": 2, "name": "[pytorch|profiler|execution_trace|thread]", "ctrl_deps": 1,
      "inputs": {"values": [], "shapes": [], "types": [], "strides": []},
      "outputs": {"values": [], "shapes": [], "types": [], "strides": []},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 0},{"name": "fw_parent", "type": "uint64", "value": 0},{"name": "seq_id", "type": "int64", "value": -1},{"name": "scope", "type": "uint64", "va
lue": 7},{"name": "tid", "type": "uint64", "value": 1},{"name": "fw_tid", "type": "uint64", "value": 0},{"name": "op_schema", "type": "string", "value": ""},{"name": "kernel_backend", "type": "string", "v
alue": ""},{"name": "kernel_file", "type": "string", "value": ""}]
    },
    {
      "id": 3, "name": "## process_group:init ##", "ctrl_deps": 2,
      "inputs": {"values": ["[{\"pg_name\": \"0\", \"pg_desc\": \"default_pg\", \"backend_config\": \"cuda:nccl\", \"ranks\": [], \"group_size\": 8, \"group_count\": 1}]"], "shapes": [[]], "types": ["Stri
ng"], "strides": [[]]},
      "outputs": {"values": [], "shapes": [], "types": [], "strides": []},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 1},{"name": "fw_parent", "type": "uint64", "value": 0},{"name": "seq_id", "type": "int64", "value": -1},{"name": "scope", "type": "uint64", "va
lue": 7},{"name": "tid", "type": "uint64", "value": 1},{"name": "fw_tid", "type": "uint64", "value": 0},{"name": "op_schema", "type": "string", "value": ""},{"name": "kernel_backend", "type": "string", "v
alue": ""},{"name": "kernel_file", "type": "string", "value": ""}]
    },
    {
      "id": 5, "name": "aten::empty", "ctrl_deps": 4,
      "inputs": {"values": [[],4,"<None>","cpu",false,"<None>"], "shapes": [[],[],[],[],[],[]], "types": ["GenericList[]","Int","None","Device","Bool","None"], "strides": [[],[],[],[],[],[]]},
      "outputs": {"values": [[6,7,0,1,8,"cpu"]], "shapes": [[]], "types": ["Tensor(long int)"], "strides": [[]]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 3},{"name": "fw_parent", "type": "uint64", "value": 0},{"name": "seq_id", "type": "int64", "value": -1},{"name": "scope", "type": "uint64", "va
lue": 0},{"name": "tid", "type": "uint64", "value": 1},{"name": "fw_tid", "type": "uint64", "value": 0},{"name": "op_schema", "type": "string", "value": "aten::empty.memory_format(SymInt[] size, *, Scalar
Type? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"},{"name": "kernel_backend", "type": "string", "value": ""},{"name": "kernel_
file", "type": "string", "value": ""}]
    },
```

You can find the Kineto trace at `device_[rank].json`.

```
       "sharedMemPerBlock": 49152, "numSms": 132
     , "regsPerMultiprocessor": 65536, "sharedMemPerBlockOptin": 232448, "sharedMemPerMultiprocessor": 233472
     },
     {
       "id": 6, "name": "NVIDIA H200", "totalGlobalMem": 150149398528,
       "computeMajor": 9, "computeMinor": 0,
       "maxThreadsPerBlock": 1024, "maxThreadsPerMultiprocessor": 2048,
       "regsPerBlock": 65536, "warpSize": 32,
       "sharedMemPerBlock": 49152, "numSms": 132
     , "regsPerMultiprocessor": 65536, "sharedMemPerBlockOptin": 232448, "sharedMemPerMultiprocessor": 233472
     },
     {
       "id": 7, "name": "NVIDIA H200", "totalGlobalMem": 150149398528,
       "computeMajor": 9, "computeMinor": 0,
       "maxThreadsPerBlock": 1024, "maxThreadsPerMultiprocessor": 2048,
       "regsPerBlock": 65536, "warpSize": 32,
       "sharedMemPerBlock": 49152, "numSms": 132
     , "regsPerMultiprocessor": 65536, "sharedMemPerBlockOptin": 232448, "sharedMemPerMultiprocessor": 233472
     }
...
   "cupti_version": 22,
   "cuda_runtime_version": 12040,
   "cuda_driver_version": 12050,
   "distributedInfo": {"backend": "nccl", "rank": 0, "world_size": 8, "pg_count": 1, "pg_config": [{"pg_name": "0", "pg_desc": "default_pg", "backend_config": "cuda:nccl", "pg_size": 8, "ranks": [0, 1, 2
  3, 4, 5, 6, 7]}], "nccl_version": "2.21.5"},
   "record_shapes": 1,
   "trace_id": "63F34A2FD8B34FE7871631449F897CA6",
   "traceEvents": [
   {
     "ph": "X", "cat": "cpu_op", "name": "autograd::engine::evaluate_function: NllLossBackward0", "pid": 2784631, "tid": 2785980,
     "ts": 871899807921.667, "dur": 49217.469,
     "args": {
       "External id": 2049,"Record function id": 1951, "Sequence number": 177, "Fwd thread id": 1, "Ev Idx": 0
     }
   },
   {
     "ph": "X", "cat": "cpu_op", "name": "NllLossBackward0", "pid": 2784631, "tid": 2785980,
     "ts": 871899807961.613, "dur": 44592.516,
     "args": {
       "External id": 2050,"Sequence number": 177, "Fwd thread id": 1, "Record function id": 1952, "Concrete Inputs": [""], "Input type": ["float"], "Input Strides": [[]], "Input Dims": [[]], "Ev Idx": 1
     }
   },
```
You can also examine the trace using the `chrome://tracing` tool. Simply open a Chrome browser, type the URL into the address bar, and open the kineto trace file.

<img width="2067" height="397" alt="resnet-50" src="https://github.com/user-attachments/assets/510642cf-5ed8-4859-92b6-bf11445e18bf" />


Once you have the PyTorch execution trace and the Kineto trace, the remaining steps to obtain a Chakra execution trace are the same as previously described. Simply follow the steps outlined in the sections for "Merging Traces with trace_link.py" and "Converting to Chakra Execution Trace." By completing these steps, you will successfully convert your collected PyTorch execution traces into the standardized Chakra execution trace format, ready for further analysis or simulation.

## 5. Closing Remarks
ASTRA-sim and PARAM provide example commands and traces for simulating pre-collected PyTorch execution traces on ASTRA-sim. For more information, refer to this page: [Running Simulation with Chakra](https://github.com/mlcommons/chakra/wiki/Running-Simulation-with-Chakra). Additionally, check out the code example at [here](https://github.com/mlcommons/chakra-old/files/13070754/examples.tgz) for a practical demonstration of collecting PyTorch execution traces and Kineto traces, as illustrated by comparing two files (`vimdiff dlrm_main_vanilla.py dlrm_main_saeed.py`).