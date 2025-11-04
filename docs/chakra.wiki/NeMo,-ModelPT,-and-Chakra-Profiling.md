**Authors**: Joongun Park (Georgia Tech)

## Overview
This page explains how [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) integrates Chakra profiling into its base model class, `ModelPT`.
NeMo (NVIDIA's training framework) is a PyTorch-native toolkit for building, training, and deploying state-of-the-art AI models, `ModelPT` serves as the base PyTorch Lightning module for all NeMo models, providing standardized training, checkpointing, and profiling capabilities.

Chakra profiling is available as an optional feature in ModelPT and enables aligned host-side and device-side execution traces that can be merged for deeper performance analysis.

## What is NVIDIA NeMo
NVIDIA NeMo is a PyTorch-native toolkit for building, training, and deploying state-of-the-art AI models across domains such as speech (ASR/TTS), language (NLP/LLM), and multimodal tasks. It provides modular configurations via Hydra/OmegaConf, and training orchestration on single- and multi-GPU systems through Lightning (PyTorch Lightning). NeMo’s collections include ready-to-use models and recipes, while the core layers implement shared infrastructure such as configuration handling, saving/restoring, and distributed training utilities.

## What `ModelPT` is in NeMo
`ModelPT` (defined in `nemo/core/classes/modelPT.py`) is the base class that NeMo models inherit to gain standardized behavior around configuration, data setup, optimization, saving/restoring, and trainer integration. It subclasses Lightning’s LightningModule and NeMo’s Model, and centralizes lifecycle hooks and utilities that every NeMo model relies on. In short, `ModelPT` is the canonical “model recipe” layer in NeMo that ties together configuration, data, training, profiling, and persistence in a consistent way across tasks.

## How Chakra is integrated into `ModelPT`
Chakra profiling is integrated as a first-class, opt-in feature within `ModelPT` and is wired through three key areas: configuration, setup, and training-time hooks. The configuration lives under cfg.chakra_profile. When present and enabled, the model will initialize a coordinated ET (Execution Trace) and Kineto capture that runs for a precise global-step window and writes per-rank outputs to dedicated directories.

The integration is initialized in `_setup_chakra_profiling()`. This method checks self.cfg.chakra_profile.enabled, validates trace_dir, and prepares two subdirectories inside it: one for ET outputs (`_chakra_trace_dir`) and one for Kineto outputs (`_kineto_trace_dir`). _setup_chakra_profiling() constructs an `ExecutionTraceObserver` and a `torch.profiler.profile` instance with CPU and CUDA activities and a schedule defined by the specified warmup and active step counts. The ET observer is passed to the profiler through the execution_trace_observer parameter so both traces are driven in sync.

## How `ModelPT` setsup Chakra profiling 
The `_setup_chakra_profiling()` method configures Chakra profiling inside NeMo’s `ModelPT` class. It reads the `chakra_profile` section from the model’s configuration and, if enabled, initializes both an `ExecutionTraceObserver` (for host-side traces) and the `Kineto` for device-side traces.

[_setup_chakra_profiling() in ModelPT](https://github.com/NVIDIA/NeMo/blob/6251117f4c12288bd9e12bbbf43b0720153d542f/nemo/core/classes/modelPT.py#L1808)

```
 def _setup_chakra_profiling(self):
        """Enables chakra profiling
        To use, add the following options to the model config:
        ## Chakra profiling options
        chakra_profile:
            enabled: False
            start_step: 2  # Global batch to start profiling
            end_step: 2 # Global batch to end profiling
            warmup_steps: 0  # Global batch to start profiling
            active_steps: 1  # Global batch to start profiling
            trace_dir: None # Path to store the profile output file
        """
        if self.cfg.get('chakra_profile', None) is not None:
            if self.cfg.chakra_profile.get('enabled', False):

                from torch.profiler import ExecutionTraceObserver

                from nemo.utils.env_var_parsing import get_envint

                self._chakra_profile_enabled = True
                self._chakra_profile_start_step = self.cfg.chakra_profile.get('start_step', 0)
                self._chakra_profile_end_step = self.cfg.chakra_profile.get('end_step', 0)
                trace_dir = self.cfg.chakra_profile.get('trace_dir', None)

...
                self._et = ExecutionTraceObserver()
                self._prof = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps),
                    execution_trace_observer=self._et,
                )
```

## How `ModelPT` starts Chakra profiling 

The `on_train_batch_start()` method is a PyTorch Lightning hook that executes at the beginning of each training batch.
In NeMo’s `ModelPT` class, it is extended to trigger the start of Chakra profiling when the configured start step is reached.

[on_train_start() in ModelPT](https://github.com/NVIDIA/NeMo/blob/6251117f4c12288bd9e12bbbf43b0720153d542f/nemo/core/classes/modelPT.py#L1965)

```
    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: int = 0) -> Optional[int]:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-start
        We use it here to enable profiling and dynamic freezing.
        """
        if self.device.type == 'cuda':
            if hasattr(self, '_chakra_profile_enabled'):
                if self._chakra_profile_enabled and not self._chakra_profile_in_progress:
                    if (
                        self.trainer.global_step >= self._chakra_profile_start_step
                        and self.trainer.global_step <= self._chakra_profile_end_step
                    ):
                        logging.info(
                            f"====== Start chakra profiling from global_step {self.trainer.global_step} ======"
                        )
                        self._et.register_callback(str(self._chakra_trace_dir / f'rank-{get_rank()}.json'))
                        self._prof.start()
                        self._chakra_profile_in_progress = True
```


## How `ModelPT` Finalizing Chakra Profiling Steps
The `on_train_batch_end()` method is a PyTorch hook that executes at the end of each training batch.
In NeMo’s `ModelPT` class, it is extended to advance or stop Chakra profiling depending on the current training step.

[on_train_batch_end() in ModelPT](https://github.com/NVIDIA/NeMo/blob/6251117f4c12288bd9e12bbbf43b0720153d542f/nemo/core/classes/modelPT.py#L2041)

```
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: int = 0) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-end
        We use it here to enable nsys profiling.
        """

        if self.device.type == 'cuda':
            if hasattr(self, '_chakra_profile_enabled'):
                # self.trainer.global_step is increaeasd before on_train_batch_end
                if self._chakra_profile_enabled and self._chakra_profile_in_progress:
                    if self.trainer.global_step - 1 >= self._chakra_profile_end_step:
                        logging.info(f"====== End chakra profiling at global_step {self.trainer.global_step} ======")
                        self._prof.stop()
                        self._prof.export_chrome_trace(str(self._kineto_trace_dir / f'rank-{get_rank()}.json'))
                        self._et.unregister_callback()
                        self._chakra_profile_in_progress = False
                    elif self.trainer.global_step - 1 >= self._chakra_profile_start_step:
                        self._prof.step()
```

When the current global step falls between the configured `start_step` and `end_step` (inclusive), Chakra profiling is initialized for that rank.
The Execution Trace observer (self._et) is registered to save host-side trace output to the appropriate JSON file in `_chakra_trace_dir`, and the torch.profiler.profile instance (self._prof) is started to record device-side activity.

## Enabling Chakra Profiling via NeMo Launcher and Hydra Configuration

The [NeMo Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher) is a configuration-driven job orchestration framework for training and fine-tuning NeMo models at scale.
It builds on top of Hydra for hierarchical configuration management and supports launching jobs across a variety of platforms, including single-node, multi-GPU, and multi-node HPC clusters.

In the context of Chakra profiling, NeMo Launcher allows you to pass profiling parameters—such as the chakra_profile section—directly in your YAML config or via command-line overrides. This means you can enable and control profiling behavior without modifying code, making it ideal for reproducible experiments and automated benchmarking workflows.

To activate Chakra profiling in NeMo’s `ModelPT`, you need to provide a chakra_profile section in your Hydra configuration file. This section defines when profiling starts and ends, how many warmup and active steps are used, and where the trace files will be stored.

The parameters are:
```
enabled: Boolean flag to turn Chakra profiling on (true) or off (false).
start_step: The global batch step at which profiling should begin.
end_step: The global batch step at which profiling should stop. Must be greater than or equal to start_step.
warmup_steps: Number of steps to run before starting active profiling.
active_steps: Number of steps to capture during each profiling cycle.
trace_dir: Filesystem path to store trace files. This directory must exist before training starts.
```

```

And here is part of example hydra file.
...
chakra_profile:
  enabled: true              # Enable Chakra profiling
  start_step: 5               # Start profiling at global batch step 5
  end_step: 10                # Stop profiling at global batch step 10
  warmup_steps: 0              # No warmup before profiling
  active_steps: 1              # Capture one step per profiling cycle
  trace_dir: /workspace/traces # Directory where output traces will be saved

nsys_profile:
  enabled: false               # Must be false when using Chakra profiling
...
```

For more details about Hydra see [here](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/core/exp_manager.html?highlight=hydra#hydra-multi-run-with-nemo)

You can now integrate Chakra profiling seamlessly into your NeMo training workflows by configuring it directly in your Hydra YAML.
Once enabled, both the Execution Trace (host-side) and Kineto (device-side) traces will be captured in perfect sync, allowing you to merge them for deep performance analysis.
