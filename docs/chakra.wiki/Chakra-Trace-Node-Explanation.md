**Authors**: Joongun Park (Georgia Tech)

## Introduction

This document provides an overview of the `PyTorchNode` and `PyTorchNodeType` classes, which are designed to represent nodes in a Chakra execution trace collected from PyTorch applications. These classes facilitate the analysis and understanding of the execution flow and performance characteristics of PyTorch models, especially when utilizing the Chakra framework for distributed and parallel computations.

--- 

## PyTorchNodeType Enumeration

The `PyTorchNodeType` enumeration categorizes nodes within the execution trace based on their operation types:

- **CPU_OP**: Represents a CPU operation.
- **GPU_OP**: Represents a GPU operation.
- **LABEL**: Represents non-operator nodes, such as labels or markers within the trace.
- **METADATA**: Represents metadata nodes, such as process group initializations or other non-computational events.

## PyTorchNode Class

### Overview

The `PyTorchNode` class encapsulates information about a single node in the execution trace. Each node corresponds to an operation or event during the execution of a PyTorch model. The class captures details about the node's identity, operation type, inputs and outputs, timing information, and relationships with other nodes in the trace.

### Key Attributes

- **id**: A unique identifier for the node.
- **name**: The name of the operation or event the node represents.
- **schema**: The schema version used for initializing and parsing the node data.
- **inputs**: A dictionary containing the inputs to the node, including values and types.
- **outputs**: A dictionary containing the outputs from the node.
- **inclusive_dur**: The inclusive duration of the node's operation, including the time spent in child operations.
- **exclusive_dur**: The exclusive duration of the node's operation, excluding child operations.
- **ts**: The timestamp when the node's operation started.
- **cat**: The category of the node, often used to identify the type of operation or event.
- **stream**: The compute stream associated with the node, relevant for GPU operations.
- **pg_name**: The process group name used for inter-GPU communication, encoded as "" if not existed.
- **data_deps**: A list of parent nodes that the current node depends on for data.
- **children**: A list of child nodes that are called or executed within the context of the current node.
- **gpu_children**: A list of child nodes specifically representing GPU operations.
- **record_param_comms_node**: An **optional ** reference to a corresponding `record_param_comms` node, if present.
- **nccl_node**: An **optional ** reference to a corresponding NCCL (NVIDIA Collective Communications Library) node, if present.

### Usage

To use the `PyTorchNode` class, you first need to create an instance by providing the schema version and the node data extracted from the execution trace:

```python
from chakra.src.converter.pytorch_node import PyTorchNode

schema_version = "1.0.3-chakra.0.0.4"
node_data = {
        "id": 2,
        "name": "node2",
        "ctrl_deps": 1,
        "inputs": {"values": [], "shapes": [], "types": []},
        "outputs": {"values": [], "shapes": [], "types": []},
        "attrs": [
            {"name": "rf_id", "type": "uint64", "value": 2},
            {"name": "fw_parent", "type": "uint64", "value": 0},
            {"name": "seq_id", "type": "int64", "value": -1},
            {"name": "scope", "type": "uint64", "value": 7},
            {"name": "tid", "type": "uint64", "value": 1},
            {"name": "fw_tid", "type": "uint64", "value": 0},
            {"name": "op_schema", "type": "string", "value": ""},
        ],
        # Additional node data...
        "exclusive_dur": 30,
    }

node = PyTorchNode(schema_version , node_data)
```

### Supported Schema Versions

The `PyTorchNode` class supports the following schema versions for parsing node data:

- `"1.0.2-chakra.0.0.4"`
- `"1.0.3-chakra.0.0.4"`
- `"1.1.0-chakra.0.0.4"`

Using an unsupported schema version will result in a `ValueError` during initialization.

## Conclusion

The `PyTorchNode` and `PyTorchNodeType` classes are for representing and analyzing nodes within a Chakra execution trace. They provide a structured way to access node information, determine operation types, and understand the relationships between different operations in a PyTorch model's execution. .
