**Authors**: Joongun Park (Georgia Tech)

# Chakra Framework Overview

## Introduction

The **Chakra Framework** is a comprehensive suite of tools designed to facilitate the performance analysis and optimization of PyTorch applications, particularly in distributed and parallel computing environments. By providing mechanisms to merge, convert, and simulate execution traces, Chakra enables developers and researchers to gain deep insights into the execution flow of their applications, identify bottlenecks, and enhance performance.

<p align="center">
  <img width="505" a lt="Screenshot 2024-01-03 at 4 10 32 PM" src="https://github.com/mlcommons/chakra/assets/7621438/67228699-cec5-4a4d-b03e-e76647a80ce8">
</p>

## Key Components

The Chakra Framework consists of several interconnected components that work together to provide a workflow from trace collection to simulation:

### 1. TraceLinker: Merging Host and Device Execution Traces

- **Purpose**: Combines Chakra host execution traces (CPU operations) with Chakra device traces (GPU operations) into a unified execution trace.
- **Functionality**:
  - Loads host and device traces using specialized loaders.
  - Enforces inter-thread execution order to maintain realistic dependencies.
  - Maps host operations to corresponding device operations based on unique identifiers and timestamps.
  - Constructs an enhanced trace data structure (ET+) that includes enriched information.
- **Documentation**: [TraceLinker: Merging Host and Device Execution Traces](Chakra-Trace-Linker)

### 2. Converter: Converting Chakra Traces to Protobuf Format

- **Purpose**: Transforms the unified Chakra execution trace from JSON format into a protobuf format suitable for simulation and further analysis.
- **Components**:
  - **PyTorchNode**:
    - Represents nodes (operations) in the Chakra execution trace.
    - Captures detailed information about each operation, including dependencies and timing.
    - **Documentation**: [PyTorchNode Class: Representing Nodes in Chakra Execution Traces](Chakra-Trace-Node-Explanation)
  - **PyTorchConverter**:
    - Handles the conversion process, including node conversion, dependency resolution, and cyclic dependency detection.
    - Ensures the converted trace is compatible with simulation tools.
    - **Documentation**: [PyTorchConverter: Converting Chakra Traces to Protobuf Format](Chakra-Converter)

### 3. ETFeeder: Feeding Chakra Traces to Simulators

- **Purpose**: Reads the converted Chakra trace and feeds the nodes to a simulator in an order that respects their data dependencies.
- **Functionality**:
  - Manages the loading and issuing of nodes from the trace file.
  - Resolves dependencies among nodes dynamically.
  - Provides nodes that are ready for simulation while handling large trace files efficiently.
- **Documentation**: [ETFeeder: Feeding Chakra Traces to Simulators](Chakra-ETFeeder)

---

## Workflow Overview

The Chakra Framework provides a structured workflow to analyze and optimize PyTorch applications:

1. **Trace Collection**: Collect execution traces from your PyTorch application, generating separate host and device traces.
2. **Trace Linking**: Use the **TraceLinker** to merge the host and device traces into a unified trace that includes both CPU and GPU operations.
3. **Trace Conversion**: Utilize the **PyTorchConverter** to convert the unified trace into protobuf format, preparing it for simulation.
4. **Feeding to Simulator**: Employ the **ETFeeder** to feed the converted trace to a simulator, ensuring operations are executed in the correct order.
5. **Analysis and Optimization**: Analyze the simulation results to identify performance bottlenecks and optimize your application accordingly.

---

## Additional Resources

- **TraceLinker Documentation**: [TraceLinker: Merging Host and Device Execution Traces](Chakra-Trace-Linker)
- **PyTorchNode Documentation**: [PyTorchNode Class: Representing Nodes in Chakra Execution Traces](Chakra-Trace-Node-Explanation)
- **PyTorchConverter Documentation**: [PyTorchConverter: Converting Chakra Traces to Protobuf Format](Chakra-Converter)
- **ETFeeder Documentation**: [ETFeeder: Feeding Chakra Traces to Simulators](Chakra-ETFeeder)
- **Chakra GitHub Repository**: [mlcommons/chakra](https://github.com/mlcommons/chakra)

---

## Conclusion

The Chakra Framework offers a powerful set of tools for performance analysis and optimization of PyTorch applications. By providing a complete workflow from trace collection to simulation, it enables developers and researchers to deeply understand their applications' execution behavior. 