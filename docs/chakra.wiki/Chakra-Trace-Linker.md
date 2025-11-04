**Authors**: Joongun Park (Georgia Tech)

## Introduction

This document provides an overview of the **Chakra Trace Linking Tool**, which is responsible for merging Chakra host execution traces with Chakra device traces to produce a unified trace that includes both CPU and GPU operations. This unified trace is essential for analyzing the complete execution flow of PyTorch applications using the Chakra framework, particularly for performance optimization and simulation.

---

## TraceLinker Class

The `TraceLinker` class is the core component of the Chakra Trace Linking Tool. It processes and combines the host and device traces, ensuring that the final output accurately reflects the execution timeline and dependencies across CPU and GPU operations.

### Key Responsibilities

- **Loading Traces**: Uses `ChakraHostTraceLoader` and `ChakraDeviceTraceLoader` to load host and device traces, respectively.
- **Enforcing Inter-Thread Order**: Identifies significant gaps in execution within threads to establish dependencies between operation groups across different threads.
- **Linking Traces**: Maps host operations to corresponding device operations, aligning CPU and GPU activities based on unique identifiers and timestamps.
- **Constructing Enhanced Trace Data**: Combines enriched information into a single data structure representing the enhanced Chakra execution trace (ET+ or Chakra HDT).
- **Dumping Output**: Writes the enhanced trace data to the specified output file in JSON format.

### Key Components and Methods

- **Attributes**:
  - `chakra_host_trace_loader`: Instance of `ChakraHostTraceLoader` to load host traces.
  - `chakra_device_trace_loader`: Instance of `ChakraDeviceTraceLoader` to load device traces.
  - `id_assigner`: Instance of `UniqueIdAssigner` to maintain unique IDs across operations.

- **Methods**:
  - `link(chakra_host_trace, chakra_device_trace, output_file)`: Main method to perform the linking process.
  - `enforce_inter_thread_order(kineto_tid_cpu_ops_map, threshold)`: Ensures correct execution order across threads by establishing dependencies based on execution gaps.
  - `link_traces(...)`: Coordinates the mapping and merging of host and device operations.
  - `map_host_to_device_ops(...)`: Maps host operations to their corresponding device operations.
  - `group_gpu_ops_by_cpu_launchers(...)`: Groups GPU operations based on their CPU launch events for accurate alignment.
  - `construct_et_plus_data(...)`: Constructs the enhanced trace data structure (ET+).
  - `dump_chakra_execution_trace_plus(...)`: Writes the enhanced trace data to the output file.

### Important Considerations

- **Inter-Thread Dependencies**: The class identifies significant execution gaps within threads to establish dependencies between different threads, ensuring realistic inter-thread execution order.
- **Operator Mapping**: Host operators are linked to device operators using unique identifiers (e.g., `rf_id`, `ev_idx`) and timestamps, which is crucial for accurate trace alignment.
- **Data Structures**: Utilizes dictionaries and lists to efficiently map and store relationships between operations, facilitating quick lookups and updates.

---

## Usage Example

To use the Chakra Trace Linking Tool, you can execute the following command:

```bash
$ chakra_trace_link \
    --chakra-host-trace /path/to/chakra_host_trace.json \
    --chakra-device-trace /path/to/chakra_device_trace.json \
    --output-file /path/to/chakra_host_device_trace.json \
    --log-level INFO
```

Replace `/path/to/chakra_host_trace.json`, `/path/to/chakra_device_trace.json`, and `/path/to/chakra_host_device_trace.json` with the actual file paths.

---

## Conclusion

The Chakra Trace Linking Tool's `TraceLinker` class plays a vital role in merging CPU and GPU execution traces into a unified format. By understanding its key responsibilities and components, users can effectively generate enhanced traces suitable for detailed analysis, performance optimization, and simulation in distributed and parallel computing environments.

---

Feel free to explore the tool further or integrate it into your workflow to gain comprehensive insights into your PyTorch application's performance with the Chakra framework.