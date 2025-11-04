**Authors**: Taekyung Heo (NVIDIA), Joongun Park (Georgia Tech)

## Introduction

The **ETFeeder** module is designed to read Chakra execution trace files and feed the trace nodes to a simulator in an order that respects their dependencies. It handles the loading of nodes from the trace file, resolves dependencies among nodes, and provides nodes that are ready to be issued to the simulator. This module is crucial for simulating the Chakra trace, as it ensures that operations are executed in the correct order, mimicking real execution.

---

## ETFeederNode Class

### Overview

The `ETFeederNode` class represents a single node (operation) in the Chakra execution trace. Each node encapsulates information about an operation, including its unique identifier, name, runtime duration, dependencies, and other attributes relevant to the simulation.

### Key Attributes

- **`node_`**: A shared pointer to the underlying `ChakraProtoMsg::Node`, which contains the raw data from the trace file.
- **`id_`**: The unique identifier of the node.
- **`name_`**: The name of the operation.
- **`runtime_`**: The duration of the operation in microseconds.
- **`is_cpu_op_`**: A boolean indicating whether the operation is a CPU operation.
- **Dependency Management**:
  - **`children_set_`**: A set of child nodes that depend on this node.
  - **`children_vec_`**: A vector of child nodes for quick iteration.
  - **`dep_unresolved_parent_ids_`**: A list of parent node IDs whose dependencies have not yet been resolved.
- **Attributes**:
  - **`other_attrs_`**: A map of additional attributes not explicitly handled, allowing for extensibility.

### Key Methods

- **Constructor**:
  - `ETFeederNode(std::shared_ptr<ChakraProtoMsg::Node> node)`: Initializes the node by extracting relevant information from the provided protobuf message.
  
- **Accessors**:
  - `std::shared_ptr<ChakraProtoMsg::Node> getChakraNode()`: Returns the underlying protobuf node.
  - `uint64_t id()`: Returns the node's unique identifier.
  - `std::string name()`: Returns the node's name.
  - `bool is_cpu_op()`: Indicates whether the node represents a CPU operation.
  - `uint64_t runtime()`: Returns the runtime duration of the node.
  - Additional getters for attributes like `comm_type()`, `comm_size()`, etc.

- **Dependency Management**:
  - `void addChild(std::shared_ptr<ETFeederNode> node)`: Adds a child node that depends on this node.
  - `std::vector<std::shared_ptr<ETFeederNode>> getChildren()`: Retrieves the list of child nodes.
  - `void addDepUnresolvedParentID(uint64_t node_id)`: Records an unresolved parent dependency.
  - `std::vector<uint64_t> getDepUnresolvedParentIDs()`: Gets the list of unresolved parent IDs.
  - `void setDepUnresolvedParentIDs(const std::vector<uint64_t>& dep_unresolved_parent_ids)`: Updates the list of unresolved parent IDs.

- **Attribute Access**:
  - `const ChakraProtoMsg::AttributeProto& get_other_attr(const std::string& attr_name) const`: Retrieves other attributes by name.
  - `bool has_other_attr(const std::string& attr_name) const`: Checks if a particular attribute exists.

### Important Considerations

- **Dependency Resolution**: The node keeps track of unresolved dependencies, which are later resolved when the missing parent nodes are loaded.
- **Avoiding Duplicates**: When adding child nodes, the class ensures that duplicates are not introduced.

---

## ETFeeder Class

### Overview

The `ETFeeder` class is responsible for reading the Chakra execution trace file, managing the nodes, resolving dependencies, and providing nodes that are ready to be issued to the simulator. It ensures that nodes are issued in an order that respects their data dependencies.

### Key Attributes

- **Trace Handling**:
  - `ProtoInputStream trace_`: The input stream for reading the trace file.
  - `const uint32_t window_size_`: The number of nodes to read into memory at a time.
  - `bool et_complete_`: Indicates whether the entire trace file has been read.

- **Node Management**:
  - `std::unordered_map<uint64_t, std::shared_ptr<ETFeederNode>> dep_graph_`: A mapping from node IDs to nodes, representing the dependency graph.
  - `std::unordered_set<uint64_t> dep_free_node_id_set_`: A set of node IDs that have no dependencies and are ready to be issued.
  - `std::priority_queue<std::shared_ptr<ETFeederNode>, ..., CompareNodes> dep_free_node_queue_`: A priority queue of dependency-free nodes, ordered by node ID.
  - `std::unordered_set<std::shared_ptr<ETFeederNode>> dep_unresolved_node_set_`: A set of nodes with unresolved dependencies.

### Key Methods

- **Constructor and Destructor**:
  - `ETFeeder(std::string filename)`: Initializes the feeder by opening the trace file, reading global metadata, and loading the initial window of nodes.
  - `~ETFeeder()`: Cleans up resources.

- **Node Management**:
  - `void addNode(std::shared_ptr<ETFeederNode> node)`: Adds a node to the dependency graph.
  - `void removeNode(uint64_t node_id)`: Removes a node from the dependency graph and reads more nodes if necessary.
  - `std::shared_ptr<ETFeederNode> lookupNode(uint64_t node_id)`: Retrieves a node by its ID from the dependency graph.

- **Issuable Nodes Handling**:
  - `bool hasNodesToIssue()`: Checks if there are any nodes ready to be issued to the simulator.
  - `std::shared_ptr<ETFeederNode> getNextIssuableNode()`: Retrieves the next node that is ready to be issued.
  - `void pushBackIssuableNode(uint64_t node_id)`: Puts a node back into the queue of issuable nodes.

- **Dependency Resolution**:
  - `void freeChildrenNodes(uint64_t node_id)`: Updates the dependencies of child nodes when a node is completed, potentially making them ready to be issued.
  - `void resolveDep()`: Attempts to resolve dependencies for nodes with unresolved parents.

- **Trace Reading**:
  - `void readGlobalMetadata()`: Reads the global metadata from the trace file.
  - `std::shared_ptr<ETFeederNode> readNode()`: Reads a single node from the trace file.
  - `void readNextWindow()`: Reads the next batch of nodes from the trace file into memory.

### Important Considerations

- **Windowed Reading**: The feeder reads nodes in windows (batches) to manage memory usage efficiently, controlled by `window_size_`.
- **Dependency Resolution**: The feeder continuously tries to resolve dependencies as new nodes are read, ensuring that nodes become issuable as soon as their dependencies are met.
- **Priority Queue**: Dependency-free nodes are stored in a priority queue, allowing the feeder to issue nodes in an order based on their IDs.

---

## Usage Workflow

1. **Initialization**:
   - Create an instance of `ETFeeder`, providing the path to the Chakra execution trace file.
   - The feeder reads global metadata and loads the initial window of nodes.

2. **Simulation Loop**:
   - Check if there are nodes to issue using `hasNodesToIssue()`.
   - Retrieve the next issuable node using `getNextIssuableNode()`.
   - Process the node in the simulator.
   - After processing, remove the node using `removeNode(node_id)`.
   - Free the child nodes' dependencies using `freeChildrenNodes(node_id)`, which may make new nodes ready to be issued.

3. **Node Retrieval**:
   - Use `lookupNode(node_id)` if you need to access a node's details during simulation.

4. **Handling Dependencies**:
   - The feeder automatically handles dependency resolution as new nodes are read and as nodes are completed during simulation.

---

## Conclusion

The **ETFeeder** module plays a crucial role in feeding Chakra execution trace nodes to a simulator while respecting data dependencies. By managing nodes efficiently and ensuring that operations are issued in the correct order, it facilitates accurate simulation of PyTorch applications. Understanding the `ETFeeder` and `ETFeederNode` classes, along with their methods and attributes, is essential for developers working on simulation tools within the Chakra framework.