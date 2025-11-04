The Chakra schema is structured using Protocol Buffers (protobuf), a tool for serializing structured data. This schema is specifically designed to detail the structure of execution traces within the Chakra framework. Below is an outline of the primary components of the schema.  For more in-depth details, please visit the [Chakra GitHub repository](https://github.com/mlcommons/chakra/blob/main/schema/protobuf/et_def.proto).

### Node
This entity represents an individual node within the execution graph, encapsulating various critical details such as its ID, name, type, control and data dependencies, timing metrics, and input/output specifics, along with other attributes.

| Field Name        | Type              |
|-------------------|-------------------|
| id                | uint64            |
| name              | string            |
| type              | NodeType          |
| ctrl_deps         | repeated uint64   |
| data_deps         | repeated uint64   |
| start_time_micros | uint64            | 
| duration_micros   | uint64            |
| inputs            | IOInfo            |
| outputs           | IOInfo            | 
| attr              | repeated AttributeProto |

### NodeType (Enum Table)
This table enumerates possible types of nodes within the execution trace, including metadata, memory operations, computational tasks, and various forms of communication nodes.


| Name               | Value |
|--------------------|-------|
| INVALID_NODE       | 0     |
| METADATA_NODE      | 1     |
| MEM_LOAD_NODE      | 2     |
| MEM_STORE_NODE     | 3     |
| COMP_NODE          | 4     |
| COMM_SEND_NODE     | 5     |
| COMM_RECV_NODE     | 6     |
| COMM_COLL_NODE     | 7     |

### AttributeProto
 Defines an attribute with a name, documentation string, and one of several possible value types (double, float, int32, int64, etc.), including single values and lists of values.

| Field Name    | Type          |
|---------------|---------------|
| name          | string        |
| doc_string    | string        |
| value         | oneof         |

### CollectiveCommType (Enum Table)
An enumeration of different types of collective communication operations, such as all-reduce, broadcast, and barrier, relevant for parallel computing scenarios.

| Collective Communication Type | Value |
|-------------------------------|-------|
| ALL_REDUCE                    | 0     |
| REDUCE                        | 1     |
| ALL_GATHER                    | 2     |
| GATHER                        | 3     |
| SCATTER                       | 4     |
| BROADCAST                     | 5     |
| ALL_TO_ALL                    | 6     |
| REDUCE_SCATTER                | 7     |
| REDUCE_SCATTER_BLOCK          | 8     |
| BARRIER                       | 9     |

### Tensor 
Defines the structure of a tensor, including its identification, storage specifics, size, and computational location, providing essential insights into data management within the framework.

| Field         | Type   |
|---------------|--------|
| tensor_id     | uint64 |
| storage_id    | uint64 |
| offset        | uint64 | 
| num_elem      | uint64 |
| elem_bytes    | uint64 |
| device        | string |

### IOInfo
Outlines the input/output characteristics for nodes, detailing the data's values, shapes, and types, which aids in understanding node processing activities.

| Field   | Type   |
|---------|--------|
| values  | string |
| shapes  | string |
| types   | string |
