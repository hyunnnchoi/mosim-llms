## Introduction

We are excited to announce the release of Chakra schema v0.0.4, dated October 10th, 2023. This release, available in commit version [c236cfa](https://github.com/mlcommons/chakra/commit/c236cfaeeff459cd508fb266b6e0e2b63dd44e87), introduces substantial updates and improvements, reflecting our commitment to staying at the forefront of technology and user needs.

## Key Changes in v0.0.4

### 1. Transition to proto3

In an effort to align with the latest technological standards and community practices, Chakra schema has been upgraded from protobuf version 2 (proto2) to version 3 (proto3). This transition is analogous to the widely recognized shift from Python 2 to Python 3, bringing a host of new features and improvements.

### 2. AttributeProto Message Type Enhancements

#### 2.1. Simplification of Field Declarations

Aligning with proto3’s streamlined approach, we have removed the "required" and "optional" field keywords to simplify the schema and enhance compatibility.

#### 2.2. Expanded Data Type Support

The `AttributeProto` message type now comprehensively covers all proto3 data types. The introduction of the `oneof` keyword allows for a more flexible and precise definition of data types, catering to a broader range of use cases.

### 3. Refined Dependency Encoding

The schema now differentiates more clearly between control dependencies (`ctrl_deps`) and data dependencies (`data_deps`). This distinction provides a clearer understanding of the execution flow and data relationships between different schema components.

### 4. Comprehensive Collective Communication Types

The `CollectiveCommType` has been extended to include a complete range of collective communication types, ensuring greater versatility in distributed computing scenarios.

### 5. Introduction of the Message Tensor Type

A new `Message Tensor` type has been introduced to provide a detailed and structured way to encode tensor information. This enhancement is particularly beneficial for complex data handling and manipulation within the schema.

### 6. IOInfo for Input/Output Specifications

The introduction of the `IOInfo` message type offers a more refined and detailed specification for input and output data, promoting greater clarity and efficiency in data handling.

### 7. Precise Timing Information

To facilitate detailed analysis and optimization of system performance, fields like `start_time_micros` and `duration_micros` have been introduced. These fields allow for precise measurement and analysis of operation timings within the schema.

## Looking Ahead: Future Revisions

### 1. Comprehensive Documentation

A detailed written specification will be developed to provide clear guidance on the utilization of new fields, ensuring users can fully leverage the capabilities of the updated schema.

### 2. Enhanced Concurrency Information

Future revisions will focus on introducing detailed concurrency information for both CPU and GPU, addressing the growing needs for parallel computing efficiency.

### 3. Revisiting Communicator Group Information

The handling of communicator group information, particularly in relation to the inputs field, is slated for a comprehensive review and potential restructuring.

## Conclusion

With these updates, Chakra schema v0.0.4 takes a significant step forward in providing a more powerful, flexible, and user-friendly tool for our community. We are committed to continuous improvement and eagerly anticipate your feedback and contributions.
