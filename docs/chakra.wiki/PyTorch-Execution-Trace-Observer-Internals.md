**Authors**: Taekyung Heo (NVIDIA)

## Introduction

The PyTorch Execution Trace Observer in PyTorch is designed to capture the execution trace of machine learning model execution. It not only tracks operators but also gathers their metadata, providing a comprehensive view of the execution process. This document delves into the detailed implementation of this tool, explaining its callback mechanism, integration with PyTorch's existing profiling functions, and the overall operational flow. 

## Callback Mechanism
The ET Observer operates using two primary callbacks: [onFunctionEnter](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/profiler/standalone/execution_trace_observer.cpp#L511) and [onFunctionExit](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/profiler/standalone/execution_trace_observer.cpp#L551). These are pivotal in monitoring the execution of operations within PyTorch. 
* **onFunctionEnter**: This function captures the inputs of an operation, detailing aspects such as the operator's name, the number of inputs, and the input values. 
* **onFunctionExit**: This function handles the outputs from an operation, collecting data like output values, the count of outputs, and details of the operator schema. 
The ET Observer leverages PyTorch's existing profiling features. A key component of this integration is the use of RecordFunctionCallback from ATen's record function, located in aten/src/ATen/record_function.cpp and .h. 

These two callback functions are registered in [addExecutionTraceObserver](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/profiler/standalone/execution_trace_observer.cpp#L641).
```cpp
    ob.cb_handle = addGlobalCallback( 
        RecordFunctionCallback(&onFunctionEnter, &onFunctionExit) 
            .needsInputs(true) 
            .needsOutputs(true) 
            .needsIds(true)); 
    // Default to disabled. 
```

Internally, addGlobalCallback registers callback functions in [the global_callbacks_ variable](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/aten/src/ATen/record_function.cpp#L123). These callbacks are later executed by the LocalCallbackManager.

## PyTorch Interpreter Interaction
During ML model execution, registered callback functions are called in the PyTorch interpreter. PyTorch codes go through multiple stages and are finally lowered to C++ kernel implementations. For a detailed understanding, Edward Yang’s blog post is recommended: [http://blog.ezyang.com/2019/05/pytorch-internals/.](http://blog.ezyang.com/2019/05/pytorch-internals/) The record function is called at `checkAndStartRecordFunction` in `torch/csrc/jit/runtime/interpreter.cpp`. `checkAndStartRecordFunction` looks up active callbacks and finally calls the registered callback functions, as shown below:

* [run](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/jit/runtime/interpreter.cpp#L1056)/[runAsync](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/jit/runtime/interpreter.cpp#L1047) 
  * [runImpl](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/jit/runtime/interpreter.cpp#L255)
    * [callFunction](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/jit/runtime/interpreter.cpp#L187C9-L187C9)
      * [checkAndStartRecordFunction](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/jit/runtime/interpreter.cpp#L895C38-L895C38) 
        * [getStepCallbacksUnlessEmpty](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/aten/src/ATen/record_function.cpp#L616)  

`Frame` plays a crucial role in profiling. A frame occurs after you call a function, with all information available in a stack. The frame implementation can be found at [`torch/csrc/jit/runtime/interpreter/frame.h`](https://github.com/pytorch/pytorch/blob/8257b867d890afd69060408cc145df6bef7da906/torch/csrc/jit/runtime/interpreter/frame.h#L17). For more details, this video clip must be helpful: https://youtu.be/egZB5Uxki0I?si=0J7SlUjGtUHkZMXE&t=1267 

## Understanding Collected Fields 
* **id**: A unique operator ID assigned by the ExecutionTraceObserver. The ET observer maintains a unique ExecutionTraceObserver to manage the state and assigns unique IDs to operators when observing them. uninitialized_id = 0, root_id = 1, all other valid ID > 1.
```cpp
static void recordOperatorStart( 
    ExecutionTraceObserver& ob,  
    FunctionCallContext& fc,  
    const RecordFunction& fn) { 
  auto tid = fn.threadId(); 
 
  try { 
    const std::lock_guard<std::mutex> lock(ob.g_mutex); 

    // if current thread stack is empty, push the root node to the stack first 
    if (ob.op_stack[tid].empty()) { 
      auto thread_node_id = ob.getNewID(); 
      ob.op_stack[tid].push(thread_node_id);
```

* **rf_id**: A unique record function ID, assigned when you instantiate RecordFunction. This can be used to correlate with trace record function ID. Invalid rf_id has value 0.
```cpp
RecordFunction::RecordFunction(StepCallbacks&& step_callbacks) 
    : step_callbacks_{std::move(step_callbacks)} {                                                                                                             
  ctx_.resize(step_callbacks_.callbacks_.size());                                                                                                              
  if (step_callbacks_.needs_ids_) {                                                                                                                            
    setHandle(next_unique_record_function_handle());                                                                                                           
  }  
}
```

* **parent**: The parent ID of an operator. The ExecutionTraceObserver maintains an op_stack, a per-thread stack that tracks operator IDs. The parent ID is retrieved by looking at the top of the stack when adding a new operator.
```cpp
struct TORCH_API RecordFunction { 
...  
 // Retrieves the thread_id that this RecordFunction ran start callbacks with.                                                                                
  // Useful for writing thread safe end callbacks that may be potentially                                                                                      
  // executed in a different thread (async ops)                                                                                                                
  uint64_t threadId() const {                                                                                                                                  
    return step_callbacks_.thread_id_;                                                                                                                         
  }
```

* **fw_parent**: Parent node ID from forward thread. This is due to PyTorch autograd runs on a separate thread from the forward thread. This allows us to reference the corresponding parent node on the forward thread. Valid values are non-zero.

* **seq_id**: Record function sequence ID used to correlate forward and backward operators. Invalid sequence ID has value -1.

* **scope**: The record scope, defined in aten/src/ATen/record_function.h.
```cpp
// Kind of record function scope;                                                                                                                              
enum class C10_API_ENUM RecordScope : uint8_t {                                                                                                                
  // c10/ATen ops, autograd nodes                                                                                                                              
  FUNCTION = 0,                                                                                                                                                
  // Functions/nodes called from the autograd                                                                                                                  
  BACKWARD_FUNCTION,                                                                                                                                           
  // TorchScript functions, methods                                                                                                                            
  TORCHSCRIPT_FUNCTION,                                                                                                                                        
  // Kernel Function dtype Tag                                                                                                                                 
  KERNEL_FUNCTION_DTYPE,                                                                                                                                       
  // Torchbind custom class,                                                                                                                                   
  CUSTOM_CLASS,                                                                                                                                                
  // Generic Build Feature                                                                                                                                     
  BUILD_FEATURE,                                                                                                                                               
  // Kernel Function dtype Tag                                                                                                                                 
  LITE_INTERPRETER,                                                                                                                                            
  // User defined scope (e.g. with record_function())                                                                                                          
  USER_SCOPE,                                                                                                                                                  
  // Scopes for static runtime, a specialized TorchScript interpreter                                                                                          
  STATIC_RUNTIME_OP,                                                                                                                                           
  STATIC_RUNTIME_MODEL,                                                                                                                                        
  NUM_SCOPES, // must be the last in the list                                                                                                                  
};          
```

* **tid**: Record function thread ID.

* **fw_tid**: Thread ID of the forward execution thread.
```cpp
struct TORCH_API RecordFunction { 
... 
 // For backward functions - thread id of the corresponding forward function,                                                                                 
  // or zero otherwise; 
  // used alongside with sequence number to correlate backward functions with                                                                                  
  // the forward ones 
  uint64_t forwardThreadId() const {                                                                                                                           
    return fwd_thread_id_;                                                                                                                                     
  }  
```

* **op_schema**: PyTorch operator schema.

* **inputs**: Array of input args. For non-tensor args, they are actual values.

* **input_shapes**: Array of input shapes. None Tensor input args have empty shape [].

* **outputs**: Same as inputs.

* **output_shapes**: Same as input_shapes.

* **output_types**: Same as input_types

## Additional Resources 
For a comprehensive understanding of the ET Observer, refer to the source code changes and documentation in PyTorch. [Louis’ pull request](https://github.com/pytorch/pytorch/commit/18d46ea9fd5071820cdbe7595b99396a9153f716) on GitHub provides practical insights into these implementations.  