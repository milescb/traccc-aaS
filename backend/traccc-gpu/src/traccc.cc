// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cuda_runtime_api.h>

#include "TracccGpuStandalone.hpp"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace Traccc {

//
// Backend that demonstrates the TRITONBACKEND API. This backend works
// for any model that has 1 input with any datatype and any shape and
// 1 output with the same shape and datatype as the input. The backend
// supports both batching and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

/////////////

extern "C" {

// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // This backend does not require any "global" state but as an
  // example create a string to demonstrate.
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
    // Delete the "global" state associated with the backend.
    void* vstate;
    RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
    std::string* state = reinterpret_cast<std::string*>(vstate);

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
            .c_str());

    delete state;

    return nullptr;  // success
}

}  // extern "C"

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//! Possible differences in this class!
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

    // Name of the input and output tensor
    const std::string &InputGeoIdTensorName() const { return input_geoid_name_; }
    const std::string &InputCellsTensorName() const { return input_clusters_name_; }
    const std::string &OutputTensorName() const { return output_name_; }

    // Datatype of the input and output tensor
    TRITONSERVER_DataType InputGeoIdTensorDataType() const { return input_geoid_datatype_; }
    TRITONSERVER_DataType InputCellsTensorDataType() const { return input_clusters_datatype_; }
    TRITONSERVER_DataType OutputTensorDataType() const
    {
        return output_datatype_;
    }

    // Shape of the input and output tensor as given in the model
    // configuration file. This shape will not include the batch
    // dimension (if the model has one).
    // const std::vector<int64_t>& TensorNonBatchShape() const { return nb_shape_;
    // }

    // Shape of the input and output tensor, including the batch
    // dimension (if the model has one). This method cannot be called
    // until the model is completely loaded and initialized, including
    // all instances of the model. In practice, this means that backend
    // should only call it in TRITONBACKEND_ModelInstanceExecute.
    const std::vector<int64_t> &InputGeoIdTensorNonBatchShape() const
    {
        return input_geoid_nb_shape_;
    }
    const std::vector<int64_t> &InputCellsTensorNonBatchShape() const
    {
        return input_clusters_nb_shape_;
    }
    const std::vector<int64_t> &OutputTensorNonBatchShape() const
    {
        return output_nb_shape_;
    }

    // Validate that this model is supported by this backend.
    TRITONSERVER_Error *ValidateModelConfig();

    //   std::string model_path;
    int64_t cellFeatures;

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

    std::string input_geoid_name_;
    std::string input_clusters_name_;
    std::string output_name_;

    TRITONSERVER_DataType input_geoid_datatype_;
    TRITONSERVER_DataType input_clusters_datatype_;
    TRITONSERVER_DataType output_datatype_;

    std::vector<int64_t> input_geoid_nb_shape_;
    std::vector<int64_t> input_clusters_nb_shape_;
    std::vector<int64_t> input_geoid_shape_;
    std::vector<int64_t> input_clusters_shape_;
    std::vector<int64_t> output_nb_shape_;
    std::vector<int64_t> output_shape_;

    bool shape_initialized_;
};

//! Also possible problems in this function!
ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), shape_initialized_(false)
{
  // Validate that the model's configuration matches what is supported
  // by this backend.
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
    // If verbose logging is enabled, dump the model's configuration as
    // JSON into the console output.
    if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
    }

    // ModelConfig is the model configuration as a TritonJson
    // object. Use the TritonJson utilities to parse the JSON and
    // determine if the configuration is supported by this backend.
    common::TritonJson::Value inputs, outputs;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

        // The model must have exactly 2 inputs and 1 output.
    RETURN_ERROR_IF_FALSE(
        inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model configuration must have 2 inputs"));
    RETURN_ERROR_IF_FALSE(
        outputs.ArraySize() == 3, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model configuration must have 3 outputs"));

    common::TritonJson::Value input_geoid, input_clusters, output;
    RETURN_IF_ERROR(inputs.IndexAsObject(0, &input_geoid));
    RETURN_IF_ERROR(inputs.IndexAsObject(1, &input_clusters));
    RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

    // Record the input and output name in the model state.
    const char *input_geoid_name, *input_clusters_name;
    size_t input_geoid_len, input_clusters_len;
    RETURN_IF_ERROR(input_geoid.MemberAsString("name", &input_geoid_name, &input_geoid_len));
    RETURN_IF_ERROR(input_clusters.MemberAsString("name", &input_clusters_name, &input_clusters_len));
    input_geoid_name_ = std::string(input_geoid_name);
    input_clusters_name_ = std::string(input_clusters_name);

    const char *output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));
    output_name_ = std::string(output_name);

    // Input and output must have same datatype
    std::string input_geoid_dtype, input_clusters_dtype, output_dtype;
    RETURN_IF_ERROR(input_geoid.MemberAsString("data_type", &input_geoid_dtype));
    RETURN_IF_ERROR(input_clusters.MemberAsString("data_type", &input_clusters_dtype));
    RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
    // RETURN_ERROR_IF_FALSE(
    //     input_dtype == output_dtype, TRITONSERVER_ERROR_INVALID_ARG,
    //     std::string("expected input and output datatype to match, got ") +
    //         input_dtype + " and " + output_dtype);
    input_geoid_datatype_ = ModelConfigDataTypeToTritonServerDataType(input_geoid_dtype);
    input_clusters_datatype_ = ModelConfigDataTypeToTritonServerDataType(input_clusters_dtype);
    output_datatype_ = ModelConfigDataTypeToTritonServerDataType(output_dtype);

    std::vector<int64_t> input_geoid_shape, input_clusters_shape, output_shape;
    RETURN_IF_ERROR(backend::ParseShape(input_geoid, "dims", &input_geoid_shape));
    RETURN_IF_ERROR(backend::ParseShape(input_clusters, "dims", &input_clusters_shape));
    RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

    input_geoid_nb_shape_ = input_geoid_shape;
    input_clusters_nb_shape_ = input_clusters_shape;
    output_nb_shape_ = output_shape;

    return nullptr; // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance 
{
private:

    ModelState* model_state_;

    ModelInstanceState(
        ModelState* model_state,
        TRITONBACKEND_ModelInstance* triton_model_instance)
        : BackendModelInstance(model_state, triton_model_instance),
            model_state_(model_state),
            host_mr_(),
            device_mr_(DeviceId())
    {
    }

public:
    static TRITONSERVER_Error* Create(
        ModelState* model_state,
        TRITONBACKEND_ModelInstance* triton_model_instance,
        ModelInstanceState** state);
    virtual ~ModelInstanceState() = default;

    // Get the state of the model that corresponds to this instance.
    ModelState* StateForModel() const { return model_state_; }

    // define standalone object
    std::unique_ptr<TracccGpuStandalone> traccc_gpu_standalone_;

    // Memory resources
    vecmem::host_memory_resource host_mr_;
    vecmem::cuda::device_memory_resource device_mr_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
    // Get the model state associated with this instance's model.
    TRITONBACKEND_Model* model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    void* vmodelstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
    ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

    // Create a ModelInstanceState object and associate it with the
    // TRITONBACKEND_ModelInstance.
    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(
        ModelInstanceState::Create(model_state, instance, &instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void*>(instance_state)));

    // Set the CUDA device for this thread
    cudaError_t err = cudaSetDevice(instance_state->DeviceId());
    if (err != cudaSuccess)
    {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to set CUDA device: " + std::string(cudaGetErrorString(err))).c_str());
    }
  
    instance_state->traccc_gpu_standalone_ = std::make_unique<TracccGpuStandalone>(
        &instance_state->host_mr_,
        &instance_state->device_mr_,
        instance_state->DeviceId()
    );
    return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    // Collect various timestamps during the execution of this batch or
    // requests. These values are reported below before returning from
    // the function.

    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    // Triton will not call this function simultaneously for the same
    // 'instance'. But since this backend could be used by multiple
    // instances from multiple models the implementation needs to handle
    // multiple calls to this function at the same time (with different
    // 'instance' objects). Best practice for a high-performance
    // implementation is to avoid introducing mutex/lock and instead use
    // only function-local and model-instance-specific state.
    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
        instance, reinterpret_cast<void**>(&instance_state)));
    ModelState* model_state = instance_state->StateForModel();

    // Set the CUDA device for this thread
    // Seems that this is necessary to set the device for each request
    // Without leads to out-of-bounds memory access error
    cudaError_t err = cudaSetDevice(instance_state->DeviceId());
    if (err != cudaSuccess)
    {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to set CUDA device: " + std::string(cudaGetErrorString(err))).c_str());
    }

    // 'responses' is initialized as a parallel array to 'requests',
    // with one TRITONBACKEND_Response object for each
    // TRITONBACKEND_Request object. If something goes wrong while
    // creating these response objects, the backend simply returns an
    // error from TRITONBACKEND_ModelInstanceExecute, indicating to
    // Triton that this backend did not create or send any responses and
    // so it is up to Triton to create and send an appropriate error
    // response for each request. RETURN_IF_ERROR is one of several
    // useful macros for error handling that can be found in
    // backend_common.h.

    std::vector<TRITONBACKEND_Response*> responses;
    responses.reserve(request_count);
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];
        TRITONBACKEND_Response* response;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
        responses.push_back(response);
    }

    // At this point, the backend takes ownership of 'requests', which
    // means that it is responsible for sending a response for every
    // request. From here, even if something goes wrong in processing,
    // the backend must return 'nullptr' from this function to indicate
    // success. Any errors and failures must be communicated via the
    // response objects.
    //
    // To simplify error handling, the backend utilities manage
    // 'responses' in a specific way and it is recommended that backends
    // follow this same pattern. When an error is detected in the
    // processing of a request, an appropriate error response is sent
    // and the corresponding TRITONBACKEND_Response object within
    // 'responses' is set to nullptr to indicate that the
    // request/response has already been handled and no further processing
    // should be performed for that request. Even if all responses fail,
    // the backend still allows execution to flow to the end of the
    // function so that statistics are correctly reported by the calls
    // to TRITONBACKEND_ModelInstanceReportStatistics and
    // TRITONBACKEND_ModelInstanceReportBatchStatistics.
    // RESPOND_AND_SET_NULL_IF_ERROR, and
    // RESPOND_ALL_AND_SET_NULL_IF_ERROR are macros from
    // backend_common.h that assist in this management of response
    // objects.

    // The backend could iterate over the 'requests' and process each
    // one separately. But for performance reasons it is usually
    // preferred to create batched input tensors that are processed
    // simultaneously. This is especially true for devices like GPUs
    // that are capable of exploiting the large amount parallelism
    // exposed by larger data sets.
    //
    // The backend utilities provide a "collector" to facilitate this
    // batching process. The 'collector's ProcessTensor function will
    // combine a tensor's value from each request in the batch into a
    // single contiguous buffer. The buffer can be provided by the
    // backend or 'collector' can create and manage it. In this backend,
    // there is not a specific buffer into which the batch should be
    // created, so use ProcessTensor arguments that cause collector to
    // manage it. ProcessTensor does NOT support TRITONSERVER_TYPE_BYTES
    // data type.

    BackendInputCollector collector(
        requests, request_count, &responses, model_state->TritonMemoryManager(),
        false /* pinned_enabled */, nullptr /* stream*/);

    // To instruct ProcessTensor to "gather" the entire batch of input
    // tensors into a single contiguous buffer in CPU memory, set the
    // "allowed input types" to be the CPU ones (see tritonserver.h in
    // the triton-inference-server/core repo for allowed memory types).
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types =
        {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

    const char *input_geoid_buffer, *input_clusters_buffer;
    size_t input_geoid_buffer_byte_size, input_clusters_buffer_byte_size;
    TRITONSERVER_MemoryType input_geoid_buffer_memory_type, input_clusters_buffer_memory_type;
    int64_t input_geoid_buffer_memory_type_id, input_clusters_buffer_memory_type_id;

    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(
            model_state->InputGeoIdTensorName().c_str(), nullptr /* existing_buffer */,
            0 /* existing_buffer_byte_size */, allowed_input_types, &input_geoid_buffer,
            &input_geoid_buffer_byte_size, &input_geoid_buffer_memory_type,
            &input_geoid_buffer_memory_type_id));

    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(
            model_state->InputCellsTensorName().c_str(), nullptr /* existing_buffer */,
            0 /* existing_buffer_byte_size */, allowed_input_types, &input_clusters_buffer,
            &input_clusters_buffer_byte_size, &input_clusters_buffer_memory_type,
            &input_clusters_buffer_memory_type_id));

    // Finalize the collector. If 'true' is returned, 'input_buffer'
    // will not be valid until the backend synchronizes the CUDA
    // stream or event that was used when creating the collector. For
    // this backend, GPU is not supported and so no CUDA sync should
    // be needed; so if 'true' is returned simply log an error.
    const bool need_cuda_input_sync = collector.Finalize();
    if (need_cuda_input_sync)
    {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            "'Traccc' backend: unexpected CUDA sync required by collector");
    }

    // 'input_buffer' contains the batched input tensor. The backend can
    // implement whatever logic is necessary to produce the output
    // tensor. This backend simply logs the input tensor value and then
    // returns the input tensor value in the output tensor so no actual
    // computation is needed.

    uint64_t compute_start_ns = 0;
    SET_TIMESTAMP(compute_start_ns);

    // Determine the number of objects in the input buffer
    size_t num_uint_geoid = input_geoid_buffer_byte_size / sizeof(std::int64_t);
    size_t num_floats_clusters = input_clusters_buffer_byte_size / sizeof(float);

    // convert to expected types
    const std::int64_t *int_geoid_ptr = reinterpret_cast<const std::int64_t *>(input_geoid_buffer);
    const float *float_ptr = reinterpret_cast<const float *>(input_clusters_buffer);

    // re-format
    size_t num_features_clusters = 46;
    size_t num_rows_clusters = num_floats_clusters / num_features_clusters;
    std::vector<std::vector<float>> input_data_clusters;

    size_t num_geoid_clusters = num_uint_geoid / 2;
    std::vector<std::pair<std::int64_t, std::int64_t>> input_data_geoid;

    input_data_geoid.reserve(num_geoid_clusters);
    input_data_clusters.reserve(num_rows_clusters);

    for (size_t i = 0; i < num_rows_clusters; ++i) {
        std::vector<float> row;
        row.reserve(num_features_clusters);
        for (size_t j = 0; j < num_features_clusters; ++j) {
            row.push_back(static_cast<float>(float_ptr[i * num_features_clusters + j]));
        }
        input_data_clusters.push_back(row);

        input_data_geoid.emplace_back(int_geoid_ptr[i * 2], int_geoid_ptr[i * 2 + 1]);
    }

    int numClusters = input_data_clusters.size();
    std::cout << "Number of clusters received: " << numClusters  << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Cluster " << i << ": ";
        std::cout << input_data_geoid[i].first << " ";
        std::cout << input_data_geoid[i].second << " ";
        for (size_t j = 0; j < num_features_clusters; ++j) {
            std::cout << input_data_clusters[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // TODO: fix this for testing purposes! Need example data first
    // Read measurements into traccc 
    std::vector<InputData> data = instance_state->traccc_gpu_standalone_->read_from_array(input_data_geoid, input_data_clusters);

    // Initialize spacepoints and measurements for this execution run, using the instance's host_mr
    traccc::edm::spacepoint_collection::host spacepoints(instance_state->host_mr_);
    traccc::measurement_collection_types::host measurements(&instance_state->host_mr_);

    // convert to traccc types
    inputDataToTracccMeasurements(data, spacepoints, measurements, 
                                    instance_state->traccc_gpu_standalone_->getAthenaToDetrayMap());
    // run the reco chain
    auto track_states = instance_state->traccc_gpu_standalone_->run(spacepoints, measurements);

    uint64_t compute_end_ns = 0;
    SET_TIMESTAMP(compute_end_ns);

    bool supports_first_dim_batching;
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        model_state->SupportsFirstDimBatching(&supports_first_dim_batching));

    // Because the output tensor values are concatenated into a single
    // contiguous 'output_buffer', the backend must "scatter" them out
    // to the individual response output tensors.  The backend utilities
    // provide a "responder" to facilitate this scattering process.
    // BackendOutputResponder does NOT support TRITONSERVER_TYPE_BYTES
    // data type.

    // Initialize the responder
    BackendOutputResponder responder(
        requests, request_count, &responses, model_state->TritonMemoryManager(),
        supports_first_dim_batching, false /* pinned_enabled */,
        nullptr /* stream*/);

    // The 'responders's ProcessTensor function will copy the portion of
    // 'output_buffer' corresponding to each request's output into the
    // response for that request.

    // Process the outputs
    {
        // --------------- Process 'TRK_PARAMS' ---------------
        size_t num_tracks = track_states.size();
        
        // Combine track parameters: chi2, ndf, phi, eta, qop
        std::vector<float> trk_params_buffer;
        trk_params_buffer.reserve(num_tracks * 5);
        
        for (size_t i = 0; i < num_tracks; ++i) {
            const auto& [fit_res, state] = track_states.at(i);
            
            // Get the fitted track parameters
            const auto& fitted_params = fit_res.fit_params;
            
            // Extract the track parameters
            traccc::scalar phi = fitted_params.phi();
            traccc::scalar theta = fitted_params.theta();
            traccc::scalar qop = fitted_params.qop();
            
            trk_params_buffer.push_back(static_cast<float>(fit_res.trk_quality.chi2));
            trk_params_buffer.push_back(static_cast<float>(fit_res.trk_quality.ndf));
            trk_params_buffer.push_back(static_cast<float>(phi));
            trk_params_buffer.push_back(static_cast<float>(theta));
            trk_params_buffer.push_back(static_cast<float>(qop));
        }
        
        std::vector<int64_t> trk_params_shape = {static_cast<int64_t>(num_tracks), 5};
        const char* trk_params_data = reinterpret_cast<const char*>(trk_params_buffer.data());
        
        responder.ProcessTensor(
            "TRK_PARAMS",
            TRITONSERVER_TYPE_FP64,
            trk_params_shape,
            trk_params_data,
            TRITONSERVER_MEMORY_CPU,
            0 /* memory_type_id */
        );

        // --------------- Process 'MEASUREMENTS' (Flattened with separators) ---------------
        // Count total measurements across all tracks + separators
        size_t total_measurements = 0;
        for (size_t i = 0; i < num_tracks; ++i) {
            const auto& [fit_res, state] = track_states.at(i);
            total_measurements += state.size();
            if (i < num_tracks - 1) { // Add separator except after last track
                total_measurements += 1; // for the -1 separator
            }
        }
        
        // Create flattened measurements buffer with separators: [total_meas_with_separators, 4]
        std::vector<float> measurements_buffer;
        std::vector<int64_t> geometry_ids_buffer;
        measurements_buffer.reserve(total_measurements * 4);
        geometry_ids_buffer.reserve(total_measurements);
        
        for (size_t track_idx = 0; track_idx < num_tracks; ++track_idx) {
            const auto& [fit_res, state] = track_states.at(track_idx);
            
            // Add measurements for this track
            for (size_t meas_idx = 0; meas_idx < state.size(); ++meas_idx) {
                const traccc::measurement& measurement = state[meas_idx].get_measurement();
                
                const std::array<float, 2> localPosition = measurement.local;
                const std::array<float, 2> localCovariance = measurement.variance;
                
                measurements_buffer.push_back(static_cast<float>(localPosition[0])); // localx
                measurements_buffer.push_back(static_cast<float>(localPosition[1])); // localy
                measurements_buffer.push_back(static_cast<float>(localCovariance[0])); // varx
                measurements_buffer.push_back(static_cast<float>(localCovariance[1])); // vary
                
                // Convert Detray ID back to Athena ID for output using reverse map
                uint64_t detray_id = measurement.surface_link.value();
                try {
                    geometry_ids_buffer.push_back(
                        instance_state->traccc_gpu_standalone_->getDetrayToAthenaMap().at(detray_id)
                    );
                } catch (const std::out_of_range& e) {
                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, 
                                ("Missing reverse mapping for Detray ID: " + std::to_string(detray_id)).c_str());
                    // Fall back to Detray ID
                    geometry_ids_buffer.push_back(detray_id);
                }
            }
            
            // Add separator between tracks (except after last track)
            if (track_idx < num_tracks - 1) {
                measurements_buffer.push_back(-1.0); // local x
                measurements_buffer.push_back(-1.0); // local y  
                measurements_buffer.push_back(-1.0); // var x
                measurements_buffer.push_back(-1.0); // var y
                
                geometry_ids_buffer.push_back(0); // geometry ID
            }
        }
        
        std::vector<int64_t> measurements_shape = {static_cast<int64_t>(total_measurements), 4};
        const char* measurements_data = reinterpret_cast<const char*>(measurements_buffer.data());
        
        responder.ProcessTensor(
            "MEASUREMENTS",
            TRITONSERVER_TYPE_FP64,
            measurements_shape,
            measurements_data,
            TRITONSERVER_MEMORY_CPU,
            0 /* memory_type_id */
        );

        // --------------- Process 'GEOMETRY_IDS' (Flattened with separators) ---------------
        std::vector<int64_t> geometry_ids_shape = {static_cast<int64_t>(total_measurements)};
        const char* geometry_ids_data = reinterpret_cast<const char*>(geometry_ids_buffer.data());
        
        responder.ProcessTensor(
            "GEOMETRY_IDS",
            TRITONSERVER_TYPE_UINT64,
            geometry_ids_shape,
            geometry_ids_data,
            TRITONSERVER_MEMORY_CPU,
            0 /* memory_type_id */
        );
    }

    // Finalize the responder. If 'true' is returned, the output
    // tensors' data will not be valid until the backend synchronizes
    // the CUDA stream or event that was used when creating the
    // responder. For this backend, GPU is not supported and so no CUDA
    // sync should be needed; so if 'true' is returned simply log an
    // error.
    const bool need_cuda_output_sync = responder.Finalize();
    if (need_cuda_output_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'traccc' backend: unexpected CUDA sync required by responder");
    }

    // Send all the responses that haven't already been sent because of
    // an earlier error.
    for (auto& response : responses) {
    if (response != nullptr) {
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
            "failed to send response");
    }
    }

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
    // For batch statistics need to know the total batch size of the
    // requests. This is not necessarily just the number of requests,
    // because if the model supports batching then any request can be a
    // batched request itself.
    size_t total_batch_size = 0;
    if (!supports_first_dim_batching) {
    total_batch_size = request_count;
    } else {
        for (uint32_t r = 0; r < request_count; ++r) 
        {
            auto& request = requests[r];
            TRITONBACKEND_Input* input = nullptr;
            LOG_IF_ERROR(
                TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
                "failed getting request input");
            if (input != nullptr) 
            {
                const int64_t* shape = nullptr;
                LOG_IF_ERROR(
                    TRITONBACKEND_InputProperties(
                        input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr),
                    "failed getting input properties");
                if (shape != nullptr) 
                {
                    total_batch_size += shape[0];
                }
            }
        }
    }
#else
    (void)exec_start_ns;
    (void)exec_end_ns;
    (void)compute_start_ns;
    (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

// Report statistics for each request, and then release the request.
for (uint32_t r = 0; r < request_count; ++r) 
{
    auto& request = requests[r];

    #ifdef TRITON_ENABLE_STATS
        LOG_IF_ERROR(
            TRITONBACKEND_ModelInstanceReportStatistics(
                instance_state->TritonModelInstance(), request,
                (responses[r] != nullptr) /* success */, exec_start_ns,
                compute_start_ns, compute_end_ns, exec_end_ns),
            "failed reporting request statistics");
    #endif  // TRITON_ENABLE_STATS

        LOG_IF_ERROR(
            TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
            "failed releasing request");
}

#ifdef TRITON_ENABLE_STATS
    // Report batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            instance_state->TritonModelInstance(), total_batch_size,
            exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
#endif  // TRITON_ENABLE_STATS

return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::traccc
