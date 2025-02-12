/* Copyright 2020-2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "config.h"
#ifdef USE_SPEECH_YES_NO_BENCHMARK

#include <Arduino.h>

#if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4)
// include main library header file
#include <TensorFlow_Lite_ESP_NN.h>
#else
// include main library header file
#include <TensorFlow_Lite_CMSIS_NN.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#ifdef ENABLE_PROFILING
#include "tensorflow/lite/micro/micro_profiler.h"
#endif

// Globals, used for compatibility with Arduino-style sketches.
namespace {
#ifdef ENABLE_PROFILING
tflite::MicroProfiler profiler;
#endif
const tflite::Model* model = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 30 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

// The name of this function is important for Arduino compatibility.
void speech_yes_no_setup() {

  // Enable serial only when profiling is enabled and you intend to connect the kit to PC,
  // on some boards it might hang otherwise
  #if defined(ENABLE_PROFILING) || defined(ENABLE_LOGGING)
  Serial.begin(115200);
  while(!Serial);
  #endif

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  #ifdef ENABLE_PROFILING
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
  #else
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  #endif
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureCount * kFeatureSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = tflite::GetTensorData<int8_t>(model_input);

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

}

// The name of this function is important for Arduino compatibility.
void speech_yes_no_loop() {
  // Fetch the spectrogram for the current time.
  const int32_t dummy_time = 0;
  int how_many_new_slices = 0;

  #ifdef ENABLE_LOGGING
  // Record time before inference
  unsigned long start_time_spectrogram = millis();
  #endif

  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      dummy_time, dummy_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }

  #ifdef ENABLE_LOGGING
  // Record time after inference
  unsigned long end_time_spectrogram = millis();

  // Calculate inference time
  unsigned long inference_time_spectrogram = end_time_spectrogram - start_time_spectrogram;

  TF_LITE_REPORT_ERROR(error_reporter, "Spectrogram builder inference time (ms): %d", inference_time_spectrogram);
  #endif

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  #ifdef ENABLE_PROFILING
  // Start profiling the inference event
  uint32_t event_handle = profiler.BeginEvent("Speech recognition invoke");
  #endif

  #ifdef ENABLE_LOGGING
  // Record time before inference
  unsigned long start_time_recognition = millis();
  #endif

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  #ifdef ENABLE_LOGGING
  // Record time after inference
  unsigned long end_time_recognition = millis();
  #endif

  #ifdef ENABLE_PROFILING
  // End profiling for this event
  profiler.EndEvent(event_handle);

  // Log the profiling data
  profiler.LogTicksPerTag();

  profiler.ClearEvents();
  #endif

  #ifdef ENABLE_LOGGING
  // Calculate inference time
  unsigned long inference_time_recognition = end_time_recognition - start_time_recognition;

  TF_LITE_REPORT_ERROR(error_reporter, "Speech ecognition inference time (ms): %d", inference_time_recognition);

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);

  // using simple argmax instead of recognizer
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;
  int max_idx = 0;
  float max_result = 0.0;
  // Dequantize output values and find the max
  for (int i = 0; i < kCategoryCount; i++) {
    float current_result =
        (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) *
        output_scale;
    if (current_result > max_result) {
      max_result = current_result; // update max result
      max_idx = i; // update category
    }
  }
  if (max_result > 0.8f) {
    TF_LITE_REPORT_ERROR(error_reporter, "Detected %s, score: %f", kCategoryLabels[max_idx],
      static_cast<double>(max_result));
  } else {
    TF_LITE_REPORT_ERROR(error_reporter, "No detection");
  }
  #endif

  delay(500);
}

#endif // USE_SPEECH_YES_NO_BENCHMARK