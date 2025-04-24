/* Licensed under the Apache License, Version 2.0 (the "License");
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
#ifdef USE_KWS_SCRAMBLED_16_8_BENCHMARK

#include <Arduino.h>

#if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4)
// include main library header file
#include <TensorFlow_Lite_ESP_NN.h>
#else
// include main library header file
#include <TensorFlow_Lite_CMSIS_NN.h>
#endif

// #include "detection_responder.h"
#include "data_provider.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#ifdef ENABLE_PROFILING
#include "tensorflow/lite/micro/micro_profiler.h"
#endif
#ifdef ENABLE_MEMORY_MONITORING
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#endif
#ifdef USE_8BIT_MODEL
#include "keyword_scrambled_8bit_model_data.h"
#else
#include "keyword_scrambled_model_data.h"
#endif

// Globals, used for compatibility with Arduino-style sketches.
namespace {
#ifdef ENABLE_PROFILING
tflite::MicroProfiler profiler;
#endif
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
#ifdef ENABLE_MEMORY_MONITORING
tflite::RecordingMicroInterpreter* interpreter = nullptr;
#else
tflite::MicroInterpreter* interpreter = nullptr;
#endif
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4)
constexpr int scratchBufSize = 60 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 20 * 1024 + scratchBufSize;
alignas(16) uint8_t tensor_arena[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

// The name of this function is important for Arduino compatibility.
void kws_scrambled_setup() {

  // Enable serial only when profiling is enabled and you intend to connect the kit to PC,
  // on some boards it might hang otherwise
  #if defined(ENABLE_PROFILING) || defined(ENABLE_LOGGING)
  Serial.begin(115200);
  while(!Serial);
  #endif

  #ifdef CONFIG_IDF_TARGET_ESP32C6
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);

  // turn the LED off by making the voltage LOW
  digitalWrite(LED_BUILTIN, LOW);
  #endif

  #ifdef ARDUINO_RASPBERRY_PI_PICO_W
  // digitalWrite(PIN_LED, HIGH);

  // Put the SMPS into PWM mode for better power consumption readibility
  // at the cost of higher idle power consumption
  digitalWrite(32+1, HIGH);
  #endif

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model_data);
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
  micro_op_resolver.AddSvdf();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  #if defined(ENABLE_MEMORY_MONITORING) && defined(ENABLE_PROFILING)
  // Both memory monitoring and profiling enabled
  static tflite::RecordingMicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
  #elif defined(ENABLE_MEMORY_MONITORING)
  // Only memory monitoring enabled
  static tflite::RecordingMicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  #elif defined(ENABLE_PROFILING)
  // Only profiling enabled
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
  #else
  // Neither memory monitoring nor profiling enabled
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

  // Report the actual memory usage
  size_t used_bytes = interpreter->arena_used_bytes();
  TF_LITE_REPORT_ERROR(error_reporter, "Tensor Arena Used: %d bytes", used_bytes);

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

}

// The name of this function is important for Arduino compatibility.
void kws_scrambled_loop() {

  // Get data from provider.
  if (kTfLiteOk != GetDataKWSInt16(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.i16)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Data load failed.");
  }

  #ifdef ENABLE_PROFILING
  // Code path when logging is enabled, affects power consumption
  // Start profiling the inference event
  uint32_t event_handle = profiler.BeginEvent("Keyword detection invoke");
  #endif

  #ifdef ENABLE_LOGGING
  // Record time before inference
  unsigned long start_time = millis();
  #endif

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  #ifdef ENABLE_LOGGING
  // Record time after inference
  unsigned long end_time = millis();
  #endif

  #ifdef ENABLE_PROFILING
  // End profiling for this event
  profiler.EndEvent(event_handle);

  // Log the profiling data
  profiler.LogTicksPerTag();

  profiler.ClearEvents();
  #endif

  #ifdef ENABLE_MEMORY_MONITORING
  // Print out detailed allocation information:
  interpreter->GetMicroAllocator().PrintAllocations();
  #endif
  
  #ifdef ENABLE_LOGGING
  // Calculate inference time
  unsigned long inference_time = end_time - start_time;

  TF_LITE_REPORT_ERROR(error_reporter, "Inference time (ms): %d", inference_time);
  #endif

  TfLiteTensor* output = interpreter->output(0);

  // Use delay to make the loop more distinguishable
  delay(500);
}

#endif // USE_KWS_SCRAMBLED_16_8_BENCHMARK