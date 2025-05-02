/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "micro_features_generator.h"

#include <cmath>
#include <cstring>
#include "audio_preprocessor_int8_model_data.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "micro_model_settings.h"
#ifdef ENABLE_PROFILING
#include "tensorflow/lite/micro/micro_profiler.h"
#endif
#ifdef ENABLE_MEMORY_MONITORING
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#endif

namespace {

#ifdef ENABLE_PROFILING
tflite::MicroProfiler spectrogram_profiler;
#endif

// FrontendState g_micro_features_state;
bool g_is_first_time = true;

const tflite::Model* model = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
#ifdef ENABLE_MEMORY_MONITORING
tflite::RecordingMicroInterpreter* interpreter = nullptr;
#else
tflite::MicroInterpreter* interpreter = nullptr;
#endif

#if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4)
constexpr int scratchBufSize = 4 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
constexpr size_t kArenaSize = 10 * 1024 + scratchBufSize;
alignas(16) uint8_t g_arena[kArenaSize];

constexpr int kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr int kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;
}  // namespace

TfLiteStatus RegisterOps(AudioPreprocessorOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
  return kTfLiteOk;
}

TfLiteStatus InitializeMicroFeatures() {
  g_is_first_time = true;

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_audio_preprocessor_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  static AudioPreprocessorOpResolver op_resolver;
  RegisterOps(op_resolver);

  #if defined(ENABLE_MEMORY_MONITORING) && defined(ENABLE_PROFILING)
  // Both memory monitoring and profiling enabled
  static tflite::RecordingMicroInterpreter static_interpreter(
      model, op_resolver, g_arena, kArenaSize, nullptr, &spectrogram_profiler);
  #elif defined(ENABLE_MEMORY_MONITORING)
  // Only memory monitoring enabled
  static tflite::RecordingMicroInterpreter static_interpreter(
      model, op_resolver, g_arena, kArenaSize);
  #elif defined(ENABLE_PROFILING)
  // Only profiling enabled
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, g_arena, kArenaSize, nullptr, &spectrogram_profiler);
  #else
  // Neither memory monitoring nor profiling enabled
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, g_arena, kArenaSize);
  #endif
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors failed for Feature provider model. Line %d", __LINE__);
    return kTfLiteError;
  }

  // MicroPrintf("AudioPreprocessor model arena size = %u",
  //             interpreter.arena_used_bytes());

  return kTfLiteOk;
}

TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);
  std::copy_n(audio_data, audio_data_size,
              tflite::GetTensorData<int16_t>(input));
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generator model invocation failed");
  }

  std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize,
              feature_output);

  return kTfLiteOk;
}

TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output) {
  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;

  #ifdef ENABLE_PROFILING
  // Start profiling the inference event
  uint32_t event_handle = spectrogram_profiler.BeginEvent("Spectrogram builder invoke");
  #endif

  while (remaining_samples >= kAudioSampleDurationCount &&
         feature_index < kFeatureCount) {
    TF_LITE_ENSURE_STATUS(
        GenerateSingleFeature(audio_data, kAudioSampleDurationCount,
                              (*features_output)[feature_index], interpreter));
    feature_index++;
    audio_data += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }

  #ifdef ENABLE_PROFILING
  // End profiling for this event
  spectrogram_profiler.EndEvent(event_handle);

  // Log the profiling data
  spectrogram_profiler.LogTicksPerTag();

  spectrogram_profiler.ClearEvents();
  #endif

  #ifdef ENABLE_MEMORY_MONITORING
  // Print out detailed allocation information:
  interpreter->GetMicroAllocator().PrintAllocations();
  #endif

  return kTfLiteOk;
}
