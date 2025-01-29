/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <cstring>
#include "feature_provider.h"

// #include "audio_provider.h"
#include "micro_features_generator.h"
#include "micro_model_settings.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

#include "test_data/yes_1000ms.h"
#include "test_data/no_1000ms.h"
#include "test_data/noise_1000ms.h"
#include "test_data/silence_1000ms.h"

// extern const uint8_t no_30ms_start[]          asm("_binary_no_30ms_wav_start");
// extern const uint8_t yes_30ms_start[]         asm("_binary_yes_30ms_wav_start");
extern const uint8_t yes_1000ms_start[];
extern const uint8_t no_1000ms_start[];
extern const uint8_t noise_1000ms_start[];
extern const uint8_t silence_1000ms_start[];

Features g_features;

FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(
    int32_t last_time_in_ms, int32_t time_in_ms, int* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    TF_LITE_REPORT_ERROR(nullptr, "Requested feature_data_ size %d doesn't match %d",
               feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  const int last_step = (last_time_in_ms / kFeatureStrideMs);
  const int current_step = (time_in_ms / kFeatureStrideMs);

  int slices_needed = current_step - last_step;
  // If this is the first call, make sure we don't use any cached information.
  if (is_first_run_) {
    TfLiteStatus init_status = InitializeMicroFeatures();
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    is_first_run_ = false;
    slices_needed = kFeatureCount;
  }

  *how_many_new_slices = kFeatureCount;
  int16_t* audio_samples = nullptr;
  int audio_samples_size = 0;
  static int cnt = 0;
  audio_samples = (int16_t *) (silence_1000ms_start + 44);
  switch(cnt++ % 4) {
    case 0: audio_samples = (int16_t*)(yes_1000ms_start + 44); break;
    case 1: audio_samples = (int16_t*)(no_1000ms_start + 44); break;
    case 2: audio_samples = (int16_t*)(noise_1000ms_start + 44); break;
    case 3: audio_samples = (int16_t*)(silence_1000ms_start + 44); break;
  }
  audio_samples_size = 16000;
  
  TfLiteStatus generate_status = GenerateFeatures(
        audio_samples, audio_samples_size, &g_features);
  if (generate_status != kTfLiteOk) {
    return generate_status;
  }
  // copy features
  for (int i = 0; i < kFeatureCount; ++i) {
    for (int j = 0; j < kFeatureSize; ++j) {
      feature_data_[i * kFeatureSize + j] = g_features[i][j];
    }
  }

  return kTfLiteOk;
}
