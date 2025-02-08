
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
#include "model_settings.h"
#include "data_provider.h"

// #include "person_image_data.h"
// #include "no_person_image_data.h"

TfLiteStatus GetData(tflite::ErrorReporter* error_reporter, int first_dimension,
                            int second_dimension, int third_dimension, int16_t* test_data) {

  int data_size = first_dimension * second_dimension * third_dimension;

  if (test_data == nullptr || data_size <= 0) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invalid test_data pointer or non-positive data_size (%d)", data_size);
    return kTfLiteError;
  }

  // Fill the test_data array with zeros
  for (int i = 0; i < data_size; ++i) {
    test_data[i] = 0; // Initialize with zeros
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Initialized test data with zeros (%d)", data_size);

  return kTfLiteOk;
}