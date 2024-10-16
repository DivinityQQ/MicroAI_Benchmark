
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
#include "image_provider.h"

#include "person_image_data.h"
#include "no_person_image_data.h"

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data, bool person) {

  if(person) {
    // Copy BMP image data into the image_data array
    for (int i = 0; i < image_width * image_height; i++) {
    // Quantize the BMP data from [0, 255] to [-128, 127]
    image_data[i] = static_cast<int8_t>(g_person_image_data[i]) - 128;
    }
    TF_LITE_REPORT_ERROR(error_reporter, "Loaded image WITH person");
  }

  else {
    // Copy BMP image data into the image_data array
    for (int i = 0; i < image_width * image_height; i++) {
    // Quantize the BMP data from [0, 255] to [-128, 127]
    image_data[i] = static_cast<int8_t>(g_no_person_image_data[i]) - 128;
    }
    TF_LITE_REPORT_ERROR(error_reporter, "Loaded image with NO person");
  }

  return kTfLiteOk;
}