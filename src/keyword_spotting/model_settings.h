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

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
#if defined(USE_DNN_SMALL_INT8_MODEL) || defined(USE_DNN_MEDIUM_INT8_MODEL) || defined(USE_DNN_LARGE_INT8_MODEL) || defined(USE_DNN_SMALL_FLOAT32_MODEL) || defined(USE_DNN_MEDIUM_FLOAT32_MODEL)
constexpr int kNumCols = 250;
constexpr int kNumRows = 1;
constexpr int kNumChannels = 1;
#else // CNN models
constexpr int kNumCols = 490;
constexpr int kNumRows = 1;
constexpr int kNumChannels = 1;
#endif

constexpr int kMaxDataSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 12;
