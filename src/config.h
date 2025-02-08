#pragma once

// Uncomment ONE model to use

#define USE_IMAGE_MODEL
// #define USE_SPEECH_MODEL

// Common configuration

// Whether or not to log inference time data,
// for measuring power consumption, not defining this is recommended
#define ENABLE_PROFILING
// #define ENABLE_LOGGING

//Image recognition benchmark configuration

#define NUM_ITERATIONS 1
#define START_WITH_PERSON 0
