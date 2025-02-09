#pragma once

/*********************************************
 *               BENCHMARK SELECTION
 *********************************************/

// Uncomment ONE benchmark to use
// #define USE_IMAGE_BENCHMARK
// #define USE_SPEECH_BENCHMARK
#define USE_KEYWORD_BENCHMARK

/*********************************************
 *          COMMON CONFIGURATION
 *********************************************/

// Whether or not to log and/or profile inference time data
// For measuring power consumption, not defining this is recommended
#define ENABLE_PROFILING
// #define ENABLE_LOGGING

/*********************************************
 *   IMAGE RECOGNITION BENCHMARK SETTINGS
 *********************************************/

#define NUM_ITERATIONS 1
#define START_WITH_PERSON 0

/*********************************************
 *  SPEECH RECOGNITION BENCHMARK SETTINGS
 *********************************************/

// (Add relevant configuration settings here)

/*********************************************
 *  KEYWORD DETECTION BENCHMARK SETTINGS
 *********************************************/

// Uncomment ONE model to use
#define USE_8BIT_MODEL
// #define USE_STANDARD_MODEL