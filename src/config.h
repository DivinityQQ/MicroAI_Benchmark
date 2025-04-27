#pragma once

/*********************************************
 *               BENCHMARK SELECTION
 *********************************************/

//    Uncomment ONE benchmark to use
//    #define USE_PERSON_DETECTION_BENCHMARK
//    #define USE_SPEECH_YES_NO_BENCHMARK
//    #define USE_KWS_SCRAMBLED_16_8_BENCHMARK
//    #define USE_VISUAL_WAKEWORD_BENCHMARK
//    #define USE_KWS_BENCHMARK
    #define USE_NOISE_REDUCTION_BENCHMARK

/*********************************************
 *          COMMON CONFIGURATION
 *********************************************/

//    Whether or not to log and/or profile inference time data
//    For measuring power consumption, not defining this is recommended
    #define ENABLE_PROFILING
    #define ENABLE_MEMORY_MONITORING
//    #define ENABLE_LOGGING

/*********************************************
 *   PERSON DETECTION BENCHMARK SETTINGS
 *********************************************/

    #define NUM_ITERATIONS 1
    #define START_WITH_PERSON 0

/*********************************************
 *  SPEECH YES NO BENCHMARK SETTINGS
 *********************************************/

//    (Add relevant configuration settings here)

/*********************************************
 *  KEYWORD SPOTTING SCRAMBLED BENCHMARK SETTINGS
 *********************************************/

//    Uncomment ONE model to use
    #define USE_8BIT_MODEL
//    #define USE_STANDARD_MODEL

/*********************************************
 *  VISUAL WAKEWORD DETECTION BENCHMARK SETTINGS
 *********************************************/

//    #define USE_128x128x1_MODEL
    #define USE_96x96x3_MODEL

/*********************************************
 *  KEYWORD SPOTTING BENCHMARK SETTINGS
 *********************************************/

    #define USE_CNN_SMALL_FLOAT32_MODEL
//    #define USE_CNN_MEDIUM_FLOAT32_MODEL
//    #define USE_CNN_SMALL_INT8_MODEL
//    #define USE_CNN_MEDIUM_INT8_MODEL
//    #define USE_CNN_LARGE_INT8_MODEL
//    #define USE_DNN_SMALL_INT8_MODEL
//    #define USE_DNN_MEDIUM_INT8_MODEL
//    #define USE_DNN_LARGE_INT8_MODEL
//    #define USE_DNN_SMALL_FLOAT32_MODEL
//    #define USE_DNN_MEDIUM_FLOAT32_MODEL
//    #define USE_DS_CNN_SMALL_INT8_MODEL
//    #define USE_DS_CNN_MEDIUM_INT8_MODEL
//    #define USE_DS_CNN_LARGE_INT8_MODEL
//    #define USE_DS_CNN_SMALL_FLOAT32_MODEL
//    #define USE_DS_CNN_MEDIUM_FLOAT32_MODEL
//    #define USE_DS_CNN_SMALL_INT16_MODEL