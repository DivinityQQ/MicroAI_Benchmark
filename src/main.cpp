#include <Arduino.h>
#include "config.h"

#ifdef USE_IMAGE_BENCHMARK
#include "image_recognition/image_recognition_benchmark.h"
#elif defined(USE_KEYWORD_BENCHMARK)
#include "keyword_detection/keyword_detection_benchmark.h"
#elif defined(USE_SPEECH_BENCHMARK)
#include "speech_recognition/speech_recognition_benchmark.h"
#elif defined(USE_WAKEWORD_BENCHMARK)
#include "wakeword_detection/wakeword_detection_benchmark.h"
#endif

void setup() {
  #ifdef USE_IMAGE_BENCHMARK
  image_recognition_setup();
  #elif defined(USE_KEYWORD_BENCHMARK)
  keyword_detection_setup();
  #elif defined(USE_SPEECH_BENCHMARK)
  speech_recognition_setup();
  #elif defined(USE_WAKEWORD_BENCHMARK)
  wakeword_detection_setup();
  #endif
}

void loop() {
  #ifdef USE_IMAGE_BENCHMARK
  image_recognition_loop();
  #elif defined(USE_KEYWORD_BENCHMARK)
  keyword_detection_loop();
  #elif defined(USE_SPEECH_BENCHMARK)
  speech_recognition_loop();
  #elif defined(USE_WAKEWORD_BENCHMARK)
  wakeword_detection_loop();
  #endif
}