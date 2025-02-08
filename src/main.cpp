#include <Arduino.h>
#include "config.h"

#ifdef USE_IMAGE_MODEL
#include "image_recognition/image_recognition_benchmark.h"
#elif defined(USE_KEYWORD_MODEL)
#include "keyword_detection/keyword_detection_8bit_benchmark.h"
#elif defined(USE_SPEECH_MODEL)
#include "speech_recognition/speech_recognition_benchmark.h"
#endif

void setup() {
  #ifdef USE_IMAGE_MODEL
  image_recognition_setup();
  #elif defined(USE_KEYWORD_MODEL)
  keyword_detection_setup();
  #elif defined(USE_SPEECH_MODEL)
  speech_recognition_setup();
  #endif
}

void loop() {
  #ifdef USE_IMAGE_MODEL
  image_recognition_loop();
  #elif defined(USE_KEYWORD_MODEL)
  keyword_detection_loop();
  #elif defined(USE_SPEECH_MODEL)
  speech_recognition_loop();
  #endif
}