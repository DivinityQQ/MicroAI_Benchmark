#include <Arduino.h>
#include "config.h"

#ifdef USE_IMAGE_MODEL
#include "image_recognition/image_recognition_benchmark.h"
#elif defined(USE_SPEECH_MODEL)
#include "speech_recognition/speech_recognition_benchmark.h"
#endif

void setup() {
  #ifdef USE_IMAGE_MODEL
  image_recognition_setup();
  #elif defined(USE_SPEECH_MODEL)
  speech_recognition_setup();
  #endif
}

void loop() {
  #ifdef USE_IMAGE_MODEL
  image_recognition_loop();
  #elif defined(USE_SPEECH_MODEL)
  speech_recognition_loop();
  #endif
}