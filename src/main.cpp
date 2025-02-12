#include <Arduino.h>
#include "config.h"

#ifdef USE_PERSON_DETECTION_BENCHMARK
#include "person_detection/person_detection_benchmark.h"
#elif defined(USE_KWS_SCRAMBLED_16_8_BENCHMARK)
#include "keyword_spotting_scrambled/kws_scrambled_benchmark.h"
#elif defined(USE_SPEECH_YES_NO_BENCHMARK)
#include "speech_yes_no_recognition/speech_yes_no_benchmark.h"
#elif defined(USE_VISUAL_WAKEWORD_BENCHMARK)
#include "visual_wakeword_detection/visual_wakeword_detection_benchmark.h"
#elif defined(USE_KWS_BENCHMARK)
#include "keyword_spotting/keyword_spotting_benchmark.h"
#endif

void setup() {
  #ifdef USE_PERSON_DETECTION_BENCHMARK
  person_detection_setup();
  #elif defined(USE_KWS_SCRAMBLED_16_8_BENCHMARK)
  kws_scrambled_setup();
  #elif defined(USE_SPEECH_YES_NO_BENCHMARK)
  speech_yes_no_setup();
  #elif defined(USE_VISUAL_WAKEWORD_BENCHMARK)
  visual_wakeword_detection_setup();
  #elif defined(USE_KWS_BENCHMARK)
  keyword_spotting_setup();
  #endif
}

void loop() {
  #ifdef USE_PERSON_DETECTION_BENCHMARK
  person_detection_loop();
  #elif defined(USE_KWS_SCRAMBLED_16_8_BENCHMARK)
  kws_scrambled_loop();
  #elif defined(USE_SPEECH_YES_NO_BENCHMARK)
  speech_yes_no_loop();
  #elif defined(USE_VISUAL_WAKEWORD_BENCHMARK)
  visual_wakeword_detection_loop();
  #elif defined(USE_KWS_BENCHMARK)
  keyword_spotting_loop();
  #endif
}