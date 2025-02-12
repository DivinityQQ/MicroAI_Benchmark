// image_recognition_benchmark.h
#pragma once

#include "config.h"
#ifdef USE_PERSON_DETECTION_BENCHMARK

void person_detection_setup();
void person_detection_loop();

#endif // USE_PERSON_DETECTION_BENCHMARK