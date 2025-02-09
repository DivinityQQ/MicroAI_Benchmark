// image_recognition_benchmark.h
#pragma once

#include "config.h"
#ifdef USE_WAKEWORD_BENCHMARK

void wakeword_detection_setup();
void wakeword_detection_loop();

#endif // USE_WAKEWORD_BENCHMARK