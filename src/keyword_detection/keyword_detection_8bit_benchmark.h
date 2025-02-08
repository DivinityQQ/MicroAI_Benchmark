// image_recognition_benchmark.h
#pragma once

#include "config.h"
#ifdef USE_KEYWORD_MODEL

void keyword_detection_setup();
void keyword_detection_loop();

#endif // USE_KEYWORD_MODEL