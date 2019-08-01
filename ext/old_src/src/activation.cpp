#include <math.h>
#include "activation.hpp"

//Similar to sigmoid but no exp
float cai::linear_rectify(float x, float alpha) {
    return ((x / (1 + fabs(x * alpha))) + 1) / 2;
}


float cai::sigmoid(float x, float alpha) {
    return 1 / (1 + exp(-x * alpha));
}
