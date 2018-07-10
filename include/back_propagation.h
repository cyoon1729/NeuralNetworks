#include <vector>
#include <cmath>
#include <string>
#include "layer.h"
#include "ann.h"
#include "matrix_operations.h"

ANN backprop(ANN neuralnet, std::vector<double> target);
