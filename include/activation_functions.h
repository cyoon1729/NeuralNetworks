#pragma once

#include <cmath>

namespace neuralnet{

inline double sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

inline double sigmoid_derivative(double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

inline double hyperbolic_tan(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

inline double hyperbolic_tan_derivative(double x){
    return 1 - pow(tanh(x), 2);
}

inline double relu(double x){
    return x > 0 ? x : 0;
}

inline double relu_derivative(double x){
    return x > 0 ? 1 : 0;
}
    
inline double linear(double x){
    return x;
}

inline double linear_derivative(double x){
    return 1;
}

};