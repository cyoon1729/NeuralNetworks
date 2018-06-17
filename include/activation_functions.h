#pragma once

#include <cmath>
#include <array>


inline double sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

inline double d_sigmoid(double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

inline double tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

inline double d_tanh(double x){
    return 1 - pow(tanh(x), 2);
}

inline double relu(double x){
    return x > 0 ? x : 0;
}

inline double d_relu(double x){
    return x > 0 ? 1 : 0;
}
    
    /*
    inline double softmax(array<int> weights_ij){
        return 
    }
    */
