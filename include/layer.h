#pragma once

#include <vector>
#include <string>
#include <functional>
#include "./core_math.h"

namespace neuralnet{

class Layer{

public:
    Layer(const size_t num_in, const size_t num_out, const std::string activation_function, const std::string initializer) 
        : fan_in(num_in), fan_out(num_out), activation_function_name(activation_function){
            initialize(neurons, num_in, 1, "zeros");
            initialize(weights, num_in, num_out, initializer);
            initialize(bias, num_out, 1, initializer);
            this->set_activation_function();     
    };

    // initialize with weights and bias with range
    Layer(const size_t num_in, const size_t num_out, const std::string activation_function, const double low, const double high) 
        : fan_in(num_in), fan_out(num_out), activation_function_name(activation_function){
            initialize(neurons, num_in, 1, "zeros");
            initialize(weights, num_in, num_out, low, high);
            initialize(bias, num_out, 1, low, high);
            this->set_activation_function();
    };

    // initialize with no bias
    Layer(const size_t num_in, const size_t num_out, const std::string activation_function, const std::string initializer, const std::string no_bias) 
        : fan_in(num_in), fan_out(num_out), activation_function_name(activation_function){
            initialize(neurons, num_in, 1, "zeros");
            initialize(weights, num_in, num_out, initializer);
            initialize(bias, num_out, 1, initializer);
            this->set_activation_function();
    };

    // initialize with no bias and weight with range
    Layer(const size_t num_in, const size_t num_out, const std::string activation_function, const double low, const double high, const std::string no_bias) 
        : fan_in(num_in), fan_out(num_out), activation_function_name(activation_function){
            initialize(neurons, num_in, 1, "zeros");
            initialize(weights, num_in, num_out, low, high);
            initialize(bias, num_out, 1, "zeros");
            this->set_activation_function();
    };

    // initialize output layer
    Layer(const size_t num_in, const std::string activation_function)
        : fan_in(num_in), fan_out(0), activation_function_name(activation_function) {
        initialize(neurons, num_in, 1, "zeros");
        this->set_activation_function();
        /*free(fan_in)
        free(weights);
        free(bias);
        free(fan_in);*/
        //delete *fan_in
    }

    //~Layer();

    // set activation function
    void set_activation_function();
    
    // activate neuron using activation function
    void activate();

    // compute gradient of activation fuction 
    matrix::Matrix gradients();
    
    // perform forward propagate: W.T @ x + b
    void pass_forward(Layer &next_layer);

    // update weights and bias one step
    void step(matrix::Matrix &weight_update);

    // set neuron values to input parameter
    void feed(std::vector<double> &input);

    // return activated neurons
    matrix::Matrix get_activated_neurons();

    // return weights
    matrix::Matrix get_weights();
    
private:
    const size_t fan_in;
    const size_t fan_out;
    const std::string activation_function_name;
    std::function<double(double)> activation_function;
    std::function<double(double)> activation_function_derivative;
    matrix::Matrix neurons;
    matrix::Matrix activated_neurons;
    matrix::Matrix weights;
    matrix::Matrix transposed_weights;
    matrix::Matrix bias;
};
};
