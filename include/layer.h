#pragma once

#include <vector>
#include <string>
#include <utility>
#include "./matrix_operations.h"

namespace neuralnet{

class Layer{

public:
    Layer(const size_t num_in, const size_t num_out, const std::string activation_function, const std::string initializer) 
        : fan_in(num_in), fan_out(num_out), activation_function_name(activation_function){
            neurons(1, num_out);
            neurons.fill_weight("zeros");
            weights(num_in, num_out);
            weights.fill_weight(initializer);
            bias(1, num_out);
            bias.fill_weight(initializer);
    };

    // initialize with weights and bias with range
    Layer(const size_t num_in, const size_t num_out, const std::string activation_function, const double low, const double high) 
        : fan_in(num_in), fan_out(num_out), activation_function_name(activation_function){
            neurons(1, num_out);
            neurons.fill_weight("zeros");
            weights(num_in, num_out);
            weights.fill_weight(low, high);
            bias(1, num_out);
            bias.fill_weight(low, high);
    };

    // initialize with no bias
    Layer(const size_t num_in, const size_t num_out, const std::string activation_function, const std::string initializer, const std::string no_bias) 
        : fan_in(num_in), fan_out(num_out), activation_function_name(activation_function){
            neurons(1, num_out);
            neurons.fill_weight("zeros");
            weights(num_in, num_out);
            weights.fill_weight(initializer)
            bias(1, num_out);
            bias.fill_weight(initializer)
    };

    // initialize with no bias and weight with range
    Layer(const size_t num_in, const size_t num_out, const std::string activation_function, const double low, const double high, const std::string no_bias) 
        : fan_in(num_in), fan_out(num_out), activation_function_name(activation_function){
            neurons(1, num_out);
            neurons.fill_weight("zeros");
            weights(num_in, num_out);
            weights.fill_weight(low, high);
            bias(1, num_out);
            bias.fill_weight("zeros");
    };

    // initialize output layer
    Layer(const size_t num_out, const std::string activation_function)
        : fan_out(num_out), activation_function_name(activation_function) {
        neurons(1, num_out, "zeros");
        free(weigts);
        free(bias);
        free(fan_in);
    }

    ~Layer();
    
    // activate neuron using activation function
    const matrix::Matrix activate();

    // compute gradient of activation fuction 
    const matrix::Matrix gradients();
    
    // perform forward propagate: W.T @ x + b
    const matrix::Matrix forward():

    // update weights and bias one step
    void step();
    
private:
    const size_t fan_in;
    const size_t fan_out;
    const std::string activation_function_name;
    // const std::string initialize_name;
    matrix::Matrix neurons;
    matrix::Matrix weights;
    matrix::Matrix bias;
};

// perform forward propagate from layer to next layer
void pass_forward(Layer& lhs, Layer& rhs);
};


class Layer{
    public:
        Layer(int _num_neurons, std::string layer_id);

        inline std::string return_layer_identity(){
            return this->layer_identity;
        
        } 
        
        inline int return_num_neurons(){
            return this->num_neurons;
        }

        inline void set_Lmomentum(double x){
            this->momentum = x;
        }

        inline void set_learningRate(double x){
            this->learning_rate = x;
        }
        
        void initialize_delta_weights(int r, int c);
        
        //only for output layer
        inline void set_delta(std::vector< std::vector<double> > d){
            this->delta = d;
        }

        inline std::vector< std::vector<double> > get_delta(){
            return this->delta;
        }

        void initialize_weights(int r, int c);
        
        void set_activation_function(std::string _activation_funct);

        double forward_activation_function(double x);

        double backward_activation_function(double x);

        void z_values(std::vector<double> in);

        std::vector<double> get_z_values();
        
        void activation_values();

        std::vector<double> get_activation_values();

        inline std::vector<Neuron> return_neurons(){
            return this->neurons;
        }
        
        std::vector< std::vector<double> > forward_propagate();

        
        inline void set_delta_weights(std::vector< std::vector<double> > dw){
            this->delta_weights = dw;
        }

        inline std::vector< std::vector<double> > get_delta_weights(){
            return this->delta_weights;
        }

        void update_weights();

        std::vector< std::vector<double> > weights;
        std::vector< std::vector<double> > weights_update;
        std::vector< std::vector<double> > delta_weights;
        std::vector< std::vector<double> > delta;
        std::vector<Neuron> neurons;

    private:
        //std::vector<Neuron> neurons;
        std::string activation_funct;
        std::string layer_identity;

        int num_neurons;
        double learning_rate;
        double momentum;
        double regularization;

};


