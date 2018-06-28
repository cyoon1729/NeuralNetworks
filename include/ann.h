#pragma once

#include "layer.h"
#include <vector>
#include <string>
#include <iostream>


class ANN{
    public:
        ANN(int _num_inputs, int _num_hidden_layers, int _num_hidden_neurons, int _num_outputs);
        
        std::vector<Layer> layers;
        //set learning rate for all layers
        void set_learning_rate(double x);
        
        //set momentum for all layers
        void set_momentum(double x);   
        
        inline void set_desired_error(double x){
            this->desired_error = x;
        }

        inline void set_max_epochs(int x){
            this->max_epochs = x;
        }

        inline std::vector<double> return_inputs(){
            return this->inputs;
        }

        inline std::vector<double> return_outputs(){
            return this->outputs;
        }

        inline double return_error(){
            return this->error;
        }

        double cross_entropy_error(std::vector< std::vector<double> > target, std::vector< std::vector<double> > output, int batch_size);

        double mean_square_error(std::vector< std::vector<double> > target, std::vector< std::vector<double> > output, int batch_size);

        void set_activation_functions(std::vector<std::string> _activation_functions);

        void run(std::vector<double> input); //std::vector<double> run(std::vector<double> input);
        
        void train(std::vector<double> input, std::vector<double> expected_output);

        void batch_learn(std::vector< std::vector<double> > inputs, std::vector< std::vector<double> > outputs, int batch_size);

        inline double ret_error(){
            return this->error;
        }

        inline std::vector<double> return_output(){
            return this->outputs;
        }

        int num_inputs;
        int num_hidden_layers;
        int num_hidden_neurons;
        int num_outputs;
        int max_epochs;
        double learning_rate;
        double momentum;
        double desired_error;
        double error;
        
        std::vector<std::string> activationFunctions;
        std::vector<double> inputs;
        std::vector<double> outputs;
    /*
    //private:
        int num_inputs;
        int num_hidden_layers;
        int num_hidden_neurons;
        int num_outputs;
        int max_epochs;
        double learning_rate;
        double momentum;
        double desired_error;
        double error;
        
        std::vector<std::string> activationFunctions;
        std::vector<double> inputs;
        std::vector<double> outputs;
    */        
};
