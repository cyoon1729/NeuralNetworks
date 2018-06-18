#pragma once

#include <vector>
#include <string>



    
struct Neuron{
    double input;
    double output;
};

class Layer{
    public:
        Layer(int _num_neurons, std::string layer_id);

        inline std::string return_layer_identity(){
            return this->layer_identity;
        
        } 
        void initialize_weights(int r, int c);
        
        void set_activation_function(std::string _activation_funct);
        
        inline std::string ret_activation_function(){
            return this->activation_funct;
        }

        double forward_activation_function(double x);

        double backward_activation_function(double x);

        void set_neuron_inputs(std::vector<double> in);

        std::vector<double> get_neuron_inputs();
        
        void set_neuron_outputs();

        std::vector<double> get_neuron_outputs();



        inline std::vector<Neuron> return_neurons(){
            return this->neurons;
        }
        
        std::vector< std::vector<double> > forward();

        void backward();

    private:
        std::vector<Neuron> neurons;
        std::string activation_funct;
        std::string layer_identity;
        std::vector< std::vector<double> > weights;

        int num_neurons;


};
