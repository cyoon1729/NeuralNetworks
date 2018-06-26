#pragma once

#include <vector>
#include <string>
    
struct Neuron{
    // z = W^T * x 
    double z;
    // activation values  a = g(z)
    double a;
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

        void back_propagate(std::vector< std::vector<double> > p_delta);
        
        inline void set_delta_weights(std::vector< std::vector<double> > dw){
            this->delta_weights = dw;
        }

        inline std::vector< std::vector<double> > get_delta_weights(){
            return this->delta_weights;
        }

        void update_weights();

    private:
        std::vector<Neuron> neurons;
        std::string activation_funct;
        std::string layer_identity;
        std::vector< std::vector<double> > weights;
        std::vector< std::vector<double> > delta_weights;
        std::vector< std::vector<double> > delta;

        int num_neurons;
        double learning_rate;
        double momentum;
        double regularization;

};

