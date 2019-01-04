#include "../include/network.h"
#include "../include/layer.h"
#include <vector>
#include <iostream>

int main(){
    /* neuralnet::Layer input(3, 4, "linear", "glorot");
    neuralnet::Layer hidden(4, 1, "linear", "glorot");
    neuralnet::Layer output(1, "sigmoid");
    std::vector<neuralnet::Layer> layers = {input, hidden, output};
    neuralnet::MLP network(layers); */

    std::vector<neuralnet::Layer> layers = {
        neuralnet::Layer(3, 4, "linear", "glorot"),
        neuralnet::Layer(4, 1, "linear", "glorot"),
        neuralnet::Layer(1, "sigmoid")
    };
    neuralnet::MLP network(layers); 
    
    std::vector<double> data = {0.3, 0.3, 1.0};
    std::cout << network.forward(data);

    return 0;
}