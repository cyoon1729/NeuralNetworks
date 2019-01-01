#include "../include/layer.h"
#include <iostream>
#include <vector>

int main(){
    neuralnet::Layer one(3, 4, "linear", "glorot");
    neuralnet::Layer two(4, 3, "linear", "glorot");
    std::vector<double> input = {0.3, 0.3, 1.0};
    one.feed(input);
    std::cout << "\n";
    one.forward(two);

    // std::cout << one.get_weights();
    // std::cout << "\n";
    // std::cout << two.get_weights();
    
    return 0;
}