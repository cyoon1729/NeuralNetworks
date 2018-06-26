#include "./include/ann.h"
#include <iostream>
#include <vector>
#include <string>

int main(){
    //initialize network (num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs)
    ANN test(2,1,2,1);
    
    //set activation functions for hidden and output layers // size = num_hidden_layers + 1
    std::vector<std::string> af; af.resize(1); af = {"sigmoid", "sigmoid"};
    test.set_activation_functions(af);

    //neural netowrk params
    test.set_learning_rate(0.05);
    test.set_momentum(0.05);
    test.set_desired_error(0.001);
    test.set_max_epochs(4000);

    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> expected_output;

    input = {0, 1};
    expected_output = {1};
    test.train(input, expected_output);
    
    std::cout << "\n\n\n\n";

    input = {0, 1};
    output = test.run(input);
    for(double x : output){
        std::cout << "\nfinal output1: " << x << "\n";
    }
    std::cout << test.ret_error() << "\n";
    
    return 0;
}