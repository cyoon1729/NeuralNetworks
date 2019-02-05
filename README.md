# saNNity

saNNity is a work-in-progress C++ neural network library 

For now: 
## Namespaces

### 1. `tensor` : core math & tensor / matrix operations
  
  ```c++
  tensor::Tensor T(3, 4) // an empty 3 by 4 rank 2 tensor
  T.fill_weights("glorot") // fill entries with glorot initializiation
  std::cout << T; // prints entries of tensor T
  
  // Operations
  tensor::Tensor A(3, 3); A.fill_weights("glorot"); 
  tensor::Tensor B(3, 3); B.fill_weights("glorot");
  double sclr = 2.0;
  // 1. Addition
  std::cout << A + B;
  
  // 2. Subtraction
  std::cout << A - B;
  
  // 3. Matrix multiplication
  std::cout << A * B;
  
  // 4. scalar multiplication
  std::cout << sclr * A;
  
  // 5. Transpose
  A.T();
  std::cout << A;
  ```

 ### 2. `neuralnet` : neural network class
 - `class Layer`: Bass layer 
 ```c++
 /*
  * Initialize layer with weights with initializer
  * params: (size_t fan_in, size_t fan_out, std::string activation_function, std::string weight_initializer)
  */
 neuralnet::Layer one(3, 4, "linear", "glorot"); // params:
 
 /*
  * Initialize layer with weights in range [low, high]
  * params: (size_t fan_in, size_t fan_out, std::string activation_function, double low, double high)
  */
 neuralnet::Layer two(4, 1, "linear", -0.001, 0.001 ); 
 

 /*
  * Initialize (output) layer 
  * params: (size_t fan_in, std::string activation_function, ...)
  */
 neuralnet::Layer two(4, "linear", "glorot"); 
 neuralnet::Layer two(4, "linear", -0.001, 0.001 ); 
 
 ```
 
 - `class MLP`: Standard multiplayer preceptron
 ```c++
    /*
     * Param: std::vector<neuralnet::layer> layers
     */
    std::vector<neuralnet::Layer> layers = {
        neuralnet::Layer(3, 4, "linear", "glorot"),
        neuralnet::Layer(4, 1, "linear", "glorot"),
        neuralnet::Layer(1, "sigmoid")
    };
    neuralnet::MLP network(layers); 
    
    // get neural network outputs given input data
    std::vector<double> data = {0.3, 0.3, 1.0};
    std::cout << network.forward(data);

 ```
 
 ### 3. `optimizer`: Optimization algorithms for updating neural network weights  
 - Vanilla Gradient Descent
```c++

    std::vector<neuralnet::Layer> layers = {
        neuralnet::Layer(3, 4, "linear", "glorot"),
        neuralnet::Layer(4, 1, "linear", "glorot"),
        neuralnet::Layer(1, "sigmoid")
    };
    neuralnet::MLP network(layers); 
    
    std::vector<double> data = {0.3, 0.3, 1.0};
    std::cout << network.forward(data);

    std::vector< std::vector<double> > dataset = {{0.3, 0.3, 1.0}};
    std::vector< std::vector<double> > correct = {{0.3}};
    optimizer::VGD optim(network, "MSE");
    optim.step(network, dataset, correct, 1);
```
