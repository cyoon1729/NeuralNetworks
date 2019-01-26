# saNNity

saNNity is a work-in-progress C++ neural network library 

## Namespaces

1. `tensor` : core math & tensor / matrix operations
  
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

  
