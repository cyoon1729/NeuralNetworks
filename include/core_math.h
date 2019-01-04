#pragma once

#include <vector>
#include <iostream>

namespace tensor{

/*
 TODO: change Tensor structure to n-dimensional
*/
struct Tensor{
    Tensor(size_t row, size_t col): m(row), n(col) {
        tensor.resize(m * n);
    };
    
    Tensor(){};

    //~Tensor();

    size_t m; // rows
    size_t n; // columns
    std::vector<double> tensor; // row major Tensor
    bool transposed = false;
    void fill_weights(const std::string initializer); // weight initialization using "glorot," "he-et-al," and "kaiming" 
    void fill_weights(const double low, const double high); // weight initialization with uniform distribution given lower and upper bound
    void T(); //transpose
    
};

//helper function for initialization
void initialize(Tensor& empty_tensor, size_t row, size_t col, std::string initializer);

// helper function for initialization
void initialize(Tensor& empty_tensor, size_t row, size_t col, double low, double high);

// copy one Tensor from another Tensor
void copy_Tensor(Tensor& A, Tensor& B);

// element wise addition
const Tensor operator+(const Tensor& lhs, const Tensor& rhs);

// element wise subtraction
const Tensor operator-(const Tensor& lhs, const Tensor& rhs);

// scalar multiplication
const Tensor operator*(const double k, const Tensor& a);

// Tensor multipication (also dot product)
const Tensor operator*(const Tensor& lhs, const Tensor& rhs);

// numpy style Tensor output
std::ostream& operator<<(std::ostream& os, const Tensor& M);

};