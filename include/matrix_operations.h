#pragma once

#include <vector>
#include <iostream>

namespace matrix{

struct Matrix{
    Matrix(size_t row, size_t col): m(row), n(col) {
        matrix.resize(m * n);
    };
    
    Matrix(){}

    size_t m; // rows
    size_t n; // columns
    std::vector<double> matrix; // row major matrix
    bool transposed = false;
    void fill_weights(const std::string initializer); // weight initialization using "glorot," "he-et-al," and "kaiming" 
    void fill_weights(const double low, const double high); // weight initialization with uniform distribution given lower and upper bound
    void T(); //transpose
    friend std::ostream& operator<<(std::ostream& os, const Matrix& M);
};

// element wise addition
const Matrix operator+(const Matrix& lhs, const Matrix& rhs);

// element wise subtraction
const Matrix operator-(const Matrix& lhs, const Matrix& rhs);

// scalar multiplication
const Matrix operator*(const double k, const Matrix& a);

// matrix multipication (also dot product)
const Matrix operator*(const Matrix& lhs, const Matrix& rhs);

};
