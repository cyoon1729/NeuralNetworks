#pragma once

#include <vector>
#include <iostream>

namespace matrix{

struct Matrix{
    Matrix(size_t row, size_t col){
        m = row;
        n = col;
        matrix.resize(row * col);
    }

    Matrix(){}

    size_t m; // rows
    size_t n; // columns
    std::vector<double> matrix; // row major matrix
    bool transposed = false;
    void fill_weights(std::string initializer); // weight initialization using "glorot," "he-et-al," and "kaiming" 
    void fill_weights(double low, double high); // weight initialization with uniform distribution given lower and upper bound
    void T(); //transpose
    friend std::ostream& operator<<(std::ostream& os, const Matrix& M);
};

// element wise addition
Matrix operator+(const Matrix& lhs, const Matrix& rhs);

// element wise subtraction
Matrix operator-(const Matrix& lhs, const Matrix& rhs);

// scalar multiplication
Matrix operator*(const double k, const Matrix& a);

// matrix multipication (also dot product)
Matrix operator*(const Matrix& lhs, const Matrix& rhs);

};
