#pragma once

#include <vector>
#include <iostream>

namespace matrix{

struct Matrix{
    Matrix(size_t row, size_t col): m(row), n(col) {
        matrix.resize(m * n);
    };
    
    Matrix(){};

    //~Matrix();

    size_t m; // rows
    size_t n; // columns
    std::vector<double> matrix; // row major matrix
    bool transposed = false;
    void fill_weights(const std::string initializer); // weight initialization using "glorot," "he-et-al," and "kaiming" 
    void fill_weights(const double low, const double high); // weight initialization with uniform distribution given lower and upper bound
    void T(); //transpose
    
};

//helper function for initialization
void initialize(Matrix& empty_matrix, size_t row, size_t col, std::string initializer);

// helper function for initialization
void initialize(Matrix& empty_matrix, size_t row, size_t col, double low, double high);

// copy one matrix from another matrix
void copy_matrix(Matrix& A, Matrix& B);

// element wise addition
const Matrix operator+(const Matrix& lhs, const Matrix& rhs);

// element wise subtraction
const Matrix operator-(const Matrix& lhs, const Matrix& rhs);

// scalar multiplication
const Matrix operator*(const double k, const Matrix& a);

// matrix multipication (also dot product)
const Matrix operator*(const Matrix& lhs, const Matrix& rhs);

// numpy style matrix output
std::ostream& operator<<(std::ostream& os, const Matrix& M);

};