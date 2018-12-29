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

// std::vector< std::vector<double> > zero_vector(int x, int y);

// //initialize zero vector of same dimensions as input vector
// std::vector< std::vector<double> > zeros_like(std::vector< std::vector<double> > a);

// //return random real number in range [-1, 1]
// double random_real(Layer layer);

// //initialize 2D vector of dimensions x, y with random doubles in [0,1]
// std::vector< std::vector<double> > random_vector(int x, int y, Layer layer);

// //calculate dot product
// double dot(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b) ;

// //matrix multiplication
// std::vector< std::vector<double> > multiply(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b) ;

// std::vector< std::vector<double> > transpose(std::vector< std::vector<double> > a);

// std::vector< std::vector<double> > element_wise_add(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b);

// std::vector< std::vector<double> > element_wise_subtract(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b);


// std::vector< std::vector<double> > element_wise_multiply(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b);

// std::vector< std::vector<double> > scalar_multiply(double s, std::vector< std::vector<double> > a);
//};