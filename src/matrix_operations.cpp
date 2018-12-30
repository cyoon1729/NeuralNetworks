#include "../include/matrix_operations.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <assert.h>

namespace matrix{

void Matrix::fill_weights(const std::string initializer){
    this->matrix.resize(this->m * this->n);
    if(initializer== "zero"){
        for(size_t entry = 0; entry < this->m * this->n; ++entry){
            this->matrix[entry] = 0.0;
        }  
    }

    if(initializer == "glorot"){
        double variance = 2.0 / (this->m + this->n);
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-std::sqrt(3.0 * variance), std::sqrt(3.0 * variance));

        for(size_t entry = 0; entry < this->m * this->n; ++entry){
            this->matrix[entry] = dist(e2);
        }  
    }
}

void Matrix::fill_weights(const double low, const double high){
    this->matrix.resize(this->m * this->n);
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(low, high);
    
    for(size_t entry = 0; entry < this->m * this->n; ++entry){
        this->matrix[entry] = dist(e2);
    }  
}

void Matrix::T(){
    std::vector<double> original = this->matrix;
    this->matrix.resize(this->n * this->m);
    for(size_t e = 0; e < this->n * this->m; ++e){
        size_t i = e/this->m;
        size_t j = e%this->m;
        this->matrix[e] = original[this->n * j + i];
    }
    std::swap(this->m, this->n);
    this->transposed = !this->transposed;
}

const Matrix operator+(const Matrix& lhs, const Matrix& rhs){
    Matrix ret(lhs.m, lhs.n);
    for(size_t entry = 0; entry < lhs.m * lhs.n; ++entry){
        ret.matrix[entry] = lhs.matrix[entry] + rhs.matrix[entry];
    }
    return ret;
}

const Matrix operator-(const Matrix& lhs, const Matrix& rhs){
    Matrix ret(lhs.m, lhs.n);
    for(size_t entry = 0; entry < lhs.m * lhs.n; ++entry){
        ret.matrix[entry] = lhs.matrix[entry] - rhs.matrix[entry];
    }
    return ret;
}

const Matrix operator*(const double k, const Matrix& rhs){
    Matrix ret(rhs.m, rhs.n);
    for(size_t entry = 0; entry < rhs.m * rhs.n; ++entry){
        ret.matrix[entry] = k * rhs.matrix[entry];
    }
    return ret;
}

const Matrix operator*(const Matrix& lhs, const Matrix& rhs){
    assert(lhs.n == rhs.m);
    Matrix ret(lhs.m, rhs.n);
    for(size_t i = 0; i < lhs.m; ++i){
        for(size_t j = 0; j < rhs.n; ++j){
            for(size_t k = 0; k < lhs.n; ++k){
                ret.matrix[j + i * rhs.n] += lhs.matrix[k + i * lhs.n] * rhs.matrix[j + k * rhs.n];
            }
        }
    }
    return ret;
}

std::ostream& operator<<(std::ostream& os, const Matrix& M){
    os << "[";
    for(size_t i = 0; i < M.m; ++i){
        if(i != 0){
            os << " ";
        }
        os << "[";
        for(size_t j = 0; j < M.n; ++j){
            os << M.matrix[j + M.n * i];
            if(j == M.n - 1){
                os << "]";
            }else{
                os << ", ";
            }
        }
        if(i == M.m - 1){
            os << "]\n ";
        }else{
            os << "\n";
        }
    }
}
}


