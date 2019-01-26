#include "../include/core_math.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <assert.h>

namespace tensor{


void Tensor::fill_weights(const std::string initializer){
    this->tensor.resize(this->m * this->n);
    if(initializer== "zero"){
        for(size_t entry = 0; entry < this->m * this->n; ++entry){
            this->tensor[entry] = 0.0;
        }  
    }

    if(initializer == "glorot"){
        double variance = 2.0 / (this->m + this->n);
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-std::sqrt(3.0 * variance), std::sqrt(3.0 * variance));

        for(size_t entry = 0; entry < this->m * this->n; ++entry){
            this->tensor[entry] = dist(e2);
        }  
    }
}

void Tensor::fill_weights(const double low, const double high){
    this->tensor.resize(this->m * this->n);
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(low, high);
    
    for(size_t entry = 0; entry < this->m * this->n; ++entry){
        this->tensor[entry] = dist(e2);
    }  
}

void Tensor::T(){
    std::vector<double> original = this->tensor;
    this->tensor.resize(this->n * this->m);
    for(size_t e = 0; e < this->n * this->m; ++e){
        size_t i = e/this->m;
        size_t j = e%this->m;
        this->tensor[e] = original[this->n * j + i];
    }
    std::swap(this->m, this->n);
    this->transposed = !this->transposed;
}

void copy_tensor(Tensor& A, Tensor& B){
    A = B;
}

void initialize(Tensor& empty_tensor, size_t row, size_t col, std::string initializer){
    empty_tensor.m = row;
    empty_tensor.n = col;
    empty_tensor.fill_weights(initializer);
}

void initialize(Tensor& empty_tensor, size_t row, size_t col, double low, double high){
    empty_tensor.m = row;
    empty_tensor.n = col;
    empty_tensor.fill_weights(low, high);
}

const Tensor operator+(const Tensor& lhs, const Tensor& rhs){
    Tensor ret(lhs.m, lhs.n);
    for(size_t entry = 0; entry < lhs.m * lhs.n; ++entry){
        ret.tensor[entry] = lhs.tensor[entry] + rhs.tensor[entry];
    }
    return ret;
}

const Tensor operator-(const Tensor& lhs, const Tensor& rhs){
    Tensor ret(lhs.m, lhs.n);
    for(size_t entry = 0; entry < lhs.m * lhs.n; ++entry){
        ret.tensor[entry] = lhs.tensor[entry] - rhs.tensor[entry];
    }
    return ret;
}

const Tensor operator*(const double k, const Tensor& rhs){
    Tensor ret(rhs.m, rhs.n);
    for(size_t entry = 0; entry < rhs.m * rhs.n; ++entry){
        ret.tensor[entry] = k * rhs.tensor[entry];
    }
    return ret;
}

const Tensor operator*(const Tensor& lhs, const Tensor& rhs){
    assert(lhs.n == rhs.m);
    Tensor ret(lhs.m, rhs.n);
    for(size_t i = 0; i < lhs.m; ++i){
        for(size_t j = 0; j < rhs.n; ++j){
            for(size_t k = 0; k < lhs.n; ++k){
                ret.tensor[j + i * rhs.n] += lhs.tensor[k + i * lhs.n] * rhs.tensor[j + k * rhs.n];
            }
        }
    }
    return ret;
}

/*const Tensor operator=(const Tensor& t){
    return t;
}*/


std::ostream& operator<<(std::ostream& os, const Tensor& M){
    os << "[";
    for(size_t i = 0; i < M.m; ++i){
        if(i != 0){
            os << " ";
        }
        os << "[";
        for(size_t j = 0; j < M.n; ++j){
            os << M.tensor[j + M.n * i];
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


