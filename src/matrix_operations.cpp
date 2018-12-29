#include "../include/matrix_operations.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <assert.h>

/*
TODO: fix tranpose, matrix multiplication
*/

namespace matrix{

void Matrix::fill_weights(std::string initializer){
    this->matrix.resize(this->m * this->n);
    if(initializer== "zero"){
        for(size_t entry = 0; entry < this->m * this->n; entry++){
            this->matrix[entry] = 0.0;
        }  
    }

    if(initializer == "glorot"){
        double variance = 2.0 / (this->m + this->n);
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-std::sqrt(3.0 * variance), std::sqrt(3.0 * variance));

        for(size_t entry = 0; entry < this->m * this->n; entry++){
            this->matrix[entry] = dist(e2);
        }  
    }
}

void Matrix::fill_weights(double low, double high){
    this->matrix.resize(this->m * this->n);
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(low, high);
    
    for(size_t entry = 0; entry < this->m * this->n; entry++){
        this->matrix[entry] = dist(e2);
    }  
}

void Matrix::T(){
    std::vector<double> original = this->matrix;
    this->matrix.resize(this->n * this->m);
    for(size_t n = 0; n < this->n * this->m; n++){
        size_t i = n/this->m;
        size_t j = n%this->n;
        this->matrix[n] = original[this->m + i];
    }
    std::swap(this->m, this->n);
    this->transposed = !this->transposed;
}

Matrix operator+(const Matrix& lhs, const Matrix& rhs){
    Matrix ret(lhs.m, lhs.n);
    for(size_t entry = 0; entry < lhs.m * lhs.n; entry++){
        ret.matrix[entry] = lhs.matrix[entry] + rhs.matrix[entry];
    }
    return ret;
}

Matrix operator-(const Matrix& lhs, const Matrix& rhs){
    Matrix ret(lhs.m, lhs.n);
    for(size_t entry = 0; entry < lhs.m * lhs.n; entry++){
        ret.matrix[entry] = lhs.matrix[entry] - rhs.matrix[entry];
    }
    return ret;
}

Matrix operator*(const double k, const Matrix& rhs){
    Matrix ret(rhs.m, rhs.n);
    for(size_t entry = 0; entry < rhs.m * rhs.n; entry++){
        ret.matrix[entry] = k * rhs.matrix[entry];
    }
    return ret;
}

Matrix operator*(const Matrix& lhs, const Matrix& rhs){
    assert(lhs.n == rhs.m);
    Matrix ret(lhs.m, rhs.n);
    for(size_t i = 0; i < lhs.m; i++){
        for(size_t j = 0; j < rhs.n; j++){
            for(size_t k = 0; k < lhs.n; k++){
                ret.matrix[i + j * k] += lhs.matrix[k + i * lhs.n] * rhs.matrix[j + k * rhs.n];
            }
        }
    }
    return ret;
}

std::ostream& operator<<(std::ostream& os, const Matrix& M){
    os << "[";
    for(size_t i = 0; i < M.m; i++){
        if(i != 0){
            os << " ";
        }
        os << "[";
        for(size_t j = 0; j < M.n; j++){
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

// std::vector< std::vector<double> > zero_vector(int x, int y){
//     std::vector< std::vector<double> > output;
//     output.resize(x);
//     for(int i = 0; i < x; i++){
//         output[i].resize(y);
//         for(int j = 0; j < y; j++){
//             output[i][j] = 0.0;
//         }
//     }
//     return output;
// }

// std::vector< std::vector<double> > zeros_like(std::vector< std::vector<double> > a){
//     int r_a = a.size(), c_a = a[0].size();
//     std::vector< std::vector<double> > output;
//     output.resize(r_a);
//     for(auto x : output){
//         x.resize(c_a);
//     }
// }

// double random_real(Layer layer){
//     double lower;
//     double upper;
//     double n = (double)layer.neurons.size();

//     upper = std::sqrt(2. / n);
//     lower = 0. - upper;
//     return lower + (rand() / ( RAND_MAX / (upper - lower) ));
// }

// double he_rand(){
//     double lower = -1.;
//     double upper = 1.;
//     std::random_device rd;
//     std::mt19937 e2(rd());
//     std::uniform_real_distribution<> dist(lower, upper);
//     return dist(e2) * std::sqrt(2.0/upper);
// }

// std::vector< std::vector<double> > random_vector(int x, int y, Layer layer){
    
//     std::vector< std::vector<double> > output;
//     output.resize(x);
//     for(int i = 0; i < x; i++){
//         output[i].resize(y);
//         for(int j = 0; j < y; j++){
//             //output[i][j] = random_real(layer);
//             output[i][j] = he_rand();
//         }
//     }

//     return output;
// }

// double dot(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b){
//     int r_a = a.size(), r_b = b.size();
//     int c_a = a[0].size(), c_b = b[0].size();

//     double dot_product = 0;
//     for(int i = 0; i < c_a; i++){
//         dot_product += a[0][i] * b[0][i];
//     }
//     return dot_product;
// }

// std::vector< std::vector<double> > multiply(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b){
//     int r_a = a.size(), r_b = b.size();
//     int c_a = a[0].size(), c_b = b[0].size();
    
//     std::vector< std::vector<double> > output = zero_vector(r_a, c_b);
//     for(int i = 0; i < r_a; i++){
//         for(int j = 0; j < c_b; j++){
//             double sum = 0;
//             for(int k = 0; k < c_a; k++){
//                 sum += a[i][k] * b[k][j];
//             }
//             output[i][j] = sum;
//         }
//     }
//     return output;
// }

// std::vector< std::vector<double> > transpose(std::vector< std::vector<double> > a){
//     int r_a = a.size(), c_a = a[0].size();
//     int d = c_a > r_a ? c_a : r_a;
//     std::vector< std::vector<double> > ap, at;
//     ap.resize(d); 
//     for(int i = 0; i < d; i++){
//         ap[i].resize(d);
//     }

//     for(int i = 0; i < c_a; ++i){
//         for(int j = 0; j < r_a; ++j){
//             ap[i][j] = a[j][i];
//         }
//     }   

//     at.resize(c_a);
//     for(int i = 0; i < c_a; i++){
//         at[i].resize(r_a);
//     }    

//     for(int i = 0; i < c_a; i++){
//         for(int j = 0; j < r_a; j++){
//             at[i][j] = ap[i][j];
//         }
//     }   

//     return at;
// }

// std::vector< std::vector<double> > element_wise_add(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b){

//     std::vector< std::vector<double> > sum;
//     sum = a;
    
//     int r_a = a.size(), r_b = b.size(), c_a = a[0].size(), c_b = b[0].size();

//     for(int i = 0; i < r_a; i++){
//         for(int j = 0; j < c_a; j++){
//             sum[i][j] += b[i][j];
//         }
//     }

//     return sum;
// }

// std::vector< std::vector<double> > element_wise_subtract(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b){
    
//     std::vector< std::vector<double> > diff;
    
//     diff = a;
    
//     int r_a = a.size(), r_b = b.size(), c_a = a[0].size(), c_b = b[0].size();

//     for(int i = 0; i < r_a; i++){
//         for(int j = 0; j < c_a; j++){
//             diff[i][j] -= b[i][j];
//         }
//     }

//     return diff;
// }


// std::vector< std::vector<double> > element_wise_multiply(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b){
//     std::vector< std::vector<double> > product;
//     product = a;

//     int r_a = a.size(), r_b = b.size(), c_a = a[0].size(), c_b = b[0].size();

//     for(int i = 0; i < r_a; i++){
//         for(int j = 0; j < c_a; j++){
//             product[i][j] *= b[i][j];
//         }
//     }

//     return product;
// };

// std::vector< std::vector<double> > scalar_multiply(double a, std::vector< std::vector<double> > b){
//     std::vector< std::vector<double> > out;
//     out = b;

//     int r_b = b.size(), c_b = b[0].size();

//     for(int i = 0; i < r_b; i++){
//         for(int j = 0; j < c_b; j++){
//             out[i][j] *= a;
//         }
//     }

//     return out;
// };

