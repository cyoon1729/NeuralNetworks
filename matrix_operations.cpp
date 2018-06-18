#include "matrix_operations.h"
#include <random>
#include <iostream>
#include <vector>
//#include <assert.h>

std::vector< std::vector<double> > zero_vector(int x, int y){
    std::vector< std::vector<double> > output;
    output.resize(x);
    for(int i = 0; i < x; i++){
        output[i].resize(y);
        for(int j = 0; j < y; j++){
            output[i][j] = 0.0;
        }
    }
    return output;
}

std::vector< std::vector<double> > zeros_like(std::vector< std::vector<double> > a){
    int r_a = a.size(), c_a = a[0].size();
    std::vector< std::vector<double> > output;
    output.resize(r_a);
    for(auto x : output){
        x.resize(c_a);
    }
}

std::vector< std::vector<double> > random_vector(int x, int y){
    std::default_random_engine gen;    
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    std::vector< std::vector<double> > output;
    
    output.resize(x);
    for(int i = 0; i < x; i++){
        output[i].resize(y);
        for(int j = 0; j < y; j++){
            output[i][j] = distribution(gen);
        }
    }

    return output;
}

double dot(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b){
    //assume a, d are row vectors
    int r_a = a.size(), r_b = b.size();
    int c_a = a[0].size(), c_b = b[0].size();
    //assert(c_a == c_b);

    double dot_product = 0;
    for(int i = 0; i < c_a; i++){
        dot_product += a[0][i] * b[0][i];
    }
    return dot_product;
}

std::vector< std::vector<double> > multiply(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b){
    int r_a = a.size(), r_b = b.size();
    int c_a = a[0].size(), c_b = b[0].size();
    //assert(c_a == r_b);
    
    std::vector< std::vector<double> > output = zero_vector(r_a, c_b);
    for(int i = 0; i < r_a; i++){
        for(int j = 0; j < c_b; j++){
            double sum = 0;
            for(int k = 0; k < c_a; k++){
                sum += a[i][k] * b[k][j];
            }
            output[i][j] = sum;
        }
    }
    return output;
}

std::vector< std::vector<double> > transpose(std::vector< std::vector<double> > a){
    int r_a = a.size(), c_a = a[0].size();
    int d = c_a > r_a ? c_a : r_a;
    std::vector< std::vector<double> > ap, at;
    ap.resize(d); 
    for(int i = 0; i < d; i++){
        ap[i].resize(d);
    }

    for(int i = 0; i < c_a; ++i){
        for(int j = 0; j < r_a; ++j){
            ap[i][j] = a[j][i];
        }
    }   

    at.resize(c_a);
    for(int i = 0; i < c_a; i++){
        at[i].resize(r_a);
    }    

    for(int i = 0; i < c_a; i++){
        for(int j = 0; j < r_a; j++){
            at[i][j] = ap[i][j];
        }
    }   

    return at;
}


void vector_add(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b){
    //assume row vectors, and a, b are of same dimension
    int c_a = a[0].size(), c_b = b[0].size();
    for(int i = 0; i < c_a; i++){
        a[0][i] += b[0][i];
    }
}


