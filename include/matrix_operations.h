#pragma once

#include <vector>

//initialize zero vector of dimensions x, y
std::vector< std::vector<double> > zero_vector(int x, int y);

//initialize zero vector of same dimensions as input vector
std::vector< std::vector<double> > zeros_like(std::vector< std::vector<double> > a);

//initialize 2D vector of dimensions x, y with random doubles in [0,1]
std::vector< std::vector<double> > random_vector(int x, int y);

//calculate dot product
double dot(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b) ;

//matrix multiplication
std::vector< std::vector<double> > multiply(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b) ;

//transpose
std::vector< std::vector<double> > transpose(std::vector< std::vector<double> > a);


//element-wise add a and b (on a)
void vector_add(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b);

