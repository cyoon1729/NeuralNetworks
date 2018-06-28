#pragma once

#include <vector>
#include "ann.h"
#include "layer.h"

//initialize zero vector of dimensions x, y
std::vector< std::vector<double> > zero_vector(int x, int y);

//initialize zero vector of same dimensions as input vector
std::vector< std::vector<double> > zeros_like(std::vector< std::vector<double> > a);

//return random real number in range [-1, 1]
double random_real(Layer layer);

//initialize 2D vector of dimensions x, y with random doubles in [0,1]
std::vector< std::vector<double> > random_vector(int x, int y, Layer layer);

//calculate dot product
double dot(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b) ;

//matrix multiplication
std::vector< std::vector<double> > multiply(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b) ;

std::vector< std::vector<double> > transpose(std::vector< std::vector<double> > a);

std::vector< std::vector<double> > element_wise_add(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b);

std::vector< std::vector<double> > element_wise_subtract(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b);


std::vector< std::vector<double> > element_wise_multiply(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b);

std::vector< std::vector<double> > scalar_multiply(double s, std::vector< std::vector<double> > a);
