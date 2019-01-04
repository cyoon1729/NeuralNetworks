#include "../include/core_math.h"
#include <iostream>

int main(){
    tensor::Tensor A(3, 4);
    A.fill_weights("glorot");
    std::cout << "\n--A--\n";
    std::cout << A;
    std::cout << "\n";

    tensor::Tensor B(3, 4);
    B.fill_weights("glorot");
    std::cout << "\n--B--\n";
    std::cout << B;
    std::cout << "\n";

    tensor::Tensor C(4, 3);
    C.fill_weights("glorot");
    std::cout << "\n--C--\n";
    std::cout << C;
    std::cout << "\n";

    double scalar = 3.0;

    std::cout << "\n--tensor addition--\n";
    std::cout << A + B;
    std::cout << "\n";

    std::cout << "\n--tensor subtraction--\n";
    std::cout << A - B;
    std::cout << "\n";

    std::cout << "\n--tensor multiplication--\n";
    std::cout << A * C;
    std::cout << "\n";

    std::cout << "\n--scalar multiplication--\n";
    std::cout << scalar * A;
    std::cout << "\n";
    
    std::cout << "\n--Transpose--\n";
    A.T();
    std::cout << A;

    return 0;
}
