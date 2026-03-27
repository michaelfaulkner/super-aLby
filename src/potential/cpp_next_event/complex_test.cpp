#include <iostream>
#include <random>
#include <cmath>
#include <complex>
#include <stdio.h>
#include <algorithm>
#include <array>
#include <vector>

int main(){
    std::vector<std::complex<double>> vect;
    
    double y = -4.5;
    std::complex<double> y_c = y;

    std::cout << y_c << std::endl;

    std::complex<double> x = std::sqrt(std::complex<double> (y));

    std::cout << x << std::endl;

    double z = -2.22;
    double a = -4.42;


    std::cout << std::abs(z-a) << std::endl;
    std::cout << std::abs(a-z) << std::endl;


return 0;
}