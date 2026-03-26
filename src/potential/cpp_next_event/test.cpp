#include<iostream>
#include<stdio.h>
#include<vector>
#include <algorithm>


int main(){
    std::vector<double> test = {10.0, 1.0, 55.0};

    for (double i : test){
    std::cout << i << ", ";
    }
    std::cout << std::endl;

    std::sort(test.begin(), test.end());

    for (double i : test){
    std::cout << i<< ", ";
    }
    std::cout << std::endl;

    return 0;
}