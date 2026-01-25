#include <iostream>
#include <cmath>
#include <format>

float future_value(float present_val, float rate, int years){
    return present_val * std::exp(rate * years);
}

float present_value(float future_val, float rate, int years){
    return future_val * std::exp(- rate * years);
}

// example usage
int main(){
    float pv_input = 100;
    float fv_input = 200;
    float rate = 0.05;
    int years = 20;
    
    float fv = future_value(pv_input, rate, years);
    float pv = present_value(fv_input, rate, years);
    
    std::cout << std::format("The future value of ${} continuously compounding at a rate of {} per annum for {} years is: ${}", pv_input, rate, years, fv) << std::endl;
    
    std::cout << std::format("The present value of ${} continuously discounted at a rate of {} per annum for {} years is: ${}", fv_input, rate, years, pv) << std::endl;
    
    return 0;
}
