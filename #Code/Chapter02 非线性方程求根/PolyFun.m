% filename: PolyFun.m

% This function could compute the functions with the following form

% y = a_0 * x^0 + a_1 * x^1 + ... + a_n * x^n

% where x is an constant produced by the rand built-in function, and the 
% matrix [a_0, a_1, ..., a_n] should be input as an arguement.

% Try to make the computing complexitity as small as possible.

function y = PolyFun(coe,x)
    n = length(coe);
    % a is the size of the matrix coe
    if(n == 1)
        y = coe;
    else
        y = coe(1) + x * PolyFun(coe(2:length(coe)),x);
    end
end