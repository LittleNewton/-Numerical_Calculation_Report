clc
a = input('Please input the coefficient matrix:\n');
for i = 1:10
    x(i) = randi([1,50]);
    y(i) = PolyFun(a,x(i));
end
x
Sorted = QuickSort(y)