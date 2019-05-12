% filename: QuickSort.m
% Sort the vector x, an old but efficient function,
% especially in solving the random sequence sorting problem.

function y = QuickSort(x)
    A = [];
    B = [];
    C = [];
    if(length(x) <= 1)
        y = x;
        return
    else
        a = floor(length(x)/2);
        for i = 1:length(x)
            if(x(i) < x(a))
                A = [A,x(i)];
            elseif(x(i) == x(a))
                B = [B,x(i)];
            else
                C = [C,x(i)];
            end
        end
    end
    A = QuickSort(A);
    C = QuickSort(C);
    y = [A,B,C];
end