# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 21:28:46 2017

@author: Newton
"""
"""filename: 2.1 get root.py"""

import math

class root:
    """This class provides some ways to find roots"""
    
    def __init__(self, fun_name, x_left, x_right = None):
        """fun_name represents the name of the function if the equation.
        
        Both left and right ends will be given by x_left and x_right.
        """
        if x_right != None:
            """Only support binary method."""
            
            if fun(x_left) * fun(x_right) < 0:
                self.x_l = x_left
                self.x_r = x_right
                self.fun = fun_name
                self.method = 'binary'
            else:
                raise ValueError("values on x_right and x_left should have opposite sign.")
        else:
            """Only support Aitken method."""
            self.x = x_left
            self.fun_after_convert = fun_name
            self.method = 'Aitken'
    
    def binary(self, e):
        """e = (b - a) / 2"""
        if self.method != 'binary':
            raise ValueError("Method does not support!")
        a = self.x_l
        b = self.x_r
        times = 0
        
        while (abs(b-a)/2) > e:
            if self.fun((a + b)/2) == 0:
                return (a + b)/2
            elif self.fun(a) * self.fun((a + b)/2) < 0:
                b = (a + b)/2
            else:
                a = (a + b)/2
            times += 1
        ans = ((a + b)/2, times)
        return ans
    
    def aitken(self, e):
        """e = x - x0"""
        if self.method != 'Aitken':
            raise ValueError("Method does not support!")
        x0 = self.x
        x1 = self.fun_after_convert(x0)
        x2 = self.fun_after_convert(x1)
        
        x = x0 - (x1-x0)*(x1-x0)/(x2-2*x1+x0)
        times = 1
        
        while abs(x-x0) > e:
            x0 = x
            x1 = self.fun_after_convert(x0)
            x2 = self.fun_after_convert(x1)
            
            x = x0 - (x1-x0)*(x1-x0)/(x2-2*x1+x0)
            
            times += 1
        ans = (x, times)
        return ans
            
if __name__ == "__main__":
    def fun(x):
        return x - math.tan(x)
    
    def fun_after_convert(x):
        return math.tan(x)
    
    c = root(fun, 4.4, 4.6)
    e = 0.00001
    tmp = c.binary(e)
    print('root.binary(x - tan(x) == 0) is ', tmp[0], 'iterations deepth:', tmp[1])
    d = root(fun_after_convert, 4.5)
    tmp = d.aitken(e)
    print('root.aitken(x - tan(x) == 0) is ', tmp[0], 'iterations deepth:', tmp[1])