# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:25:09 2017

@author: Newton
"""

"""filename: 1. Numerical Integration.py"""

class Interp:
    """This class aims to make the interpolation method combined. each method
    member of this class represents a method of interpolation.
    
    +---------------+-----------+
    |     Name      |   Method  |
    +---------------+-----------+
    |    Newton     |   Newton  |
    |    Lagrange   |   Lagr    |
    +---------------+-----------+
    
    """
    
    def __init__(self, x_known, y_known, x_unknown):
        """The (x, y) points we have already known is essential to the 
        interpolation."""
        self.x = x_known    # x_known is a list
        self.y = y_known    # y_known is a list
        self.ux = x_unknown # need to be computed
        
        if len(self.x) != len(self.y):
            raise ValueError("Bad input, len(x) should equal to len(y)")
    
    def getDiffQuotientTab(self):
        """Generate a matrix which represents the difference quotient table
        of (x_known, y_known).
        """
        n = len(self.x) - 1
        
        ans = [[None for i in range(n)] for i in range(n)]
        # initialize it with default setting None.
        
        for i in range(n):          # column
            for j in range(i, n):   # row
                if i == 0:
                    ans[j][i] = (self.y[j+1] - self.y[j]) \
                    / (self.x[j+1] - self.x[j])
                else:
                    ans[j][i] = (ans[j][i-1] - ans[j-1][i-1]) \
                    / (self.x[j+1] - self.x[j-1])
        
        return ans
    
    def Newton(self):
        """Need self.getDiffQuotientTab method.
        
        
        """
        step0 = self.getDiffQuotientTab()
        step1 = list()
        for i in range(len(self.x)-1):
            step1.append(step0[i][i])
        
        ans = [0 for i in range(len(self.ux))]
        
        for i in range(len(self.ux)):       # generate a list of y we needed
            for j in range(len(self.x)):    # a long polynomial function
                if j == 0:
                    ans[i] += self.y[j]
                else:
                    tmp = 1
                    for k in range(j):
                        tmp *= (self.ux[i] - self.x[k])
                    tmp *= step1[j-1]
                    
                    ans[i] += tmp
        
        return ans
    
    def Lagr(self):
        n = len(self.x)
        m = len(self.ux)
        
        ans = []
        
        for i in range(m):          # all the x unknown
            s = 0
            for k in range(n):      # sum
                p = 1
                for j in range(n):  # multi
                    if j != k:
                        p = p * ((self.ux[i] - self.x[j]) / (self.x[k] - self.x[j]))
                s = s + p * self.y[k]
            ans.append(s)
        return ans

class Integrate:
    """This class aims to compute the numerical integration of a function, or
    just some discrete points.
    """
    def __init__(self, x_min, x_max, function_name=None, step=None):
        """
        function_name:        the function needs to be calculated
        x_min:                the beginning of the range of x
        x_max:                the end of the range of x
        
        If the input is in this format, we can generate a list which represents
        the value of the function under step. 
        
        if the number of inputs is 2, it means two list, x and y. 
        """
        if function_name == None and step == None:
            self.x = x_min
            self.y = x_max
        else:
            from numpy import arange
            self.x = list(arange(x_min, x_max, step))      # this is a list
            self.y = list()
            for i in range(len(self.x)):
                self.y.append(function_name(self.x[i]))
    
    def Linear(self):
        ans = 0
        for i in range(len(self.x)-1):
            ans += (self.y[i] + self.y[i+1]) * (self.x[i+1] - self.x[i]) / 2
        return ans
    
    def Simpson(self):
        ans = 0
        point_m = list()
        for i in range(1, len(self.y), 2):
            point_m.append(self.y[i])
        s_m = sum(point_m)
        
        point_double = list()
        for i in range(2, len(self.y), 2):
            point_double.append(self.y[i])
        s_d = sum(point_double)
        
        ans = (self.y[0] + self.y[-1] + 4 * s_m + 2 * s_d) * (self.x[1] - self.x[0]) / 3
        return ans

if __name__ == '__main__':
    from math import sin as sin
    
    def func(x):
        y = x / (4 + x*x)
        return y
    
    c = Integrate(0, 1, func, .1/.8)
    print('+-----------+--------------------+')
    print('|  Method   |  Value             |')
    print('+-----------+--------------------+')
    print('|  Linear   | ', c.Linear(), '  |')
    print('+-----------+--------------------+')
    print('|  Simpson  | ', c.Simpson(), '    |')
    print('+-----------+--------------------+')