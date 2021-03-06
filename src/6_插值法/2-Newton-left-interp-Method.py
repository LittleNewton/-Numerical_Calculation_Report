
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:18:27 2017

@author: Newton
"""

"""filename: 2. Newton left-interp Method.py"""

class Interp:
    """This class aims to make the interpolation method combined. each method
    member of this class represents a method of interpolation.
    
    +---------------+-----------+
    |     Name      |   Method  |
    +---------------+-----------+
    |  Newton       |   Newton  |
    |  Lagrange     |   Lagr    |
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
    
    def getDiffTab(self):
        
        n = len(self.x) - 1
        
        ans = [[None for i in range(n)] for i in range(n)]
        
        for i in range(n):          # column 
            for j in range(i, n):   # row
                if i == 0:
                    ans[j][i] = self.y[j+1] - self.y[j]
                else:
                    ans[j][i] = ans[j][i-1] - ans[j-1][i-1]
        return ans
    
    def getDiffQuotientTab(self):
        """Generate a matrix which represents the difference quotient table
        of (x_known, y_known).
        """
        
        equidistant = False     # equidistant is false by defualt
        
        t = self.x[1] - self.x[0]
        for i in range(1, len(self.x)-1):
            if round(t, 1) == round(self.x[i+1] - self.x[i], 1):
                equidistant = True
            else:
                equidistant = False
                break
        
        if equidistant == False:
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
            pass
            return ans
        
        else:
            from math import factorial as fc
            from math import pow as pow
            
            n = len(self.x) - 1
            
            ans = [[None for i in range(n)] for i in range(n)]
            
            diffTab = self.getDiffTab()
            
            for i in range(n):
                low = fc(i+1) * pow(t, i+1)
                up = diffTab[i][i]
                ans[i][i] = up/low
            
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

if __name__ == '__main__':
    
    x = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y = [-0.916291, -0.693147, -0.510826, -0.357765, -0.223144, -0.105361]
    m = [0.41, 0.51, 0.61, 0.71, 0.81, 0.89]
    c = Interp(x, y, m)
    
    ans = Interp(x, y, [0.78])
    print(ans.Newton()[0])
    import math
    print(math.log(0.78))
    
    ans_newton = c.Newton()
    ans_lagr = c.Lagr()
    
    ans_n = list()
    for i in ans_newton:
        ans_n.append(round(i, 4))
    
    ans_l = list()
    for i in ans_lagr:
        ans_l.append(round(i, 4))
    
    print('+----------+----------+---------------+')
    print('|  Method  |    x     |       y       |')
    print('+----------+----------+---------------+')
    print('|   Lagr   |  ',m[0], '  |   ', ans_l[0], '   |')
    print('|          |  ',m[1], '  |   ', ans_l[1], '   |')
    print('|          |  ',m[2], '  |   ', ans_l[2], '   |')
    print('|          |  ',m[3], '  |   ', ans_l[3], '   |')
    print('|          |  ',m[4], '  |   ', ans_l[4], '   |')
    print('|          |  ',m[5], '  |   ', ans_l[5], '   |')
    print('+----------+----------+---------------+')
    print('|  Newton  |  ',m[0], '  |   ', ans_n[0], '   |')
    print('|          |  ',m[1], '  |   ', ans_n[1], '   |')
    print('|          |  ',m[2], '  |   ', ans_n[2], '   |')
    print('|          |  ',m[3], '  |   ', ans_n[3], '   |')
    print('|          |  ',m[4], '  |   ', ans_n[4], '   |')
    print('|          |  ',m[5], '  |   ', ans_n[5], '   |')
    print('+----------+----------+---------------+')
   
    import matplotlib.pyplot as pl
    
    pl.grid()
    pl.title("Simple Visualization",fontsize=16)
    pl.plot(x, y, 'o-g', label = 'y = ln(x)')
    pl.plot(m, ans_l, 'ro', label = 'Lagrange Method')
    pl.plot(m, ans_n, 'b-', label = 'Newton-Left Method')
    pl.legend()
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.show()