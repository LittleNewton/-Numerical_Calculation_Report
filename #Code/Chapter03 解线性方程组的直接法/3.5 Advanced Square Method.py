# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:54:13 2017

@author: Newton
@fielname: 3.5 Advanced Square Method
"""

import matplotlib.pyplot as pl
import numpy as np

class Matrix:
    """Abatrct class representing a Matrix.
    
    The internal structure is a two dimension list.
    """
    def __init__(self,m,n,mainCol = None):
        """
        self.row:            the row of the Matrix
        self.col:            the collumn of the Matrix
        self.CheckedRow:     if one row has been checked,
                             it should not be checked again
        self.body:           the internal storing structure
        self.mainCol:        the coefficient matrix
        """
        self.row = m
        self.col = n
        self.CheckedRow = set(range(self.row))
        self.body = [[0 for i in range(n)] for i in range(m)]
        self.mainCol = mainCol

    def getVal(self):
        """Giving value to each element of the matrix.
        
        Overwrite the original value zeros.
        """
        for i in range(self.row):
            for j in range(self.col):
                self.body[i][j] = input()

    def valid(self,e,kind = None):
        """If these two matrices are not in the right form, return false,
        else return true.
        
        It is useful in the next functions.
        """
        if kind == 'multi':
            # SELF * E
            return self.col == e.row
        
        if self.row != e.row or self.col != e.col:
            return False
        else:
            return True

    def matrixAdd(self,e):
        """One matrix add to another and generate a new one.
        
        The original one would not change.
        """
        self.valid(e)
        # if e and self are not in the same form, return false.
        
        ans = Matrix(self.row,self.col,self.mainCol)
        
        for i in range(self.row):
            for j in range(self.col):
                ans.body[i][j] = self.body[i][j] + e.body[i][j]

        return ans

    def matrixConstMulti(self,const):
        """A constant number multiple a matrix."""
        
        ans = Matrix(self.row,self.col,self.mainCol)
        
        for i in range(self.row):
            for j in range(self.col):
                ans.body[i][j] = self.body[i][j] * const

        return ans

    def matrixMulti(self,e):
        """Return the multiplication of two matrices.
        
        Attention! ans = self * e, not e * self.
        """
        
        self.valid(e,'multi')
        # if e and self could not make multiplication, return false
        
        ans = Matrix(self.row, e.col, e.col)

        for i in range(self.row):
            for j in range(e.col):
                tmp = 0
                for k in range(self.col):
                    tmp += self.body[i][k] * e.body[k][j]
                ans.body[i][j] = tmp
        
        return ans

    def matrixTransform(self,target_row,source_row,times=None):
        """
        case 1:    two coefficient, make coe(1) row times coe(2)
        case 2:    exchange the two rows with 'exchange' reminding
        case 3:    coe(1) rows add the coe(3) times of coe(2) row.
        
        """
        
        ans = Matrix(self.row, self.col, self.mainCol)
        for i in range(self.row):   # deep copy
            for j in range(self.col):
                ans.body[i][j] = self.body[i][j]
        
        if times == None:    # special case of matrixTransform(TarRow,times)
            times_tmp = source_row
            for j in range(self.col):
                ans.body[target_row][j] = \
                ans.body[target_row][j] * times_tmp

        elif times == 'exchange':
            for i in range(self.col):
                ans.body[target_row][i], ans.body[source_row][i] \
                    = ans.body[source_row][i], ans.body[target_row][i]
        
        else:
            for i in range(self.col):
                ans.body[target_row][i] += \
                times * ans.body[source_row][i]
        
        return ans

    def matrixTranspose(self):
        """Generate a new matrix which is the transpose of the old one."""
        
        ans = Matrix(self.col,self.row,self.row)
        # main column of ans is meaningless
        
        for i in range(self.row):
            for j in range(self.col):
                ans.body[j][i] = self.body[i][j]
        return ans

    def matrixNormVal(self, kind = 2):
        """
        Return the Normal value of this matrix.
        """
        
        if kind == 'inf':
            """NorVal is the maximum value among the sums of every row. """
            sums = list()
            tmp = 0
            for i in range(self.row):
                for j in range(self.col):
                    tmp += self.body[i][j]
                sums.append(tmp)
                tmp = 0
        
            sums = Quick_Sort(sums)     # quick sort algorithm
            return sums[-1]
    
        elif kind == 1:
            """NorVal is the maximum value among the sums of every column. """
            sums = list()
            tmp = 0
            for i in range(self.col):
                for j in range(self.row):
                    tmp += self.body[j][i]
                sums.append(tmp)
                tmp = 0
        
            sums = Quick_Sort(sums)
            return sums[-1]
    
        elif kind == 2:
            """
            The solving of eig is a little difficult for me,
            I will change the code after learning more later.
        
            Now I use the numpy package to solve it.
            """
            import numpy as np
        
            MUL = self.matrixMulti(self.matrixTranspose())
            tmp = np.array(MUL.body)

            a,b = np.linalg.eig(tmp)

            return np.sqrt(max(a))
        
        else:
            raise ValueError("""Bad input!\n""")
    
    def matrixInversion(self):
        """Generate a new matrix which is the old one's inversion."""
        
        Max = 0                         # initialize the variable
        position_row = 0                # the row number of the maximum value
        position_col = 0

        
        ans = Matrix(self.row, self.col, self.mainCol)
        for i in range(self.row):
            for j in range(self.col):
                ans.body[i][j] = self.body[i][j]

        eyes = Matrix(self.row, self.col, self.mainCol)
        for i in range(eyes.row):
            eyes.body[i][i] = 1
            ans.body[i] += eyes.body[i]
        
        ans.mainCol = ans.col
        ans.col *= 2
        
        for i in range(ans.row):
            for j in ans.CheckedRow:
                for k in range(ans.mainCol):  # not in all the columns
                    if abs(Max) <= abs(ans.body[j][k]):
                        Max = ans.body[j][k]
                        position_row = j
                        position_col = k
        
            ans = ans.matrixTransform(position_row, 1 / Max)
            
            ans.CheckedRow.remove(position_row)
            
            for j in range(ans.row):
                if j != position_row:
                    ans = ans.matrixTransform\
                    (j, position_row,-1 * ans.body[j][position_col])
            
            Max = 0
            position_row = 0
            position_col = 0
        
        begin = 0
        for j in range(ans.mainCol):
            for i in range(ans.row):
                if ans.body[i][j] == 1:
                    ans = ans.matrixTransform(begin,i,'exchange')
                    begin += 1
        
        new = Matrix(self.row, self.row, None)
        for i in range(new.row):
            for j in range(new.col):
                new.body[i][j] = ans.body[i][j+ans.row]
        ans = new
        return ans

def square(m):
    for i in range(2, m.row+1):
        for j in range(i):
            
        
if __name__ == "__main__":
    A = Matrix(4, 5, 4)
    A.body = [[ 4,  1, -1,  0,  7],\
              [ 1,  3, -1,  0,  8],\
              [-1, -1,  5,  2, -4],\
              [ 0,  0,  2,  4,  6]]
    
    
    
    
    
    
    