# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:30:53 2017

@author: Newton
"""

"""filename: 4.5 Relaxation Method.py"""

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

def Quick_Sort(L):
    """A simple code to order a list with quicksort algorithm."""
    
    if len(L) <= 1:
        return L
    
    else:
        less = []
        equal = []
        bigger = []

        pivot = (L[0] + L[-1] + L[len(L)//2]) / 3
        # pivot should be taken seriously!

        for i in range(len(L)):
            if L[i] < pivot:
                less.append(L[i])
            elif(L[i] == pivot):
                equal.append(L[i])
            else:
                bigger.append(L[i])
        less = Quick_Sort(less)
        bigger = Quick_Sort(bigger)

        return less + equal + bigger

#---------------------------new class---------------------------

class MatrixIterMethods:
    """This class includes three methods which could be used in the 
    solving of linear equations.
    
    Method 1:    Jabobi method
    Method 2:    Gauss-Seidel method
    Method 3:    Relaxation method
    
    All these three methods will return x and the iteration deepth.
    """
    
    def __init__(self,A,b,x0,omega=1):
        """Initialize this class.
        
        A:    coefficient matrix
        b:    where Ax = b
        x0:   the original value of x we choose
        
        These global variables would not change after init operation.
        """
        
        if omega == 1:
            if b.col != 1:
                raise ValueError('Bad inputs, please Check it!')
            
            tmp_b = Matrix(b.row, b.col, b.mainCol)
            for i in range(b.row):
                for j in range(b.col):
                    tmp_b.body[i][j] = b.body[i][j]
            
            for i in range(tmp_b.row):
                tmp_b.body[i][0] = (tmp_b.body[i][0]) / A.body[i][i]
            
            tmp_A = Matrix(A.row, A.col, A.mainCol)
            for i in range(A.row):
                for j in range(A.col):
                    tmp_A.body[i][j] = A.body[i][j]
            
            for i in range(A.row):
                tmp_A = tmp_A.matrixTransform(i,1 / A.body[i][i])
        
            B = Matrix(A.row, A.col, A.mainCol)
            for i in range(B.row):
                for j in range(B.col):
                    if i == j:
                        B.body[i][j] = 0
                    else:
                        B.body[i][j] = -1 * tmp_A.body[i][j]
            ans = (B,tmp_b)
        
        else:
            L = Matrix(A.row, A.col, A.mainCol)
            for i in range(A.row):
                for j in range(A.col):
                    L.body[i][j] = A.body[i][j]
                    if i <= j:
                        L.body[i][j] = 0
            
            U = Matrix(A.row, A.col, A.mainCol)
            for i in range(A.row):
                for j in range(A.col):
                    U.body[i][j] = A.body[i][j]
                    if i >= j:
                        U.body[i][j] = 0
            
            D = Matrix(A.row, A.col, A.mainCol)
            for i in range(A.row):
                for j in range(A.col):
                    D.body[i][j] = A.body[i][j]
                    if i != j:
                        D.body[i][j] = 0
            
            step1 = L.matrixConstMulti(omega)
            step2 = D.matrixAdd(step1)
            step3 = step2.matrixInversion() # mark
            
            step4 = D.matrixConstMulti(-1 * omega)
            step5 = U.matrixConstMulti(-1 * omega)
            step6 = D.matrixAdd(step4)
            step7 = step6.matrixAdd(step5)  # mark
            
            step8 = b.matrixConstMulti(omega)
            
            end_B = step3.matrixMulti(step7)
            end_b = step3.matrixMulti(step8)
            
            ans = (end_B, end_b)
        
        self.B = ans[0]
        self.b = ans[1]
        self.x = Matrix(x0.row, x0.col, x0.mainCol)
        
        for i in range(x0.row):
            for j in range(x0.col):
                self.x.body[i][j] = x0.body[i][j]
    
    def GS_op1(self):
        """Do one time iteration with GS method or Relaxation method.
        
        self.x will be changed after this operation.
        """
        for i in range(self.x.row):
            tmp = 0
            for j in range(self.B.col):
                tmp += self.B.body[i][j] * self.x.body[j][0]
            self.x.body[i][0] = tmp + self.b.body[i][0]
    
    def jacobiMethod(self,accuracy):
        """The iteration would not stop by iter times.
        
        It will stop while the accuracy gets.
        """
        self.__init__(A,b,x0)   # omega uses the default setting
        
        save_tmp = self.x.matrixConstMulti(-1)
        # save the original value for comparing.
        
        step1 = self.B.matrixMulti(self.x)
        step2 = step1.matrixAdd(self.b)
        
        self.x = step2
        # update the value of self.x
        
        p = step2.matrixAdd(save_tmp)
        accu = p.matrixNormVal()
        
        iter_deepth = 1
        
        while(accu > accuracy):
            save_tmp = self.x.matrixConstMulti(-1)
            # save the original value for comparing.
        
            step1 = self.B.matrixMulti(self.x)
            step2 = step1.matrixAdd(self.b)
        
            self.x = step2
            # update the value of self.x
        
            p = step2.matrixAdd(save_tmp)
            accu = p.matrixNormVal()
            
            iter_deepth += 1
        
        return (self.x, iter_deepth)
    
    def gsMethod(self, accuracy):
        """Iteration will stop when the accuracy gets."""
        
        self.__init__(A,b,x0)
        
        save_tmp = self.x.matrixConstMulti(-1)
        # save the original value for comparing.
        
        self.GS_op1()
        # self.x has been changed auto
        
        p = self.x.matrixAdd(save_tmp)
        accu = p.matrixNormVal()
        
        iter_deepth = 1
        
        while(accu > accuracy):
            save_tmp = self.x.matrixConstMulti(-1)
            # save the original value for comparing.
        
            self.GS_op1()
            # self.x has been changed
        
            p = self.x.matrixAdd(save_tmp)
            accu = p.matrixNormVal()
        
            iter_deepth += 1
        
        return(self.x, iter_deepth)
    
    def relaxMethod(self, omega, accuracy):
        """Iteration will stop when the accuracy gets."""
        
        self.__init__(A,b,x0, omega)   # omega uses the default setting
        
        save_tmp = self.x.matrixConstMulti(-1)
        # save the original value for comparing.
        
        step1 = self.B.matrixMulti(self.x)
        step2 = step1.matrixAdd(self.b)
        
        self.x = step2
        # update the value of self.x
        
        p = step2.matrixAdd(save_tmp)
        accu = p.matrixNormVal()
        
        iter_deepth = 1
        
        while(accu > accuracy):
            save_tmp = self.x.matrixConstMulti(-1)
            # save the original value for comparing.
        
            step1 = self.B.matrixMulti(self.x)
            step2 = step1.matrixAdd(self.b)
        
            self.x = step2
            # update the value of self.x
        
            p = step2.matrixAdd(save_tmp)
            accu = p.matrixNormVal()
            
            iter_deepth += 1
        return (self.x, iter_deepth)

def Plot(A, b, x0, e=1e-16):
    """A pure function for ploting.
    
    e is the target accuracy.
    """
    
    ERROR = list()
    tmp = 1
    while tmp > e:
        tmp /= 10
        ERROR.append(tmp)
    
    deepth_1 = list()
    for i in ERROR:
        J = MatrixIterMethods(A, b, x0)
        M_1 = J.relaxMethod(1.0, i)
        # Relaxation factor is 1.0, equal to GSM
        
        deepth_1.append(M_1[1])
    
    deepth_2 = list()
    for i in ERROR:
        M_1 = J.relaxMethod(1.03, i)
        # Relaxation factor is 1.03
        deepth_2.append(M_1[1])
    
    deepth_3 = list()
    for i in ERROR:
        M_1 = J.relaxMethod(1.1,i)
        # Relaxation factor is 1.1
        deepth_3.append(M_1[1])
    
    import math
    for i in range(len(ERROR)):
        ERROR[i] = -1 * math.log10(ERROR[i])
    
    pl.grid()
    pl.title("Same Accuracy: Iteration deepth and Omega's Value",fontsize=16)
    pl.plot(ERROR, deepth_1, 'o-r',label = 'omega is 1.00')
    pl.plot(ERROR, deepth_2, 'o-g',label = 'omega is 1.03')
    pl.plot(ERROR, deepth_3, 'o-b',label = 'omega is 1.10')
    pl.legend()
    pl.xlabel('Level of Accuracy')
    pl.ylabel('Iteration Deepth')
    pl.show()

if __name__ == "__main__":
    
    A = Matrix(6, 6)
    A.body = [[4, -1,  0, -1,  0,  0]\
            ,[-1,  4, -1,  0, -1,  0]\
            ,[ 0, -1,  4,  0,  0, -1]\
            ,[-1,  0,  0,  4, -1,  0]\
            ,[ 0, -1,  0, -1,  4, -1]\
            ,[ 0,  0, -1,  0, -1,  4]]
    
    b = Matrix(6, 1)
    b.body = [[0],[5],[0],[6],[-2],[6]]
    
    x0 = Matrix(6, 1)
    x0.body = [[1],[1],[1],[1],[1],[1]]
    
    Plot(A, b, x0, 1e-20)
    
    A = Matrix(3, 3)
    A.body = [[4, -1,  0]\
            ,[-1,  4, -1]\
            ,[ 0, -1,  4]]

    b = Matrix(3, 1)
    b.body = [[1],[4],[-3]]

    x0 = Matrix(3, 1)
    x0.body = [[1],[1],[1]]

    Plot(A, b, x0, 1e-20)