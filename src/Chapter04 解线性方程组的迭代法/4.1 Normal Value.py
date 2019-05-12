"""filename: 4.1 Normal Value.py"""

"""This is a universally usable code to get different
kind of normal value of an matrix. And the matrix is 
a instance of Class Matrix.

Word is clean, this is the code!
"""

class Matrix:
    """Abatrct class representing a Matrix.
    
    The internal structure is a two dimension list.
    """
    def __init__(self,m,n,mainCol):
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
                self.body[i][j] = int(input())

    def valid(self,e,kind = None):
        """If these two matrix are not in the same form, return false,
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
        """A methon does not used in the pivot PCA algorithm.
        
        Maybe it will be used in other programs.
        """
        self.valid(e)
        tmp = Matrix(self.row,self.col,self.mainCol)
        for i in range(self.row):   # deep copy
            for j in range(self.col):
                tmp.body[i][j] = self.body[i][j]

        for i in range(self.row):
            for j in range(self.col):
                tmp.body[i][j] += e.body[i][j]
        return tmp

    def matrixConstMulti(self,const):
        """A constant number multiple a matrix."""
        tmp = Matrix(self.row,self.col,self.mainCol)
        for i in range(self.row):   # deep copy
            for j in range(self.col):
                tmp.body[i][j] = self.body[i][j]

        for i in range(self.row):
            for j in range(self.col):
                tmp.body[i][j] *= const
        return tmp

    def matrixMulti(self,e):
        """Return the multiplication of two matrix."""
        self.valid(e,'multi')

        ans = Matrix(self.row,e.col,e.col)  # e.col has no meaning

        for i  in range(self.row):
            for j in range(e.col):
                tmp = 0
                for k in range(self.col):
                    tmp += self.body[i][k] * e.body[k][j]
                ans.body[i][j] = tmp
        
        return ans

    def matrixTransform(self,target_row_Number,source_row_Number,times=None):
        """There is a big problem, every decimal number we see is stored in the
        RAM with the binary platform.
        
        I can import the decimal lib to solve this problem, but I did not!
        """
        if times == None:    # special case of matrixTransform(TarRow,times)

            times_tmp = source_row_Number
            for j in range(self.col):
                self.body[target_row_Number][j] *= times_tmp
            return
        elif times == 'exchange':
            for i in range(self.col):
                self.body[target_row_Number][i],\
                    self.body[source_row_Number][i] \
                    = self.body[source_row_Number][i],\
                    self.body[target_row_Number][i]
            return
        else:
            for i in range(self.col):
                self.body[target_row_Number][i] += \
                times * self.body[source_row_Number][i]

    def matrixTranspose(self):
        ans = Matrix(self.col,self.row,self.row)
        # main column of ans is meaningless
        for i in range(self.row):
            for j in range(self.col):
                ans.body[j][i] = self.body[i][j]
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

def GetNormalValue(M,kind=1):
    """M is a matrix.
    
    Return the Normal value of M.
    """
    if kind == 'inf':
        """NorVal is the maximum value among the sums of every row. """
        sums = list()
        tmp = 0
        for i in range(M.row):
            for j in range(M.col):
                tmp += M.body[i][j]
            sums.append(tmp)
            tmp = 0
        
        sums = Quick_Sort(sums)
        return sums[-1]
    
    elif kind == 1:
        """NorVal is the maximum value among the sums of every column. """
        sums = list()
        tmp = 0
        for i in range(M.col):
            for j in range(M.row):
                tmp += M.body[j][i]
            sums.append(tmp)
            tmp = 0
        
        sums = Quick_Sort(sums)
        return sums[-1]
    
    elif kind == 2:
        Multi = M.matrixMulti(M.matrixTranspose())
        """The solve of elg is a little bit difficult,
        I will change the code after learning.
        
        Now I use the numpy package to solve it.
        """
        import numpy as np
        
        MUL = M.matrixMulti(M.matrixTranspose())
        tmp = np.array(MUL.body)

        a,b = np.linalg.eig(tmp)

        return np.sqrt(max(a))
        
    else:
        raise ValueError("""Bad input!\n""")

"""-------------------------my Main Function-------------------------"""

a = Matrix(2,2,2)
print('+-----------------------+-----------------------+')
a.body = [[1,-1],[2,1]]
b = GetNormalValue(a,'inf')
print('| infinity normal value |  ',b,'                  |')
b = GetNormalValue(a,1)
print('| 1 normal value        |  ',b,'                  |')
b = GetNormalValue(a,2)
print('| 2 normal value        |  ',b,'      |')
print('+-----------------------+-----------------------+')

a.body = [[10,15],[0,1]]
b = GetNormalValue(a,'inf')
print('| infinity normal value |  ',b,'                 |')
b = GetNormalValue(a,1)
print('| 1 normal value        |  ',b,'                 |')
b = GetNormalValue(a,2)
print('| 2 normal value        |  ',b,'      |')
print('+-----------------------+-----------------------+')

a.body = [[0.6,-0.5],[-0.1,0.3]]
b = GetNormalValue(a,'inf')
print('| infinity normal value |  ',b,'|')
b = GetNormalValue(a,1)
print('| 1 normal value        |  ',b,'                |')
b = GetNormalValue(a,2)
print('| 2 normal value        |  ',b,'     |')
print('+-----------------------+-----------------------+')