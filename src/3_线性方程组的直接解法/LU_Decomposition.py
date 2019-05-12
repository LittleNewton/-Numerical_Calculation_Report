# filename: LU_Decomposition.py

class Matrix:
    """Abatrct class representing a Matrix.
    
    The internal structure is two dimension list.
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

    def valid(self,e):
        """If the two matrix are not in the same form, return false,
        else return true.
        
        It is useful in the next functions.
        """
        if self.row != e.row or self.col != e.col:
            return False
        else:
            return True

    def add(self,e):
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

    def constMulti(self,const):
        """A constant number multiple a matrix."""
        tmp = Matrix(self.row,self.col,self.mainCol)
        for i in range(self.row):   # deep copy
            for j in range(self.col):
                tmp.body[i][j] = self.body[i][j]

        for i in range(self.row):
            for j in range(self.col):
                tmp.body[i][j] *= const
        return tmp

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
    
    def Peel(self,layer):
        """Getting part of the matrix."""
        B = Matrix(self.row-layer,self.col-layer,self.mainCol)
        for i in range(self.row-layer):
            for j in range(self.col-layer):
                B.body[i][j] = A.body[i + layer][j + layer]
        return B
    
    def combine(self,B,layer):
        """A long with method Peel."""
        for i in range(self.row-layer):
            for j in range(self.col-layer):
                A.body[i + layer][j + layer] = B.body[i][j]

def LU_Decompose_Doolittle(A):
    """Decompose matrix A with Dolittle method.
    
    In this function, I use devide and conquer method.
    """
    for i in range(A.col-1):
        if i == 0:
            for j in range(1,A.row):
                A.body[j][i] = A.body[j][i] / A.body[0][0]
        else:
            for j in range(A.col-i):
                for k in range(i):
                    A.body[i][i+j] = A.body[i][i+j] - \
                        A.body[i][k] * A.body[k][i+j]
            for j in range(A.row-i-1):
                for k in range(i):
                    A.body[i+j+1][i] = A.body[i+j+1][i] - \
                        A.body[i+j+1][k] * A.body[k][i]
                A.body[i+j+1][i] = A.body[i+j+1][i] / A.body[i][i]
    return A

"""-------------------------my Main Function-------------------------"""

A = Matrix(4,5,4)
A.body[0] = [12,-3,3,4,15]
A.body[1] = [-18,3,-1,-1,-15]
A.body[2] = [1,1,1,1,6]
A.body[3] = [3,1,-1,1,2]

print('The original matrix is shown below:\n')
for i in range(A.row):
    for j in range(A.col):
        print(int(A.body[i][j]),'\t',end='')
    print('')

A = LU_Decompose_Doolittle(A)

B = Matrix(4,4,4)
for i in range(A.row):
    for j in range(A.row):
        B.body[i][j] = A.body[i][j]

L1 = Matrix(4,4,4)

for i in range(L1.col):
    for j in range(L1.row-i):
        L1.body[j+i][i] = B.body[j+i][i]
for i in range(L1.row):
    L1.body[i][i] = 1

U = L1.constMulti(-1).add(B)    # I am the best! professional!
for i in range(L1.row):
    U.body[i][i] += 1

print('-----------------------------------\nA:\n')
for i in range(4):
    for j in range(4):
        if B.body[i][j] == 0.0:
            B.body[i][j] = 0
        if B.body[i][j] == -0.0:
            B.body[i][j] = 0
        if B.body[i][j] - round(B.body[i][j]) == 0:
            B.body[i][j] = round(B.body[i][j])
        print(round(B.body[i][j],2),'\t',end='')
    print('')

print('-----------------------------------\nU:\n')

for i in range(4):
    for j in range(4):
        if U.body[i][j] == 0.0:
            U.body[i][j] = 0
        if U.body[i][j] == -0.0:
            A.body[i][j] = 0
        if U.body[i][j] - round(U.body[i][j]) == 0:
            U.body[i][j] = round(U.body[i][j])
        print(round(U.body[i][j],2),'\t',end='')
    print('')

print('-----------------------------------\nL:\n')

for i in range(4):
    for j in range(4):
        if L1.body[i][j] == 0.0:
            L1.body[i][j] = 0
        if L1.body[i][j] == -0.0:
            L1.body[i][j] = 0
        if L1.body[i][j] - round(L1.body[i][j]) == 0:
            L1.body[i][j] = round(L1.body[i][j])
        print(round(L1.body[i][j],2),'\t',end='')
    print('')