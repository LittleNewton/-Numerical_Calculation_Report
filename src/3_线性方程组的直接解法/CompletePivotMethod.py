# filename: 3.2 CompletePivotMethod.py

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

def GaussJordanCol_complete(A):
    Max = 0                         # initialize the variable
    position_row = 0                # the row number of the maximum value
    position_col = 0
    
    for i in range(A.row):
    
        for j in A.CheckedRow:
            for k in range(A.mainCol):  # not in all the columns
                if abs(Max) <= abs(A.body[j][k]):
                    Max = A.body[j][k]
                    position_row = j
                    position_col = k
    
        A.matrixTransform(position_row, 1 / Max)
        
        A.CheckedRow.remove(position_row)
        
        for j in range(A.row):
            if j != position_row:
                A.matrixTransform(j,position_row,-1 * A.body[j][position_col])
        
        Max = 0         
        position_row = 0
        position_col = 0
    
    begin = 0
    for j in range(A.mainCol):
        for i in range(A.row):
            if A.body[i][j] == 1:
                A.matrixTransform(begin,i,'exchange')
                begin += 1
    return A

"""-------------------------my Main Function-------------------------"""

A = Matrix(4,8,4)
A.body[0] = [ 2, 1,-3, 1, 1, 0, 0, 0]
A.body[1] = [ 3, 1, 0, 7, 0, 1, 0, 0]
A.body[2] = [-1, 2, 4,-2, 0, 0, 1, 0]
A.body[3] = [ 1, 0,-1, 5, 0, 0, 0, 1]

print('The original matrix is shown below:\n')
for i in range(A.row):
    for j in range(A.col):
        print(int(A.body[i][j]),'\t',end='')
    print('')

A = GaussJordanCol_complete(A)

print('\nThe matrix after primary transformation is shown below:\n')
for i in range(A.row):
    for j in range(A.col):
        if A.body[i][j] == 0.0:
            A.body[i][j] = 0
        if A.body[i][j] == -0.0:
            A.body[i][j] = 0
        if A.body[i][j] - round(A.body[i][j]) == 0:
            A.body[i][j] = round(A.body[i][j])
        print(round(A.body[i][j],4),'   ',end='')
    print('')