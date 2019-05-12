# filename: 3.4 Seize Method.py

class ThreeTriangleMatrix:
    def __init__(self,a,b,c,m):
        self.a = a
        self.b = b
        self.c = c
        self.row = m
        self.col = m

    def calculate(self):
        alpha = []
        beta = []
        gamma = self.a
        
        alpha.append(self.b)
        beta.append(self.c / alpha[0])
        
        for i in range(1,self.row-1):
            alpha.append(self.b - gamma * beta[i-1])
            beta.append(self.c / alpha[i])
        
        alpha.append(self.b - gamma * beta[-1])
        
        return [gamma,alpha,beta]

A = ThreeTriangleMatrix(-1,4,-1,5)
A.calculate()

L = [[0 for i in range(A.row)] for j in range(A.col)]

for i in range(A.row-1):
    L[i+1][i] = A.calculate()[0]
    L[i][i] = A.calculate()[1][i]

L[A.row-1][A.row-1] = (A.calculate()[1])[A.row-1]

U1 = [[0 for i in range(A.row)] for j in range(A.col)]
for i in range(A.row):
    U1[i][i] = 1

for i in range(A.row-1):
    U1[i][i+1] = A.calculate()[2][i]

print('----------------------------------\nU1:\n')
for i in range(A.row):
    for j in range(A.col):
        print(round(U1[i][j],2),'\t',end='')
    print ('')

print('----------------------------------\nL:\n')
for i in range(A.row):
    for j in range(A.col):
        print(round(L[i][j],2),'\t',end='')
    print('')