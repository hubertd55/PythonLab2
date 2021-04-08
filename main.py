import numpy as np
import matplotlib.pyplot as plt
# probne zadania z numpy

#tworzenie tablicy
#arr=np.array([1, 2, 3, 4, 5])
#print(arr)

'''
A=np.array([[1, 2, 3], [7, 8, 9]])
print(A)

v = np.arange(1,7)#od 1 co 1 do 6
print(v,"\n")
v1 = np.arange(-2,7)#od -2 co 1 do 6
print(v1,"\n")
v2 = np.arange(1,10,3) #od 1 co 3 do mniejszej niz 10
print(v2,"\n")
v3 = np.arange(1,10.1,3) #od 1 co 3 do mniejszej niz 10.1
print(v3,"\n")
v4 = np.arange(1,11,3)#od 1 co 3 do mniejszej niz 11
print(v4,"\n")
v5 = np.arange(1,2,0.1)#od 1 do 2 co 0.1
print(v5,"\n")
'''


'''
v = np.linspace(1,3,4)#od 1 do 3 tworzy 4 elementy
print(v)
v1 = np.linspace(1,10,4)#od 1 do 10 tworzy 4 elementy
print(v1)
'''
'''
X = np.ones((2, 3))#dwa wiersze 3 kolumny 1

Y = np.zeros((2, 3, 4))#dwa wiersze po macierzy 3x4 z zer

Z = np.eye(2)#tworzenie macierzy jednostkowych
Z1 = np.eye(2, 2)
Z2 = np.eye(2, 3)

Q = np.random.rand(2, 5)#macierz 2x5 z losowymi wartościami

Q1 = np.round(10*np.random.rand(3, 3))

print(X,"\n-----------\n",Y,"\n--------------\n",Z,"\n---------------\n",Z1,"\n--------------\n",Z2,"\n-----------------\n",Q,"\n--------------\n",Q1)
'''

'''
#tworzenie z innych tablic
A=np.array([[1, 2, 3], [7, 8, 9]])
Z = np.eye(2)#tworzenie macierzy jednostkowych
X = np.ones((2, 3))#dwa wiersze 3 kolumny 1

U = np.block([X,Z])
print(U)

#odwoluje sie do konkretnych elementow tablicy
#V = np.block([[np.block([np.block([[np.linspace(1,3,3)],[np.zeros((2,3))]]) ,np.ones((3,1))])],[np.array([100, 3, 1/2, 0.333])]] )
# print(V)
# print("\n-----\n")
# print( V[0,2] )
# print("\n-----\n")
# print( V[3,0] )
# print("\n-----\n")
# print( V[3,3] )
# print("\n-----\n")
# print( V[-1,-1] )
# print("\n-----\n")
# print( V[-4,-3] )
# print("\n-----\n")
#
# print( V[3,:] )
# print("\n-----\n")
# print( V[:,2] )
# print("\n-----\n")
# print( V[3,0:3] )
# print("\n-----\n")
# print( V[np.ix_([0,2,3],[0,-1])] )
# print("\n-----\n")
# print( V[3] )
# print("\n-----\n")

# print(V)
# Q=np.delete(V,2,0)
# print("\n-----\n")
# print(Q)
# print("\n-----\n")
v = np.arange(1,7)
print(v)
print("\n-----\n")
#print( np.delete(v, 3, 0) )
#print("\n--------------------\n")
#sprawdzanie rozmiarów tablic
print(np.size(v))#out 6
print(np.shape(v))#out (6,)


print(np.size(A))#out 6
print(np.shape(A))#out (2,3)
'''

'''
#operacje na macierzach

A = np.array([[1, 0, 0],[2, 3, -1],[0, 7, 2]] )
B = np.array([[1, 2, 3],[-1, 5, 2],[2, 2, 2]] )
#print(A,"\n---\n",B,"\n---\n",A+B,"\n---\n",A-B,"\n---\n",A+2,"\n---\n",2*A)#out kolejne przekształcone macierze

#mnożenie macierzowe
MM1=A@B
#print(MM1)
MM2=B@A
#print(MM2)
#mnożenie i dzielenie tablicowe
MT1=A@B
MT2=B@A
DT1=A/B
#print(A,"\n---\n",B,"\n---\n",MT1,"\n---\n",MT2,"\n---\n",DT1)

#dzielenie macierzowe -URL
C = np.linalg.solve(A,MM1)
#print(C,"\n\n",B)#C = B

x = np.ones((3,1))
b =  A@x
y = np.linalg.solve(A,b)
#print(y)

#potęgowanie
PM = np.linalg.matrix_power(A,2) # por. A@A
#print(PM,"\n\n",A@A)#identyczne
PT = A**2  # por. A*A
#print(PT,"\n\n",A*A)#identyczne

#transpozycja
#print(A,"\n---\n",A.T)
#A.T=A.tranpose()

#A.conj().T # hermitowskie sprzezenie macierzy (dla m. zespolonych)
# A.conj().transpose()

#porównania
#print(np.logical_not(A),"\n---\n",np.logical_and(A,B),"\n---\n",np.logical_or(A,B),"\n---\n",np.logical_xor(A,B),"\n---\n")

# print( np.all(A) )
# print( np.any(A) )
# print( A > 4 )
# print( np.logical_or(A>4, A<2))
# print( np.nonzero(A>4) )
# print( A[np.nonzero(A>4) ] )

print(np.max(A),"\n---\n")
print(np.min(A),"\n---\n")
print(np.max(A,0),"\n---\n")
print(np.max(A,1),"\n---\n")
print( A.flatten() ,"\n---\n")
print( A.flatten('F'),"\n---\n")
'''


#------------------------------rozdział 5-------------------
# x = [1,2,3]
# y = [4,6,5]
# plt.plot(x,y)
# plt.show()

#sinus
# x = np.arange(0.0, 2.0, 0.01)
# y = np.sin(2.0*np.pi*x)
# plt.plot(x,y)
# plt.show()

#sinus2
# x = np.arange(0.0, 2.0, 0.01)
# y = np.sin(2.0*np.pi*x)
# plt.plot(x,y,'r:',linewidth=6)
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Nasz pierwszy wykres')
# plt.grid(True)
# plt.show()

#wiele wykresów
# x = np.arange(0.0, 2.0, 0.01)
# y1 = np.sin(2.0*np.pi*x)
# y2 = np.cos(2.0*np.pi*x)
# plt.plot(x,y1,'r:',x,y2,'g')
# plt.legend(('dane y1','dane y2'))
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Wykres')
# plt.grid(True)
# plt.show()

#wiele wykresów2
# x = np.arange(0.0, 2.0, 0.01)
# y1 = np.sin(2.0*np.pi*x)
# y2 = np.cos(2.0*np.pi*x)
# y = y1*y2
# l1, = plt.plot(x,y,'b')
# l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
# plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Wykres ')
# plt.grid(True)
# plt.show()

###------------------------------------REALIZACJA ZADAN----------------------------------



#zad3

a1=np.array([np.linspace(1,5,5),
            np.linspace(5,1,5)])
a2=np.zeros((3,2))
a3=np.array(np.full((2,3),2))
a4=np.arange(-90,-60,10)
a5=np.array(np.full((5,1),10))

A=np.block([[a3],[a4]])
A=np.block([[a2,A]])
A=np.block([[a1],[A]])
A=np.block([[A,a5]])
#print(A)

#zad4
B=A[2,:] + A[4,:]

#zad5
C=[np.max(A[:,0]),np.max(A[:,1]),np.max(A[:,2]),np.max(A[:,3]),np.max(A[:,4]),np.max(A[:,5])]

#zad6
D1=np.delete(B,5,0)
D=np.delete(D1,0,0)

