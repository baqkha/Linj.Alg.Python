import numpy as np
import matplotlib.pyplot as plt



a = 2
h = 0.01 #mellan punkter distans

n = int(2*a / h) + 1
X = np.zeros((n,n))
Y = np.zeros((n,n))
Z = np.zeros((n,n))

for j in range(0, n):
    for k in range (0, n):
        X[j, k] = -a + h*k
        Y[j, k] = -a + h*j
        Z[j, k] = X[j,k]**2 + Y[j,k]**2


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, color='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

'''
x = np.linspace (-2*np.pi, 2*np.pi, 100)
type(x)
x.size


y = np.sin(x)
    #samma size som x


fig, (hej1, hej2) = plt.subplots(2,1)
hej1.plot(x,y)
hej2.plot(x, y, linewidth=10, color='y', marker='2', markersize=10)
plt.show()
'''
''''
A = np.array([ [ 1, 2, 3], [4, 5, 6], [0, 9, 8]])
print(A)
A1 = np.linalg.inv(A)
'''

'''
import matplotlib.pyplot as plt
x = np.linspace (-2*np.pi, 2*np.pi, 100)
type(x)
x.size

räkna sin:
    y = np.sin(x)
    samma size som x

fig, plot = plt.subplots(1,1)
    fig, plot 

plot.plot(x,y)
plt.show()


fig, (hej1, hej2) = plt.subplots(2,1)
hej1.plot(x,y)
hej2.plot(x, y, linewidth=10, color='y', marker='2', markersize=10)
plt.show()



----
spara i pdf:

hej.plot(x,y)
plt.savefig('trstbiild.pdf')
-----
plotta polynom:
    def p(x);
        return x**2 - 2*x + 1
    print(p(o))
x = np.linspace(-3, 3, 20) // skapa 20 punkter mellan -3, 3
y = p(x)

fig, hej = plt.subplots(1,1)
hej.plot(x,y, marker = 'o')
plt.show()
-----------------
3D grafer

z = x^2 + y^2

a = 2
h = 1 //mellan punkter distans
n = int(2*a / h) + 1
X = np.zeros((n,n))
Y = np.zeros((n,n))
Z = np.zeros((n,n))

for j in range(0, n);
    for k in range (0, n):
        X[j, k] = -a + h*k
        Y[j, k] = -a + h*j
        Z[j, k] = X[j,k]**2 + Y[j,k]**2


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, color='blue')
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_xlabel('z')
plt.show()

'''




#  A = numpy.array([ [ 1, 2, 3], [4, 5, 6], [0, 9, 8]]) // skapa matris
# A eller print(A)
# matrismultiplikation: A @ A

# y = np.array([[1], [2], [3]])
# np.linalg.inv(A) @ y     //räkna invers
# np.lingalg.solve (A, y) // räknar ixå ut Ax = y


# np.eye(3)


#format ( ..., '.30f') //antal decimaler


# transponera matris = A.T = A.transpose()
#minsta högsta element i A = A.min() , A.max()
# np.ones((2,3)) // skapa matris bara med ettor
# np.zerios((2,3)) // skapa matris med nollor
# A.sum() = summa av alla element