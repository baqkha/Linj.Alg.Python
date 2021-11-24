import numpy as np
import matplotlib.pyplot as plt

#upg a


'''
ANTECKNINGAR - LITE KAOS 
------------------------------------------
HUR MAN RITAR: x + y + z = 1

x + y + z = 1

skriv om z = 1 - x - y 

a = 2
h = 0.01 (steg mellan punkter)
n = int(2*a / h) + 1

X = np.zeros((n,n))
Y = np.zeros((n,n))
Z = np.zeros((n,n))

for j in range(0,n):
    for k in range(0,n):
        X[j, k] = -a + h*k
        Y[j, k] = -a + h*j
        Z[j, k] = 1 - X[j, k] - Y[j, k]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, color='blue')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

-----------------------------------------------------------------
Hur man ritar en  (vektor) normalvektor
quiver = pil



ax.quiver(0, 0, 1,1,1,1, color='red')  # (0,1,1) = från vilken pos. (1,1,1) = Läng


---------------------------------------------------------------

hur man ritar linje:
Linje mellan två punkter

x = np.array([-1, 1])
y = np.array([-1, 1])
z = np.array([-1, 1])

ax.plot(x, y, z, color='red')

------------------------------
Räkna skärningspunkt:

x = np.array([1/3])
y = np.array([1/3])
z = np.array([1/3])

ax.scatter(x, y, z, marker='o', s=50)

-----------------------------------------------------------
Illustrera uppgift från boken

l1: (x,y,z) = (1, -1, 1) + t(-2,1,2)
l2: (x,y,z) = (0,1,1) + s(1,-1,0)

Räkna avstånd mellan dessa två linjer

pi = plan är parallel med l1 och l2 och innehåller tex. l1

Vektorprodukt (för att få pi) = (2,2,1)   # (-2,1,2)x(1,1,0)

pi = 2x + 2y + z = 1

z = 1 - 2x - 2y



a = 2
h = 0.01 (steg mellan punkter)
n = int(2*a / h) + 1

X = np.zeros((n,n))
Y = np.zeros((n,n))
Z = np.zeros((n,n))

for j in range(0,n):
    for k in range(0,n):
        X[j, k] = -a + h*k
        Y[j, k] = -a + h*j
        Z[j, k] = 1 - 2*X[j, k] - 2*Y[j, k]

x = np.array([3, -1])
y = np.array([-2, 0])
z = np.array([-1, 3])
ax.plot(x, y, z =color'red')

x = np.array([-1, 3])
y = np.array([2, -2])
z = np.array([1, 1])




l1 innehåller punkterna = (3, -2, -1) och (-1, 0, 3)  #subtrahera punkter på l1
ax.quiver(1,-1,1,-1,2,0, color='red')  # från (1,-1,1) till (0,1,1) = u = (-1, 2, 0)

u = np.array([-1, 2, 0])
n = np.array([2/3, 2/3, 1/3])

print((u*n).sum())

'''