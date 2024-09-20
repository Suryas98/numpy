""
                     # NumPy 

# NumPy
import numpy as np
arr = np.array([1,2,3,4,5])
print(arr)
print(type(arr))
print(np.__version__)   

# Use a tuple

import numpy as np
arr = np.array((1,2,3,4,5))
print(arr)  

# 0-D array

import numpy as np
arr = np.array(42)
print(arr) 

# 1-D array and 2-D array

import numpy as np
arr = np.array([1,2,3,4,5])
arr1 = np.array([[1,2,3], [4,5,6]])
print(arr)
print(arr1)  

# 3-D array

import numpy as np
arr = np.array([[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]])
print(arr)    

# Check number of dimensions

import numpy as np
a=np.array(42)
b = np.array([1,2,3,4,5])
c = np.array([[1,2,3], [4,5,6]])
d = np.array([[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)  

# 5-dimensions

import numpy as np
arr = np.array([1,2,3,4], ndmin=5)
print(arr)
print('number of dimensions :', arr.ndim)

# NumPy indexing

import numpy as np
arr = np.array([1,2,3,4])
print(arr[0])          #1st value
print(arr[1])          #2nd value
print(arr[2],arr[3])   #3rd & 4th value   
print(arr[1]+arr[3])   #1st & 2nd value  

# Access the element on the 2nd row, 5th column

import numpy as np
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('5th element on 2nd row: ', arr[1,4]) 

#Access the 3rd element of the 2nd array of the 1st array

import numpy as np
arr = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
print(arr[1,1,2]) 

# NEGATIVE Indexing

import numpy as np
arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print('Last element from 2nd dim: ', arr[1,-1]) 

# Slicing Array

import numpy as np
arr = np.array([1,2,3,4,5,6,7])
print(arr[1:5])
print(arr[4:])
print(arr[:4]) 

#Negative slicing Index

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1])   

# Step Slicing

import numpy as np
arr = np.array([1,2,3,4,5,6,7])
print(arr[1:5:2])
print(arr[::2]) 

# 2-D Slicing Arrays

import numpy as np
arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr[1, 1:4])
print(arr[0:2, 2])
print(arr[0:2, 1:4])  

# NumPy Data Types

import numpy as np
a=np.array([1,2,3,4])
b=np.array(['apple','banana','cherry'])
c=np.array([1,2,3,4], dtype='S')
d=np.array([1,2,3,4], dtype='i4')
print(a.dtype)
print(b.dtype)
print(c)
print(c.dtype)
print(d)
print(d.dtype)  

# Converting Data Type on Existing Arrays

import numpy as np
a = np.array([1.1, 2.1, 3.1])
newa = a.astype('i')
print(newa)
print(newa.dtype) 

# Boolean

import numpy as np
a = np.array([-1,0,1])
newa = a.astype(bool)
print(newa)
print(newa.dtype)  

# Copy

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)

# View

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)  

#Check if Array Owns its Data

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
y = arr.view()
print(x.base)
print(y.base)

#Get the shape of an Array

import numpy as np
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a.shape)  

import numpy as np
a=np.array([1,2,3,4], ndmin=5)
print(a)
print(a.shape)  

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
a = np.array([1,2,3,4,5,6,7,8])
newa = arr.reshape(4, 3)
newb = arr.reshape(2, 3, 2)
print(newa)
print(newb)
print(a.reshape(2,4).base)  

#UnKnown Dimension

import numpy as np
arr = np.array([1,2,3,4,5,6,7,8])
newarr = arr.reshape(2, 2, -1)
print(newarr) 

# Flattening the Arrays

import numpy as np
arr = np.array([[1,2,3],[4,5,6]])
newarr = arr.reshape(-1)
print(newarr) 

#NumPy Array Iterating

# 1-D array

import numpy as np
arr = np.array([1,2,3])
for x in arr:
    print(x)  

# 2-D array

import numpy as np
arr = np.array([[1,2,3], [4,5,6]])
for x in arr:
    print(x)   

# Scalar 2-D array

import numpy as np
arr = np.array([[1,2,3],[4,5,6]])
for x in arr:
    for y in x:
        print(y)  

# Iterating 3-D Array

import numpy as np
arr = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
for x in arr:
    print(x)  

#Iterate down to scalars

import numpy as np
arr = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
for x in arr:
    for y in x:
        for z in y:
            print(z) 

# Iteration Array using nditer()

import numpy as np
arr = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
for x in np.nditer(arr):
    print(x) 

#Iterating array with different data types

import numpy as np
arr = np.array([1,2,3])
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
    print(x)  

# Iterating with Different Step Size

import numpy as np
arr = np.array([[1,2,3,4],[5,6,7,8]])
for x in np.nditer(arr[:, ::2]):
    print(x)  

# Enumerated Iteration using ndenumerate

import numpy as np
arr = np.array([1,2,3])
for idx, x in np.ndenumerate(arr):
    print(idx, x)  

#2-D array
import numpy as np
arr = np.array([[1,2,3,4], [5,6,7,8]])
for idx, x in np.ndenumerate(arr):
    print(idx, x)  

# NumPy Joining Array

import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.concatenate((arr1, arr2))
print(arr)

# 2-D Array

import numpy as np
arr1 = np.array([[1,2], [3,4]])
arr2 = np.array([[5,6], [7,8]])
arr = np.concatenate((arr1, arr2), axis=1 )
print(arr)  

# Array Using Stack Function

import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.stack((arr1, arr2), axis=1)
print(arr) 

# Stacking Along Rows, Columns and Height(depth)
import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr3 = np.hstack((arr1, arr2))  # Rows
arr4 = np.vstack((arr1, arr2))  # Columns
arr5 = np.dstack((arr1, arr2))  # Depth 
print(arr3)
print(arr4)
print(arr5)  

# Splitting NumPy Arrays

import numpy as np
arr = np.array([1,2,3,4,5,6])
newarr = np.array_split(arr, 3)
newarr1 = np.array_split(arr, 4)
print(newarr)
print(newarr1)
print(newarr[0])
print(newarr[1])
print(newarr[2])  

# Splitting 2-D Array

import numpy as np
arr = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
newarr = np.array_split(arr, 3)
print(newarr)  

# 2-D array into three 2-D arrays

import numpy as np
arr = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]])
newarr = np.array_split(arr,3)
newarr1 = np.array_split(arr, 3, axis=1)
newarr2 = np.hsplit(arr, 3)
print(newarr)
print(newarr1)
print(newarr2)  

# Searching Arrays

import numpy as np
arr = np.array([1,2,3,4,5,4,4])
a = np.array([1,2,3,4,5,6,7,8])
x = np.where(arr == 4)
x1 = np.where(a%2 == 0)
x2 = np.where(a%2 == 1)
print(x)
print(x1)
print(x2)   

# Search Sorted

import numpy as np
arr = np.array([6,7,8,9])
x = np.searchsorted(arr, 7)
x1 = np.searchsorted(arr, 7, side='right')  #Right side
print(x)
print(x1)  

# Multiple Values

import numpy as np
arr = np.array([1,3,5,7])
x = np.searchsorted(arr, [2,4,6])
print(x)

# Sorting Array

import numpy as np
a = np.array([3,2,0,1])
b = np.array(['banana','apple','cherry'])
c = np.array([True, False, True])
d = np.array([[3,2,4], [5,0,1]])
print(np.sort(a))
print(np.sort(b))
print(np.sort(c))
print(np.sort(d)) 

# Filtering Array

import numpy as np
arr = np.array([41,42,43,44])
x = [True, False, True, False]
newarr = arr[x]
print(newarr)

import numpy as np
arr = np.array([41,42,43,44])
filter_arr = []
for element in arr:
    if element > 42:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

import numpy as np
arr = np.array([41,42,43,44])
filter_arr = []
for element in arr:
    if element % 2 == 0:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

import numpy as np
arr = np.array([41,42,43,44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

# NumPy unfunc

# Integers

from numpy import random
x = random.randint(100, size=(5))
y = random.randint(100, size=(3, 5))
print(x)
print(y) 

# Float

from numpy import random
x = random.rand(5)
y = random.rand(3, 5)
print(x)
print(y)  

# Generate random number from Array

from numpy import random
x = random.choice([3,5,7,9])
y = random.choice([3,5,7,9], size=(3, 5))
print(x)
print(y) 

# Random Data Distribution

from numpy import random
x = random.choice([3,5,7,9], p=[0.1,0.3,0.6,0.0], size=(100))
y = random.choice([3,5,7,9], p=[0.1,0.3,0.6,0.0], size=(3, 5))
print(x)
print(y)

#Shuffling Array

from numpy import random
import numpy as np
arr = np.array([1,2,3,4,5])
random.shuffle(arr)
print(arr)

# Generating Permutation of Arrays

from numpy import random
import numpy as np
arr = np.array([1,2,3,4,5])
print(random.permutation(arr))  

# Seaborn

import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot([0,1,2,3,4,5])
plt.show() 

# Normal Distribution

from numpy import random
x = random.normal(size=(2,3))
y = random.normal(loc=1, scale=2, size=(2,3))
print(x)
print(y) 

# Visualization of Normal Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.normal(size=1000), hist=False)
plt.show() 

# Binomial Distribution

from numpy import random
x = random.binomial(n=10, p=0.5, size=10)
print(x)

# Visualization of Binomial Distribution

from numpy import random
import matplotlib.pyplot as plot
import seaborn as sns
sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)
plt.show()  


# Diff b/w Normal and binomial Distrubution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')
plt.show() 

# Poission Distrubution

from numpy import random
x = random.poisson(lam=2, size=10)
print(x)

# Visualization of poisson Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.poisson(lam=2, size=1000), kde=False)
plt.show()  

# Uniform Distribution

from numpy import random
x = random.uniform(size=(2,3))
print(x)

# Visualization of Uniform Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.dist
plot(random.uniform(size=1000), hist=False)
plt.show()

# Logistic Distribution

from numpy import random
x = random.logistic(loc=1, scale=2, size=(2,3))
print(x)

# Visualization of Logistic Distribution

from numpy import random
import marplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.logistic(size=1000), hist=False)
plt.show()  

# Multinomial Distribution

from numpy import random
x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print(x) 

# Exponential Distribution

from numpy import random
x = random.exponential(scale=2, size=(2, 3))
print(x)  

# Visualization of Exponential Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.exponential(size=1000), hist=False)
plt.show()  

# Chi Square Distribution

from numpy import random
x = random.chisquare(df=2, size=(2, 3))
print(x)

# Visualization of Chi Square Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.chisquare(df=1, size=1000), hist=False)
plt.show() 

# Rayleigh Distribution

from numpy import random
x = random.rayleigh(scale=2, size=(2, 3))
print(x) 

# Visualization of Rayleigh Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.rayleigh(size=1000), hist=False)
plt.show() 

# Pareto Distribution

from numpy import random
x = random.pareto(a=2, size=(2, 3))
print(x)

# Visualization of Pareto Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.displot(random.pareto(a=2, size=1000), kde=False)
plt.show()  

# Zipf Distribution

from numpy import random
x = random.zipf(a=2, size=(2, 3))
print(x)

# Visualization of zipf Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
x = random.zipf(a=2, size=1000)
sns.distplot(x[x<10], kde=False)
plt.show()  

# Ufunc Intro

x =[1,2,3,4]
y =[4,5,6,7]
z =[]
for i, j in zip(x,y):
    z.append(i + j)
print(z)

# add() function

import numpy as np
x = [1,2,3,4]
y = [4,5,6,7]
z = np.add(x,y)
print(z) 

# Ufunc Create Function

import numpy as np
def myadd(x,y):
    return x+y
myadd = np.frompyfunc(myadd, 2, 1)
print(myadd([1,2,3,4], [5,6,7,8]))

# Check if a function is ufunc

import numpy as np
print(type(np.add))

# Concatenate()

import numpy as np
print(type(np.concatenate)) 

# Ufunc Logs

import numpy as np
arr = np.arange(1, 10)
print(np.log2(arr))
print(np.log10(arr))
print(np.log(arr)) 

# Log at Any Base

from math import log
import numpy as np
nplog = np.frompyfunc(log, 2, 1)
print(nplog(100, 15))

# Ufunc Summations

import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
a = np.add(arr1, arr2)
b = np.sum([arr1, arr2])
c = np.sum([arr1, arr2], axis=1)
d = np.cumsum(arr1)
print(a)
print(b)
print(c)
print(d) 

# Ufunc Products

import numpy as np
arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
a = np.prod(arr1)
b = np.prod([arr1, arr2])
c = np.prod([arr1, arr2], axis = 1)
d = np.cumprod(arr2)
print(a)
print(b)
print(c)
print(d) 

# Difference

import numpy as np
ar = np.array([10, 15, 25, 5])
a = np.diff(ar)
b = np.diff(ar, n=2)
print(a)
print(b) 

# LCM in Array

import numpy as np
n1 = 4
n2 = 6
ar1 = np.array([3,6,9])
ar2 = np.arange(1,11)

n = np.lcm(n1,n2)
a = np.lcm.reduce(ar1)
b = np.lcm.reduce(ar2)

print(n)
print(a)
print(b) 

# GCM in Array

import numpy as np
n1 = 6
n2 = 9
ar = np.array([20,8,32,36,16])
a = np.gcd(n1,n2)
b = np.gcd.reduce(ar)
print(a)
print(b) 

# Convert Degree into Radian

import numpy as np
a = np.array([90, 180, 270, 360])
b = np.array([np.pi/2, np.pi, 1.5*np.pi, 2*np.pi])
c = np.deg2rad(a)
d = np.rad2deg(b)
print(c)
print(d) 

# Finding Angles

import numpy as np
a = np.arcsin(1.0)
b = np.array([1, -1, 0.1])
c = np.arcsin(b)
print(a)
print(c) 

# Hypotenues

import numpy as np
base = 3
perp = 4
x = np.hypot(base, perp)
print(x)  

# Hyperbolic Functions

import numpy as np
a = np.sinh(np.pi/2)
ar = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
b = np.cosh(ar)

print(a)
print(b)  

# Finding Angles

import numpy as np
a = np.arcsinh(1.0)
ar = np.array([0.1, 0.2, 0.5])
b = np.arctanh(ar)
print(a)
print(b) 

# Creare sets in NumPy

import numpy as np
ar = np.array([1,1,1,2,3,4,5,5,6,7])
x = np.unique(ar)
print(x)

# Finding Union

import numpy as np
ar1 = np.array([1,2,3,4])
ar2 = np.array([3,4,5,6])
a = np.union1d(ar1, ar2)
print(a)

# Finding Intersection

import numpy as np
ar1 = np.array([1,2,3,4])
ar2 = np.array([3,4,5,6])
a = np.intersect1d(ar1, ar2, assume_unique=True)
print(a)

# Finding Difference

import numpy as np
ar1 = np.array([1,2,3,4])
ar2 = np.array([3,4,5,6])
a = np.setdiff1d(ar1, ar2, assume_unique=True)
print(a)

# Finding Symmetric Difference

import numpy as np
ar1 = np.array([1,2,3,4])
ar2 = np.array([3,4,5,6])
a = np.setxor1d(ar1, ar2, assume_unique=True)
print(a)







