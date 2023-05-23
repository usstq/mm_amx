import mmopt
import sys

print("========== tensorND to numpy (zero-copy) =============== ")

a = mmopt.tensorND([2,3,4], False)
print(a)

# view
m = memoryview(a)

print(m.shape)
print(m.strides)
print(m.format)

m[0,0,0] = 0

print(m.obj)

# import into numpy (zero-copy)
import numpy

arr = numpy.array(a, copy=False)

print(arr.shape)
print(arr.strides)

arr[0,0,:] = 1
arr[:,0,0] = 2

print(arr)

print(a)

del a
print(m[0,0,0])
del m
arr[0,0,:] = 2
print(arr)
del arr

print("========== numpy to tensorND (zero-copy) =============== ")

n0 = numpy.ones([3,3], dtype=numpy.float32)

a = mmopt.tensorND(n0)
print(a)

n0[2,2] = 2
print(a)

print("========== return tensorND =============== ")

a = mmopt.g()
print(a)

# wrap it as numpy array
arr = numpy.array(a, copy=False)

arr[0,:] = 99
print(a)

# mmopt.h(mmopt.tensorND(arr), 0.23)
mmopt.h(a, 0.23)

print(a)
del a

print("========== return tensorND =============== ")

arr = mmopt.g2()
print(arr)

print("finished")
