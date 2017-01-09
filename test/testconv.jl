using CUDex

x = DexArray(rand(6,5,4,3))
w = DexArray(rand(3,3,4,2))

t1 = conv_f(x, w)
t2 = relu(t1)
t3 = pool_f(t2)
