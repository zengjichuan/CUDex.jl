using CUDex

SIZE = 100000

x32 = DexArray(rand(Float32,SIZE))
x64 = DexArray(rand(Float64,SIZE))
s32 = rand(Float32)
s64 = rand(Float64)

# sqrt
@time sqrt(x32)
@time sqrt(x64)

# exp
@time exp(x32)
@time exp(x64)

# log
@time log(x32)
@time log(x64)

# tanh
@time tanh(x32)
@time tanh(x64)

# .*
@time x32.*s32
@time x64.*s64
