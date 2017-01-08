VERSION >= v"0.4.0-dev+6521" && __precompile__()

module CUDex

# package code goes here
const libcudex = Libdl.find_library(["libcudex"], [dirname(@__FILE__)])

dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

export gpu, DexArray, relu, invx, sigm, logp, conv_f, pool_f

include("gpu.jl")
include("dexptr.jl")
include("dexarray.jl")
include("basic_bi.jl")
include("basic_uni.jl")
include("cublas.jl")
include("cudnn.jl")

# See if we have a gpu at initialization:
function __init__()
    try
        r = gpu(true)
        info(r >= 0 ? "Using GPU $r" : "No GPU found, Using CPU")
    catch e
        warn("Using CPU: $e")
        gpu(false)
    end
end

end # module
