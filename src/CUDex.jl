module CUDex

# package code goes here
const libcudex = Libdl.find_library(["libcudex"], [dirname(@__FILE__)])

dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

export gpu, relu

include("gpu.jl")
include("karray.jl")
include("basic.jl")
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
