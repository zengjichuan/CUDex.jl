Base.compilecache("CUDex")
try success(`nvcc --version`)
    cd("../src/cuda") do
        run(`make libcudex.so`)
    end
catch
    warn("CUDA not installed, GPU support will not be available.")
end
