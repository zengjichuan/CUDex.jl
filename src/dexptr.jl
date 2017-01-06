# DexPtr type holds a gpu (dev>=0) or cpu (dev=-1) allocated pointer.
# We try to minimize the number of actual allocations, which are slow,
# by reusing preallocated but garbage collected pointers.

type DexPtr
    ptr::Cptr
    len::Int
    dev::Int
    parent
end

# We use the DexPtrs type to keep track of allocated and garbage
# collected pointers: We keep one DexPtrs struct per size per device.

type DexPtrs
    used::Int                   # number of allocated pointers
    free::Array{Cptr,1}         # pointers available for reuse
    DexPtrs()=new(0,Array(Cptr,0))
end

# DexFree[dev+2] will hold a dictionary from sizes to DexPtrs for
# device dev.  DexFree[1] reserved for the cpu.  It is initialized in
# DexPtr(nbytes).

DexFree = nothing
initDexFree()=(global DexFree=[ Dict{Int,DexPtrs}() for i=1:gpuCount()+1 ])

# When Julia gc reclaims a DexPtr object, the following special
# finalizer does not actually release the memory, but inserts it in
# the DexFree[dev+2] dict keyed by length in bytes so it can be
# reused.

function freeDexPtr(p::DexPtr)
    ptrs = DexFree[p.dev+2][p.len]
    push!(ptrs.free,p.ptr)
end

# This is the low level DexPtr constructor, it adds the finalizer and
# sets parent to `nothing` which is only needed for shared pointers.

function DexPtr(ptr::Cptr,len::Int,dev::Int)
    kp = DexPtr(ptr,len,dev,nothing)
    finalizer(kp, freeDexPtr)
    return kp
end

# This constructor is used to create a shared pointer.  We need to
# keep the parent field to prevent premature gc of the parent.  The
# child does not need a special finalizer.

function DexPtr(parent::DexPtr, offs::Integer, len::Integer)
    if len < 0 || offs < 1 || offs+len-1 > parent.len
        throw(BoundsError())
    end
    DexPtr(parent.ptr+offs-1, len, parent.dev, parent)
end

# This the main DexPtr constructor.  It tries to avoid actual
# allocation which is slow.  First it tries to find a previously
# allocated and garbage collected pointer in DexFree[dev+2].  If not
# available it tries to allocate a new one (about 10 μs).  Otherwise
# it tries running gc() and see if we get a pointer back (about 75
# ms).  Finally if all else fails, it calls dexgc which cleans up all
# allocated and garbage collected DexPtrs on the current device and
# tries allocation one last time.

function DexPtr(nbytes::Integer)
    DexFree==nothing && initDexFree()
    dev = gpu()
    ptrs = get!(DexPtrs,DexFree[dev+2],nbytes)
    if !isempty(ptrs.free)
        return DexPtr(pop!(ptrs.free),nbytes,dev)
    end
    ptr = dexMalloc(nbytes)
    if ptr != nothing
        ptrs.used += 1
        return DexPtr(ptr,nbytes,dev)
    end
    gc(); print(".")
    if !isempty(ptrs.free)
        return DexPtr(pop!(ptrs.free),nbytes,dev)
    end
    knetgc(); print("+")
    ptr = dexMalloc(nbytes)
    if ptr != nothing
        ptrs.used += 1
        return DexPtr(ptr,nbytes,dev)
    end
    error("Out of gpu memory")
end

# This does the actual allocation, returns `nothing` in case of error
function dexMalloc(nbytes::Int)
    gpu() >= 0 || return(convert(Cptr, pointer(Array(UInt8,nbytes))))
    ptr = Cptr[0]
    ret = ccall(("cudaMalloc","libcudart"),UInt32,(Ptr{Cptr},Csize_t),ptr,nbytes)
    if ret == 0
        return ptr[1]
    else # error("cudaMalloc($nbytes) error $ret")
        return nothing
    end
end

# If you really want to clean up memory you need to call dexgc: Note
# that this only cleans the current device.  It frees all pointers
# previously allocated and garbage collected.
function dexgc(dev=gpu())
    gc_enable(false)
    for v in values(DexFree[dev+2])
        if dev >= 0
            for p in v.free
                @cuda(cudart,cudaFree,(Cptr,),p)
            end
        end
        v.used -= length(v.free)
        empty!(v.free)
    end
    gc_enable(true)
end

# Some utilities
meminfo(i=gpu())=[(k,v.used,length(v.free)) for (k,v) in DexFree[i+2]]
gpufree(i=gpu())=nvmlDeviceGetMemoryInfo(i)[2]

function gpuinfo(msg="",dev=gpu())
    msg != "" && print("$msg ")
    g = nvmlDeviceGetMemoryInfo(dev)
    println((:dev,dev,:total,g[1],:free,g[2],:used,g[3]))
    for (s,u,f) in sort(meminfo(dev), by=(x->x[1]), rev=true)
        println((:size,s,:alloc,u,:avail,f))
    end
end

function memdbg(msg="")
    m = nvmlDeviceGetMemoryInfo()
    c = cudaGetMemInfo()
    x = [0,0]
    for (s,u,f) in meminfo()
        x[1] += s*u
        x[2] += s*f
    end
    println("$msg: ntotal: $(m[1]) nfree: $(m[2]) nused: $(m[3]) ktotal: $(x[1]) kfree: $(x[2])\n nfree+used: $(m[2]+m[3]) ctotal: $(c[2]) cfree: $(c[1]) ktotal-kfree: $(x[1]-x[2]) nused-ktotal: $(m[3]-x[1])")
end
