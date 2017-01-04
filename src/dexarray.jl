"""

DexArray is a container for GPU arrays that supports most of the
AbstractArray interface.  Important differences from the alternative
CudaArray are: (1) a custom memory manager that minimizes the number
of calls to the slow cudaMalloc by reusing already allocated but
garbage collected GPU pointers.  (2) a custom getindex that handles
ranges such as `a[5:10]` as views (with memory shared with the
original array) instead of copies.  DexArrays can be created by
specifying the element type and dimensions or by conversion from
regular Arrays and they can be converted back to regular Arrays (which
involve copying to and from the GPU memory):

    a = DexArray(Float32,2,3)
    b = DexArray(zeros(2,3))
    c = Array(b)

"""
type DexArray{T,N}
    ptr::DexPtr
    dims::NTuple{N,Int}
end

# Aliases:
typealias DexMatrix{T} DexArray{T,2}
typealias DexVector{T} DexArray{T,1}
typealias DexVecOrMat{T} Union{DexVector{T}, DexMatrix{T}}

# Constructors:
DexArray{T,N}(::Type{T}, dims::NTuple{N,Int})=DexArray{T,N}(DexPtr(sizeof(T)*prod(dims)), dims)
DexArray(T::Type, dims::Int...)=DexArray(T,dims)
DexArray(T::Type, d::Integer...)=DexArray(T,convert(Tuple{Vararg{Int}}, d))

# Conversions:
import Base: convert, reshape, vec, unsafe_convert, pointer
# DexArray <- DexArray
convert{T,N}(::Type{DexArray}, x::DexArray{T,N}) = x
convert{T,N}(::Type{DexArray{T}}, x::DexArray{T,N}) = x
convert{T,N}(::Type{DexArray{T,N}}, x::DexArray{T,N}) = x
convert{T,N,S}(::Type{DexArray{T}}, x::DexArray{S,N}) = convert(DexArray{T,N}, x)
convert{T,N,S}(::Type{DexArray{T,N}}, x::DexArray{S,N}) = convert(DexArray{T,N},unsafe_copy!(Array(S, size(x)), 1, x, 1, length(x)))
reshape{T}(a::DexArray{T},dims::Dims)=(if dims==size(a); a; elseif prod(dims)!=length(a); throw(DimensionMismatch()); else; DexArray{T,length(dims)}(a.ptr,dims); end)
reshape(a::DexArray, dims::Int...) = reshape(a, dims)
vec(a::DexArray) = reshape(a, length(a))
# DexArray <- AbstractArray
convert{T,N}(::Type{DexArray}, x::AbstractArray{T,N}) = convert(DexArray{T,N}, x)
convert{T,N,S}(::Type{DexArray{T}}, x::AbstractArray{S,N}) = convert(DexArray{T,N}, x)
convert{T,N,S}(::Type{DexArray{T,N}}, x::AbstractArray{S,N}) = unsafe_copy!(DexArray(T, size(x)), 1, convert(Array{T,N},x), 1, length(x))
# Array <- DexArray
convert{T,N}(::Type{Array}, x::DexArray{T,N}) = convert(Array{T,N}, x)
convert{T,N,S}(::Type{Array{T}}, x::DexArray{S,N}) = convert(Array{T,N}, x)
convert{T,N,S}(::Type{Array{T,N}}, x::DexArray{S,N}) = convert(Array{T,N},unsafe_copy!(Array(S, size(x)), 1, x, 1, length(x)))
# Ptr <- DexArray
unsafe_convert{T}(::Type{Ptr{T}}, a::DexArray) = unsafe_convert(Ptr{T}, pointer(a))
pointer{T}(a::DexArray{T})=convert(Ptr{T}, a.ptr.ptr)
pointer{T}(a::DexArray{T},i)=convert(Ptr{T}, a.ptr.ptr + (i-1)*sizeof(T))

# AbstractArray interface
import Base: eachindex, elsize, eltype, endof, fill!, first, length, linearindexing, ndims, ones, similar, size, stride, zeros
eachindex(a::DexArray) = (1:length(a))
elsize{T}(::DexArray{T}) = sizeof(T)
eltype{T}(::DexArray{T})=T
eltype{T}(::Type{DexArray{T}}) = T
eltype{T,n}(::Type{DexArray{T,n}}) = T
endof(a::DexArray) = length(a)
fill!{T}(a::DexArray{T},x)=(dexfill!(a,T(x),1,length(a));a)
first(a::DexArray) = a[1]
length(a::DexArray)=prod(size(a))
linearindexing(::DexArray)=LinearFast()
ndims{T,N}(a::DexArray{T,N})=N
ones{T}(a::DexArray{T})=fill!(similar(a),one(T))
similar(a::DexArray, T, dims::Dims)      = DexArray(T, dims)
similar(a::DexArray, T, dims::Int...)    = similar(a, T, dims)
similar(a::DexArray, T)                  = similar(a, T, size(a))
similar{T}(a::DexArray{T})               = similar(a, T, size(a))
similar{T}(a::DexArray{T}, dims::Dims)   = similar(a, T, dims)
similar{T}(a::DexArray{T}, dims::Int...) = similar(a, T, dims)
size(a::DexArray)=a.dims
size{T,N}(a::DexArray{T,N},i::Integer)=(if i>N; 1; else; size(a)[i]; end)
stride{T,N}(a::DexArray{T,N},i::Integer)=(if i>N; length(a); else; s=1; for n=1:(i-1); s*=size(a,n); end; s; end)
strides{T,N}(a::DexArray{T,N})=ntuple(n->stride(a,n), N)
zeros{T}(a::DexArray{T})=fill!(similar(a),zero(T))


# Indexing:
import Base: getindex, setindex!

# We will implement indexing ranges as views not copies, if possible (when contiguous).
# For contiguous memory without stride all but the last >1 dimension must be full

# The original getindex(a,i:j...) for AbstractArray copies:
# function _getindex(l::LinearIndexing, A::AbstractArray, I::Union{Real, AbstractArray, Colon}...)
# in abstractarray.jl:487,multidimensional.jl:184.

# which getindex ops does array implement?
# getindex(A::Array, i1::Real)
# getindex(A::Array, i1::Real, i2::Real, I::Real...)
# getindex(A::Array, I::UnitRange{Int})
# getindex(A::Array, c::Colon)
# getindex{T<:Real}(A::Array, I::Range{T})

# Julia #14770
# If I is shorter than ndims(A) but longer than 1 the remaining indices assumed =1
# Also extra 1's at the end of I are ignored

# These two are not sufficient in spite of what the documentation says:
# display goes into an infinite loop!
# getindex{T}(A::DexArray{T}, i::Int)=unsafe_copy!(T[0], 1, A, i, 1)[1]
# setindex!{T}(A::DexArray{T}, v, i::Int)=unsafe_copy!(A, i, T[v], 1, 1)

# First deal with the easy cases: integer indices, a Colon or a UnitRange.

function getindex{T}(A::DexArray{T}, I::Real)
    J = Int(I)
    1 <= J <= length(A) || throw(BoundsError(A,J))
    Dex!(T[0], 1, A, J, 1)[1]
end

function setindex!{T}(A::DexArray{T}, v, I::Real)
    J = Int(I)
    1 <= J <= length(A) || throw(BoundsError(A,J))
    unsafe_copy!(A, J, T[v], 1, 1)
end

function getindex{T}(A::DexArray{T}, I::Real...)
    J = Base.to_indexes(I...)
    @inbounds for j=1:length(J)
        1 <= J[j] <= size(A,j) || throw(BoundsError(A,J))
    end
    i = sub2ind(size(A), J...)
    unsafe_copy!(T[0], 1, A, i, 1)[1]
end

function setindex!{T}(A::DexArray{T}, v, I::Real...)
    J = Base.to_indexes(I...)
    @inbounds for j=1:length(J)
        1 <= J[j] <= size(A,j) || throw(BoundsError(A,J))
    end
    i = sub2ind(size(A), J...)
    unsafe_copy!(A, i, T[v], 1, 1)
end

function getindex{T}(A::DexArray{T}, I::UnitRange)
    1 <= first(I) <= last(I) <= length(A) || throw(BoundsError(A,I))
    off = 1+(first(I)-1)*sizeof(T)
    len = length(I)*sizeof(T)
    ptr = DexPtr(A.ptr, off, len)
    DexArray{T,1}(ptr, (length(I),))
end

function setindex!{T}(A::DexArray{T}, v, I::UnitRange)
    1 <= first(I) <= last(I) <= length(A) || throw(BoundsError(A,I))
    if isa(v,Number)
        dexfill!(A,T(v),first(I),length(I))
    elseif (isa(v,DexArray) || isa(v,Array))
        length(v)==length(I) || throw(DimensionMismatch())
        eltype(v)==T || (v = convert(Array{T},v))
        unsafe_copy!(A,first(I),v,1,length(I))
    else
        throw(MethodError(setindex!, A, v, I))
    end
end

function getindex(A::DexArray, I::Colon)
    reshape(A,length(A))
end

function setindex!{T}(A::DexArray{T}, v, I::Colon)
    if isa(v,Number)
        dexfill!(A,T(v),1,length(A))
    elseif (isa(v,DexArray) || isa(v,Array))
        length(v)==length(A) || throw(DimensionMismatch())
        eltype(v)==T || (v = convert(Array{T},v))
        unsafe_copy!(A,1,v,1,length(A))
    else
        throw(MethodError(setindex!, A, v, I))
    end
end

# TODO: the following getindex, setindex! work for 1 and 2 dimensions only, write general versions.

function getindex{T,N}(A::DexArray{T,N}, I::Union{Real, UnitRange, Colon}...)
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I...)
    B1 = isa(I[1],Colon) ? size(A,1) : length(I[1])
    B2 = isa(I[2],Colon) ? size(A,2) : length(I[2])
    if ncols == 1
        off = 1+(firstindex-1)*sizeof(T)
        len = nrows*sizeof(T)
        ptr = DexPtr(A.ptr, off, len)
        DexArray{T,2}(ptr, (B1,B2))
    else
        B = similar(A, (B1,B2))
        nrows *= sizeof(T); astep *= sizeof(T)
        ccall((:xcopy,libknet8),Void,(Cint,Cint,Cptr,Cint,Cptr,Cint),
              nrows, ncols, pointer(A,firstindex), astep, B, nrows)
        return B
    end
end

function setindex!{T,N}(A::DexArray{T,N}, B, I::Union{Real, UnitRange, Colon}...)
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I...)
    aptr0 = pointer(A, firstindex)
    if isa(B,Number)
        B = T(B)
        if ncols == 1
            if T <: Float32;    ccall((:fill_32,libcudex),Void,(Cint,Cfloat, Ptr{Cfloat}), nelts,B,aptr0)
            elseif T<: Float64; ccall((:fill_64,libcudex),Void,(Cint,Cdouble,Ptr{Cdouble}),nelts,B,aptr0)
            else error("$T not supported"); end
        else
            if T <: Float32;    ccall((:xfill_32,libcudex),Void,(Cint,Cint,Cfloat, Ptr{Cfloat}, Cint),nrows,ncols,B,aptr0,astep)
            elseif T<: Float64; ccall((:xfill_64,libcudex),Void,(Cint,Cint,Cdouble,Ptr{Cdouble},Cint),nrows,ncols,B,aptr0,astep)
            else error("$T not supported"); end
        end
    else
        length(B) == nelts || throw(DimensionMismatch())
        B = convert(DexArray{T},B)
        if ncols == 1
            @cuda(cudart,cudaMemcpyAsync,(Cptr,Cptr,Csize_t,UInt32,Cptr),
                  aptr0, B, nelts*sizeof(T), cudadir(A,B), C_NULL)
        else
            nrows *= sizeof(T); astep *= sizeof(T)
            ccall((:xcopy,libknet8),Void,(Cint,Cint,Cptr,Cint,Cptr,Cint), nrows, ncols, B, nrows, aptr0, astep)
        end
    end
    return A
end

function indexparams{T,N}(A::DexArray{T,N}, I::Union{Real, UnitRange, Colon}...)
    N > 2 && error("setindex for ndims > 2 not implemented yet")
    skipped = false
    nrows = nelts = 1
    subs1 = ones(Int,2)
    astep = length(A)
    for i=1:length(I)
        Ii = I[i]
        Ai = size(A,i)
        if isa(Ii, Colon)
            Li = Ai
            subs1[i] = 1
        elseif isa(Ii, Real)
            1 <= Ii <= Ai || throw(DimensionMismatch())
            Li = 1
            subs1[i] = Int(Ii)
        else
            1 <= first(Ii) <= last(Ii) <= Ai || throw(DimensionMismatch())
            Li = length(Ii)
            subs1[i] = first(Ii)
        end
        nelts *= Li
        if !skipped
            nrows *= Li
            if Li < Ai
                skipped = true
                astep = stride(A,i+1)
            end
        end
    end
    ncols = div(nelts, nrows)
    firstindex = sub2ind(size(A),subs1...)
    return (nelts,nrows,ncols,firstindex,astep)
end


# Concatenation:
import Base: hcat, vcat, cat

# Benchmarks in μs for hcat and vcat: a=rand(1000,1000) v=rand(1000), t=v'
#		cpu	gpu	g->c->g	vkernel
# hcat(a,a)	2350	225	16160
# hcat(a,v)	1230	115	6490
# hcat(v,a)	1220	120	6490
# hcat(v,v)	3.53	12.53	48.49
# vcat(a,a)	2630	10980	16590	665
# vcat(a,t)	1350	10860	6550	338
# vcat(t,a)	1360	10850	6570	338
# vcat(v,v)	2.13	12.33	45.40	13.58

# setindex! methods called by hcat/vcat:
# hcat(v,v): I = (Colon(),1:1) I = (Colon(),2:2)
# vcat(v,v): uses single index
# hcat(m,m): I = (Colon(),1:5) I = (Colon(),6:10)
# vcat(m,m): I = (1:3,Colon()) I = (4:6,Colon())

function hcat{T}(a::DexVecOrMat{T}, b::DexVecOrMat{T})
    size(a,1)==size(b,1) || throw(DimensionMismatch())
    c1 = size(a,1)
    c2 = size(a,2) + size(b,2)
    c = DexArray(T, (c1,c2))
    c[:,1:size(a,2)] = a
    c[:,1+size(a,2):end] = b
    return c
end

function vcat{T}(a::DexVector{T}, b::DexVector{T})
    c = DexArray(T, length(a)+length(b))
    c[1:length(a)] = a
    c[1+length(a):end] = b
    return c
end

function vcat{T}(a::DexVecOrMat{T}, b::DexVecOrMat{T})
    size(a,2)==size(b,2) || throw(DimensionMismatch())
    c1 = size(a,1) + size(b,1)
    c2 = size(a,2)
    c = DexArray(T, (c1,c2))
    c[1:size(a,1),:] = a
    c[1+size(a,1):end,:] = b
    return c
end

function cat{T}(d, a::DexVecOrMat{T}, b::DexVecOrMat{T})
    if     d==1; vcat(a,b)
    elseif d==2; hcat(a,b)
    else error("cat($d) not implemented.")
    end
end


# Utilities:

# Generalizing low level copy using linear indexing to/from gpu arrays:
# copy!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)
# Note that this is an unsafe operation, no argument or bounds checking performed.
# Defined in Base:
# unsafe_copy!{T}(dest::Ptr{T}, src::Ptr{T}, n) at array.jl:73
# unsafe_copy!{T}(dest::Array{T,N}, doffs, src::Array{T,N}, soffs, n) at array.jl:79

import Base: unsafe_copy!, copy

function unsafe_copy!{T}(dest::Union{DexArray{T},Array{T}}, doffs, src::Union{DexArray{T},Array{T}}, soffs, n; stream=C_NULL)
    @cuda(cudart,cudaMemcpyAsync,(Cptr,Cptr,Csize_t,UInt32,Cptr),
          pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), cudadir(dest,src), stream)
    return dest
end

function cudadir(a,b)
    deva = isa(a,DexArray) && a.ptr.dev >= 0
    devb = isa(b,DexArray) && b.ptr.dev >= 0
    if !deva && !devb; return 0
    elseif deva && !devb; return 1
    elseif !deva && devb; return 2
    elseif deva && devb;  return 3
    end
end

copy(a::DexArray)=unsafe_copy!(similar(a),1,a,1,length(a))

# Efficient fill:
for S in (32,64); T = Symbol("Float$S"); F = "fill_$S"
    @eval function dexfill!(a::DexArray{$T},v::$T,off,len)
        ccall(($F,$libcudex),Void,(Cint,$T,Ptr{$T}),len,v,pointer(a,off))
    end
end

# # AutoGrad functions:
# import AutoGrad: zeroslike, sum_outgrads, OneHot, unary_nd, indexed_function, isequivalent
# zeroslike(a::DexArray)=zeros(a)
# sum_outgrads{T}(a::DexArray{T},b::DexArray{T})=(a+b)
# sum_outgrads(a::DexArray,b::OneHot)=setindex!(a,sum_outgrads(getindex(a,b.index...),b.value),b.index...)
# unary_nd(f, x::DexArray, eps) = reshape(eltype(x)[unary_nd(indexed_function(f, x, i), x[i], eps) for i in 1:length(x)], size(x))
# isequivalent(x::Union{DexArray,AbstractArray}, y::Union{DexArray,AbstractArray}; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))

# Hack for printing without copying the whole DexArray and without inheriting AbstractArray:
import Base: display, summary
type DexDisplay{T,N} <: AbstractArray{T,N}; a::DexArray{T,N}; end
getindex(a::DexDisplay, i...) = getindex(a.a, i...)
size(a::DexDisplay) = size(a.a)
summary(a::DexDisplay) = summary(a.a)
summary(a::DexArray) = string(Base.dims2string(size(a)), " ", typeof(a))
display(a::DexArray) = display(DexDisplay(a))
AutoGrad._dbg(a::DexArray) = "K$(join([AutoGrad.id2(a),size(a)...],'_'))"

# curand functions:

import Base: rand!
rand!(a::DexArray{Float32})=(@cuda(curand,curandGenerateUniform,(Cptr,Ptr{Cfloat},Csize_t),rng(),a,length(a)); a)
rand!(a::DexArray{Float64})=(@cuda(curand,curandGenerateUniformDouble,(Cptr,Ptr{Cdouble},Csize_t),rng(),a,length(a)); a)

let RNG=0
global rng
function rng(init=false)
    if RNG==0 || init
        ptr = Cptr[0]
        # CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
        @cuda(curand,curandCreateGenerator,(Cptr,Cint),ptr,100)
        RNG = ptr[1]
    end
    return RNG
end
end
