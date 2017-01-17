# cudnn wrapper

"""

`conv_f(x,w;kwargs...)` executes convolutions or cross-correlations
using filters specified with `w` over tensor `x`.
Here is a description of all available keyword arguments:
* padding: the number of extra zeros implicitly concatenated at the start and at the end of each dimension. Default=floor((filterSize-1)/2) which preserves the input size when filterSize is odd and stride=1.
* stride: the number of elements to slide to reach the next filtering window. Default=1.
* upscale: upscale factor for each dimension. Default=1.
* mode: 0 for convolution and 1 for cross-correlation.  Default=0.
* alpha: can be used to scale the result. Default=1.
* algo: specifies which convolution algorithm shoud be used to compute the results. Default=0. See the CUDNN User Guide for details.
* workSpace: data pointer to GPU memory to a workspace needed to able to execute the specified algorithm. Default=C_NULL.
* workSpaceSizeInBytes: the size in bytes of the provided workSpace. Default=0.
* handle: handle to a previously created cuDNN context. Default=Dex allocated context.

"""
function conv_f{T}(x::DexArray{T},w::DexArray{T};
                  handle=cudnnhandle, alpha=one(T), beta=zero(T),
                  algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    y = similar(x, cdims(x,w;o...))
    @cuda(cudnn, cudnnConvolutionForward,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,UInt32,Cptr,Csize_t,Ptr{T},Cptr,Ptr{T}),
          handle,Ref(alpha),TD(x),x,FD(w),w,CD(x,w;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),TD(y),y)
    return y
end

"""

`conv_bx(x,w,dy;kwargs...)` This function computes the convolution gradient with
respect to the output tensor, returning results in gradDesc.

"""
function conv_bx{T}(x::DexArray{T},w::DexArray{T},dy::DexArray{T};
                   handle=cudnnhandle, alpha=one(T), beta=zero(T),
                   algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    dx = similar(x)
    @cuda(cudnn,cudnnConvolutionBackwardData,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(alpha),FD(w),w,TD(dy),dy,CD(x,w;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),TD(dx),dx)
    return dx
end

"""

`conv_bw(x,w,dy;kwargs...)` This function computes the convolution gradient with
 respect to filter coefficients, returning results in gradDesc.

"""
function conv_bw{T}(x::DexArray{T},w::DexArray{T},dy::DexArray{T};
                   handle=cudnnhandle, alpha=one(T), beta=zero(T),
                   algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    dw = similar(w)
    @cuda(cudnn,cudnnConvolutionBackwardFilter,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(alpha),TD(x),x,TD(dy),dy,CD(x,w;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),FD(dw),dw)
    return dw
end

"""

`pool_f(x;kwargs...)` computes pooling of input values (i.e., the
maximum or average of several adjacent values) to produce an output
with smaller height and/or width.

Here is a description of all available keyword arguments:
* window: the pooling window size for each dimension. Default=2.
* padding: the number of extra zeros implicitly concatenated at the start and at the end of each dimension. Default=0.
* stride: the number of elements to slide to reach the next pooling window. Default=same as window.
* mode: 0 for max, 1 for average including padded values, 2 for average excluding padded values.  Default=0.
* maxpoolingNanOpt: Nan numbers are not propagated if 0, they are propagated if 1. Default=0.
* alpha: can be used to scale the result. Default=1.
* handle: Handle to a previously created cuDNN context. Default=Dex allocated context.

"""

function pool_f{T}(x::DexArray{T}; handle=cudnnhandle, alpha=one(T), beta=zero(T), o...)
    y = similar(x, pdims(x; o...))
    @cuda(cudnn, cudnnPoolingForward,
          (Cptr, Cptr,      Ptr{T},    Cptr,Ptr{T},Ptr{T},   Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(alpha),TD(x),x,    Ref(beta),TD(y),y)
    return y
end

function pool_b{T}(x::DexArray{T},y::DexArray{T},dy::DexArray{T};
                  handle=cudnnhandle, alpha=one(T), beta=zero(T), o...)
    dx = similar(x)
    @cuda(cudnn,cudnnPoolingBackward,
          (Cptr,Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Ptr{T},Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(alpha),TD(y),y,TD(dy),dy,TD(x),x,Ref(beta),TD(dx),dx)
    return dx
end

"""

`dropout_f(x;kwargs...)` computes the forward dropout operation over input values
to produce an output.

Here is a description of all available keyword arguments:
* window: the pooling window size for each dimension. Default=2.
* padding: the number of extra zeros implicitly concatenated at the start and at the end of each dimension. Default=0.
* stride: the number of elements to slide to reach the next pooling window. Default=same as window.
* mode: 0 for max, 1 for average including padded values, 2 for average excluding padded values.  Default=0.
* maxpoolingNanOpt: Nan numbers are not propagated if 0, they are propagated if 1. Default=0.
* alpha: can be used to scale the result. Default=1.
* handle: Handle to a previously created cuDNN context. Default=Dex allocated context.

"""

function dropout_f{T}(x::DexArray, dropout;
                      handle=cudnnhandle)
    y = similar(x)
    @cuda(cudnn, cudnnDropoutForward,
          (Cptr,Cptr,        Cptr,Ptr{T}
          handle,DD(dropout),TD(x)),x,

end

# cudnn datatype
# an enumerated type indicating the data type to which a tensor descriptor
# or filter descriptor refers.
DT(::DexArray{Float32})=UInt32(0)
DT(::DexArray{Float64})=UInt32(1)
DT(::DexArray{Float16})=UInt32(2)

# accquire dimension
function cdims{T,N}(x::DexArray{T,N},w::DexArray{T,N}; padding=0, stride=1, o...)
    ntuple(N) do i
        if i < N-1
            pi = if isa(padding,Number); padding; else padding[i]; end
            si = if isa(stride,Number); stride; else stride[i]; end
            1 + div(size(x,i) - size(w,i) + 2*pi, si)
        elseif i == N-1
            size(w,N)
        else # i == N
            size(x,N)
        end
    end
end

function pdims{T,N}(x::DexArray{T,N}; window=2, padding=0, stride=window, o...)
    ntuple(N) do i
        if i < N-1
            wi = (if isa(window,Number); window; else window[i]; end)
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            1 + div(size(x,i) + 2*pi - wi, si)
        else
            size(x,i)
        end
    end
end

# cudnn descriptors

type TD; ptr
    function TD(a::DexArray)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
        n = ndims(a)
        sz = [Cint(size(a,n-i+1)) for i=1:n]    # why dim reversed?
        st = [Cint(stride(a,n-i+1)) for i=1:n]
        @cuda(cudnn,cudnnSetTensorNdDescriptor,
              (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
              d[1], DT(a), n, sz, st)
        td = new(d[1])
        finalizer(td, x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.ptr))
        return td
    end
end

type FD; ptr
    function FD(a::DexArray)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateFilterDescriptor,(Ptr{Cptr},),d)
        n = ndims(a)
        sz = [Cint(size(a,n-i+1)) for i=1:n]
        @cuda(cudnn,cudnnSetFilterNdDescriptor,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
              d[1], DT(a), 0,     n,   sz)
        fd = new(d[1])
        finalizer(fd, x->@cuda(cudnn,cudnnDestroyFilterDescriptor,(Cptr,),x.ptr))
        return fd
    end
end

type CD; ptr
    function CD(x::DexArray,w::DexArray; padding=0, stride=1, upscale=1, mode=0)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateConvolutionDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        @cuda(cudnn,cudnnSetConvolutionNdDescriptor,
              (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
              d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,DT(x))
        cd = new(d[1])
        finalizer(cd, x->@cuda(cudnn,cudnnDestroyConvolutionDescriptor,(Cptr,),x.ptr))
        return cd
    end
end

type PD; ptr
    function PD(x::DexArray; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreatePoolingDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        @cuda(cudnn,cudnnSetPoolingNdDescriptor,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
              d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        pd = new(d[1])
        finalizer(pd, x->@cuda(cudnn,cudnnDestroyPoolingDescriptor,(Cptr,),x.ptr))
        return pd
    end
end

type DD; ptr
    function DD(dropout;handle=cudnnhandle)
        d = Cptr[0]
        @cuda(cudnn, cudnnCreateDropoutDescriptor, (Ptr{Cptr},),d)
        statessize_p = Csize_t[0]
        @cuda(cudnn, cudnnDropoutGetStatesSize, (Cptr,Ptr{Csize_t}), handle,statessize_p)
        statessize = statessize_p[1]
        states = DexArray(Int8, Int(statessize))
        @cude(cudnn, cudnnSetDropoutDescriptor,
              (Cptr,Cptr, Cfloat,Cptr,  Cint,Culonglong),
              d[1],handle,Cfloat(dropout),state,statessize, 0)
        dd = new(d[1])
        finalizer(dd, x->@cuda(cudnn, cudnnDestroyDropoutDescriptor, (Cptr,),x.ptr)
        return dd
    end
end

import Base: unsafe_convert
unsafe_convert(::Type{Cptr}, td::TD)=td.ptr
unsafe_convert(::Type{Cptr}, fd::FD)=fd.ptr
unsafe_convert(::Type{Cptr}, cd::CD)=cd.ptr
unsafe_convert(::Type{Cptr}, pd::PD)=pd.ptr

function cdsize(w, nd)
    if isa(w,Integer)
        fill(Cint(w),nd)
    elseif length(w)==nd
        [ Cint(w[nd-i+1]) for i=1:nd ]
    else
        throw(DimensionMismatch())
    end
end

# convolution padding size that preserves the input size when filter size is odd and stride=1
padsize(w)=ntuple(i->div(size(w,i)-1,2), ndims(w)-2)
