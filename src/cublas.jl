# cublas wrapper

import Base: *, A_mul_B!
import Base: A_mul_Bt, A_mul_Bt!, A_mul_Bc, A_mul_Bc!
import Base: At_mul_B, At_mul_B!, Ac_mul_B, Ac_mul_B!
import Base: At_mul_Bt, At_mul_Bt!, Ac_mul_Bc, Ac_mul_Bc!
import Base.LinAlg.BLAS: gemm!

A_mul_B!{T}(C::DexMatrix{T}, A::DexMatrix{T}, B::DexMatrix{T})=gemm!('N','N',one(T),A,B,zero(T),C)
(*){T}(A::DexMatrix{T},B::DexMatrix{T})=A_mul_B!(similar(A,(size(A,1),size(B,2))),A,B)

A_mul_Bt!{T}(C::DexMatrix{T}, A::DexMatrix{T}, B::DexMatrix{T})=gemm!('N','T',one(T),A,B,zero(T),C)
A_mul_Bt{T}(A::DexMatrix{T}, B::DexMatrix{T})=A_mul_Bt!(similar(A,(size(A,1),size(B,1))),A,B)
A_mul_Bc!{T<:Real}(C::DexMatrix{T}, A::DexMatrix{T}, B::DexMatrix{T})=A_mul_Bt!(C,A,B)
A_mul_Bc{T<:Real}(A::DexMatrix{T}, B::DexMatrix{T})=A_mul_Bt(A,B)

At_mul_B!{T}(C::DexMatrix{T}, A::DexMatrix{T}, B::DexMatrix{T})=gemm!('T','N',one(T),A,B,zero(T),C)
At_mul_B{T}(A::DexMatrix{T}, B::DexMatrix{T})=At_mul_B!(similar(A,(size(A,2),size(B,2))),A,B)
Ac_mul_B!{T<:Real}(C::DexMatrix{T}, A::DexMatrix{T}, B::DexMatrix{T})=At_mul_B!(C,A,B)
Ac_mul_B{T<:Real}(A::DexMatrix{T}, B::DexMatrix{T})=At_mul_B(A,B)

At_mul_Bt!{T}(C::DexMatrix{T}, A::DexMatrix{T}, B::DexMatrix{T})=gemm!('T','T',one(T),A,B,zero(T),C)
At_mul_Bt{T}(A::DexMatrix{T}, B::DexMatrix{T})=At_mul_Bt!(similar(A,(size(A,2),size(B,2))),A,B)
Ac_mul_Bc!{T<:Real}(C::DexMatrix{T}, A::DexMatrix{T}, B::DexMatrix{T})=At_mul_Bt!(C,A,B)
Ac_mul_Bc{T<:Real}(A::DexMatrix{T}, B::DexMatrix{T})=At_mul_Bt(A,B)


function gemm!{T}(transA::Char, transB::Char, alpha::Number, A::DexArray{T}, B::DexArray{T}, beta::Number, C::DexArray{T})
    cublasop(c::Char)=(if c=='N'; 0; elseif c=='T'; 1; elseif c=='C'; 2; else error("Unknown cublas op $c"); end)
    cublascheck(x) = (x==0 || error("CUBLAS error $x"))
    size2(x,i)=(if ndims(x)<=2; size(x,i); elseif i==1; div(length(x),size(x,ndims(x))); elseif i==2; size(x,ndims(x)); else 1; end)
    if transA == 'N'
        m=size2(A,1); k=size2(A,2)
    else
        m=size2(A,2); k=size2(A,1)
    end
    if transB == 'N'
        n=size2(B,2); k==size2(B,1) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    else
        n=size2(B,1); k==size2(B,2) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    end
    (m == size2(C,1) && n == size(C,2)) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    transa = cublasop(transA); transb = cublasop(transB)
    alpha = T(alpha); beta = T(beta)
    lda = size2(A,1); ldb = size2(B,1); ldc = size2(C,1)
    if T<:Float32
        cublascheck(ccall((:cublasSgemm_v2, "libcublas"), UInt32, (Ptr{Void}, UInt32, UInt32, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), cublashandle, transa, transb, m, n, k, [alpha], A, lda, B, ldb, [beta], C, ldc))
    elseif T<:Float64
        cublascheck(ccall((:cublasDgemm_v2, "libcublas"), UInt32, (Ptr{Void}, UInt32, UInt32, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), cublashandle, transa, transb, m, n, k, [alpha], A, lda, B, ldb, [beta], C, ldc))
    else
        error("CUBLAS does not support $T")
    end
    return C
end
