include("gpu.jl")

import Base: .+, .-, .*, ./, .^, max, min, .==, .>, .>=, .<, .<=, +, -, *, /, \

basic_opts = [
("add",".+","s+xi"),
("sub",".-","s-xi"),
("mul",".*","s*xi"),
("div","./","s/xi"),
("pow",".^","pow(s,xi)"),
("max","max","(xi>s?xi:s)"),
("min","min","(xi<s?xi:s)"),
("eq",".==","xi==s"),
("gt",".>","xi>s"),
("ge",".>=","xi>=s"),
("lt",".<","xi<s"),
("le",".<=","xi<=s"),
]

max{T<:Real,S<:Real}(a::DexArray{T},s::S)=max(T(s),a)
max{T<:Real,S<:Real}(s::S,a::DexArray{T})=max(T(s),a)
min{T<:Real,S<:Real}(a::DexArray{T},s::S)=min(T(s),a)
min{T<:Real,S<:Real}(s::S,a::DexArray{T})=min(T(s),a)
max{T<:Real,S<:Number}(a::DexArray{T},s::S)=max(T(s),a)
max{T<:Real,S<:Number}(s::S,a::DexArray{T})=max(T(s),a)
min{T<:Real,S<:Number}(a::DexArray{T},s::S)=min(T(s),a)
min{T<:Real,S<:Number}(s::S,a::DexArray{T})=min(T(s),a)
(+)(a::DexArray{Bool},s::Bool)=(.+)(s,a)
(+)(s::Bool,a::DexArray{Bool})=(.+)(s,a)
(-)(a::DexArray{Bool},s::Bool)=(.+)(-s,a)
(-)(s::Bool,a::DexArray{Bool})=(.-)(s,a)
.^(x::Base.Irrational{:e}, a::DexArray)=.^(float(x),a)

# For array,scalar we can get some for free:
# Only type corrected number,array need implementing for basic arithmetic:
(.+){T}(a::DexArray{T},s::Number)=(.+)(T(s),a)
(.+){T}(s::Number,a::DexArray{T})=(.+)(T(s),a)
(.-){T}(a::DexArray{T},s::Number)=(.+)(T(-s),a)
(.-){T}(s::Number,a::DexArray{T})=(.-)(T(s),a)
(.*){T}(a::DexArray{T},s::Number)=(.*)(T(s),a)
(.*){T}(s::Number,a::DexArray{T})=(.*)(T(s),a)
(./){T}(a::DexArray{T},s::Number)=(.*)(T(1/s),a)
(./){T}(s::Number,a::DexArray{T})=(./)(T(s),a)
#(.^){T}(a::DexArray{T},s::Number) # cannot convert to an s,a operation
(.^){T}(s::Number,a::DexArray{T})=(.^)(T(s),a)
max{T}(a::DexArray{T},s::Number)=max(T(s),a)
max{T}(s::Number,a::DexArray{T})=max(T(s),a)
min{T}(a::DexArray{T},s::Number)=min(T(s),a)
min{T}(s::Number,a::DexArray{T})=min(T(s),a)

.=={T}(a::DexArray{T},s::Number)=(T(s).==a)
.=={T}(s::Number,a::DexArray{T})=(T(s).==a)
.>{T}(a::DexArray{T},s::Number)=(T(s).<a)
.>{T}(s::Number,a::DexArray{T})=(T(s).>a)
.>={T}(a::DexArray{T},s::Number)=(T(s).<=a)
.>={T}(s::Number,a::DexArray{T})=(T(s).>=a)
.<{T}(a::DexArray{T},s::Number)=(T(s).>a)
.<{T}(s::Number,a::DexArray{T})=(T(s).<a)
.<={T}(a::DexArray{T},s::Number)=(T(s).>=a)
.<={T}(s::Number,a::DexArray{T})=(T(s).<=a)

# familiar aliases for broadcasting operations of array & scalar (#7226):
(+){T}(a::DexArray{T},s::Number)=(.+)(T(s),a)
(+){T}(s::Number,a::DexArray{T})=(.+)(T(s),a)
(-){T}(a::DexArray{T},s::Number)=(.+)(T(-s),a)
(-){T}(s::Number,a::DexArray{T})=(.-)(T(s),a)
(*){T}(a::DexArray{T},s::Number)=(.*)(T(s),a)
(*){T}(s::Number,a::DexArray{T})=(.*)(T(s),a)
(/){T}(a::DexArray{T},s::Number)=(.*)(T(1/s),a)
(\){T}(s::Number,a::DexArray{T})=(.*)(T(1/s),a)
#(/){T}(s::Number,a::DexArray{T})=(.*)(T(1/s),a) # not defined in base
#(^){T}(a::DexArray{T},s::Number)=(.^)(a,T(s)) # linalg
#(^){T}(s::Number,a::DexArray{T})=(.^)(T(s),a) # linalg

function basic_def(f, j=f, o...)
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_01"
        @eval begin
            function $J(s::$T,x::DexArray{$T})
                y = similar(x)
                ccall(($F,$libxflow),Void,(Cint,$T,Ptr{$T},Ptr{$T}),length(y),s,x,y)
                return y
            end
        end
    end
end

#if isdefined(:libknet8)
    for f in basic_opts
        isa(f,Tuple) || (f=(f,))
        basic_def(f...)
    end
#end
