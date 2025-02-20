module BeamLib
import Base.convert
using LinearAlgebra

export PhasedArray, PhasedArrayND, PhasedArray1D, PhasedArray2D, PhasedArray3D, ArrayManifold, NestedArray, steerphi, steerk, 
        dsb_weights, dsb_weights_k, mvdr_weights, mvdr_weights_k,
        mpdr_weights, mpdr_weights_k, capon_weights, capon_weights_k,
        whitenoise, diffnoise

c_0 = 299792458.0
@enum WaveDirection begin
    Incoming = -1
    Outgoing = 1
end

# generic phased arrays
abstract type PhasedArray end

# phased arrays with only isotropic radiators as elements 
# perfect for quick evaluation of array factors
abstract type PhasedArrayND <: PhasedArray end

# placeholder for future implementations
# TODO: implement
abstract type ArrayManifold <: PhasedArray end

struct PhasedArray1D <: PhasedArrayND
    elements::Vector{Tuple{<:Number}}
end

PhasedArray1D(elements::Tuple{<:Number}...) = PhasedArray1D([e for e in elements])
PhasedArray1D(elements::Vector{<:Number}) = PhasedArray1D(map((x) -> (x,), elements))
PhasedArray1D(elements::Number...) = PhasedArray1D([Tuple(e) for e in elements])

struct PhasedArray2D <: PhasedArrayND
    elements::Vector{Tuple{<:Number, <:Number}}
end

PhasedArray2D(elements::Tuple{<:Number, <:Number}...)= PhasedArray2D([e for e in elements])
PhasedArray2D(x::PhasedArray1D)= PhasedArray2D([(e[1], Base.convert(typeof(e[1]),0)) for e in x.elements])
Base.convert(::Type{PhasedArray2D}, x::PhasedArray1D) = PhasedArray2D(x)

struct PhasedArray3D <: PhasedArrayND
    elements::Vector{Tuple{<:Number, <:Number, <:Number}}
end

PhasedArray3D(elements::Tuple{<:Number, <:Number, <:Number}...) = PhasedArray3D([e for e in elements])
PhasedArray3D(x::PhasedArray1D) = PhasedArray3D([(e[1], Base.convert(typeof(e[1]),0), Base.convert(typeof(e[1]),0)) for e in x.elements])
PhasedArray3D(x::PhasedArray2D) = PhasedArray3D([(e[1], e[2], Base.convert(typeof(e[1]),0)) for e in x.elements])
Base.convert(::Type{PhasedArray3D}, x::PhasedArray1D) = PhasedArray3D(x)
Base.convert(::Type{PhasedArray3D}, x::PhasedArray2D) = PhasedArray3D(x)

struct NestedArray <: PhasedArray
    subarrays::Vector{<:PhasedArray}
    elements::PhasedArray3D
end

function steerphi(x::PhasedArray3D, f, ϕ, θ=1/2π; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    ζ = [cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)] 
    ζ = ζ*Int(direction) # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2) + α[3].*getindex.(x.elements, 3)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

steerphi(x::PhasedArray2D, f, ϕ, θ=1/2π; fs=nothing, c=c_0, direction::WaveDirection=Incoming) = steerphi(convert(PhasedArray3D, x), f, ϕ, θ; fs=fs, c=c, direction=direction)
steerphi(x::PhasedArray1D, f, ϕ, θ=1/2π; fs=nothing, c=c_0, direction::WaveDirection=Incoming) = steerphi(convert(PhasedArray3D, x), f, ϕ, θ; fs=fs, c=c, direction=direction)

function steerphi(x::NestedArray, f, ϕ, θ=1/2π; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    v = steerphi(x.elements, f, ϕ, θ, fs=fs, c=c, direction=direction)
    return reduce(vcat, Tuple(v[i]*steerphi(s, f, ϕ, θ, fs=fs, c=c, direction=direction) for (i,s) in enumerate(x.subarrays)))
end

function steerk(x::PhasedArray3D, f, kx, ky=0, kz=0; fs=nothing, c=c_0)
    k = 2π*f/c # wavenumber
    ζ = [kx/k, ky/k, kz/k] # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2) + α[3].*getindex.(x.elements, 3)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

steerk(x::PhasedArray2D, f, kx, ky=0, kz=0; fs=nothing, c=c_0) = steerk(convert(PhasedArray3D, x), f, kx, ky, kz; fs=fs, c=c)
steerk(x::PhasedArray1D, f, kx, ky=0, kz=0; fs=nothing, c=c_0) = steerk(convert(PhasedArray3D, x), f, kx, ky, kz; fs=fs, c=c)

function dsb_weights(x::PhasedArray, f, ϕ, θ=1/2π; fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)/length(x.elements)
end

function dsb_weights_k(x::PhasedArray, f, kx, ky=0, kz=0; fs=nothing,  c=c_0)
    return steerk(x, f, kx, ky, kz; fs=fs, c=c)/length(x.elements)
end

function mvdr_weights(x::PhasedArray, Snn, f, ϕ, θ=1/2π; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    v = steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

function mvdr_weights_k(x::PhasedArray, Snn, f, kx, ky=0, kz=0; fs=nothing, c=c_0)
    v = steerk(x, f, kx, ky, kz; fs=fs, c=c)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

mpdr_weights(x::PhasedArray, Sxx, f, ϕ, θ=1/2π; fs=nothing, c=c_0, direction::WaveDirection=Incoming) = mvdr_weights(x, Sxx, f, ϕ, θ; fs=fs, c=c, direction=direction)
mpdr_weights_k(x::PhasedArray, Sxx, f, kx, ky=0, kz=0; fs=nothing, c=c_0) = mvdr_weights_k(x, Sxx, f, kx, ky, kz; fs=fs, c=c)

const capon_weights = mpdr_weights
const capon_weights_k = mpdr_weights_k

function whitenoise(x::PhasedArrayND, σ²)
    return σ²*I(length(x.elements))
end

function diffnoise(x::PhasedArrayND, σ², f, c=c_0)
    ω = 2π*f
    k = ω/c
    p(x, i) = [e for e in x.elements[i]] 
    si(x) = sinc(x/π)
    Γ(x, n, m, k) = si(k*norm(p(x,m)-p(x,n)))
    n = 1:length(x.elements)
    return σ²*Γ.(Ref(x), n, n', Ref(k))
end

end