module BeamLib
import Base.convert
using LinearAlgebra
using PhysicalConstants.CODATA2018: c_0

export PhasedArray, PhasedArray1D, PhasedArray2D, PhasedArray3D, steerphi, steerk, 
        dsb_weights, dsb_weights_k, mvdr_weights, mvdr_weights_k,
        mpdr_weights, mpdr_weights_k, capon_weights, capon_weights_k,
        whitenoise, diffnoise

@enum WaveDirection begin
    Incoming = -1
    Outgoing = 1
end

abstract type PhasedArray end

struct PhasedArray1D <: PhasedArray
    elements::Vector{Tuple{Number}}
end

PhasedArray1D(elements::Tuple{Number}...) = PhasedArray1D{T}([e for e in elements])
PhasedArray1D(elements::Vector{Number}) = PhasedArray1D{T}(map((x) -> (x,), elements))
PhasedArray1D(elements::T...) = PhasedArray1D{T}([Tuple(e) for e in elements])

struct PhasedArray2D <: PhasedArray
    elements::Vector{Tuple{Number, Number}}
end

PhasedArray2D(elements::Tuple{Number, Number}...)= PhasedArray2D{Number}([e for e in elements])
PhasedArray2D(x::PhasedArray1D)= PhasedArray2D([(e[1], Base.convert(typeof(e[1]),0)) for e in x.elements])
Base.convert(::Type{PhasedArray2D}, x::PhasedArray1D) = PhasedArray2D(x)

struct PhasedArray3D <: PhasedArray
    elements::Vector{Tuple{Number, Number, Number}}
end

PhasedArray3D(elements::Tuple{Number, Number, Number}...) = PhasedArray3D([e for e in elements])
PhasedArray3D(x::PhasedArray1D) = PhasedArray3D([(e[1], Base.convert(typeof(e[1]),0), Base.convert(typeof(e[1]),0)) for e in x.elements])
PhasedArray3D(x::PhasedArray2D) = PhasedArray3D([(e[1], e[2], Base.convert(typeof(e[1]),0)) for e in x.elements])
Base.convert(::Type{PhasedArray3D}, x::PhasedArray1D) = PhasedArray3D(x)
Base.convert(::Type{PhasedArray3D}, x::PhasedArray2D) = PhasedArray3D(x)

function steerphi(x::PhasedArray1D, f, ϕ; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    ζ = cos(ϕ)
    ζ = ζ*Int(direction) # propagation direction
    α = ζ/c # slowness vector
    Δ = α.*getindex.(x.elements, 1)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

function steerphi(x::PhasedArray2D, f, ϕ, θ; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    ζ = [cos(ϕ)*sin(θ), sin(ϕ)*sin(θ)] 
    ζ = ζ*Int(direction) # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

function steerphi(x::PhasedArray3D, f, ϕ, θ; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    ζ = [cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)] 
    ζ = ζ*Int(direction) # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2) + α[3].*getindex.(x.elements, 3)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

function steerk(x::PhasedArray1D, f, kx; fs=nothing, c=c_0)
    k = 2π*f/c # wavenumber
    ζ = kx/k # propagation direction
    α = ζ/c # slowness vector
    Δ = α.*getindex.(x.elements, 1)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

function steerk(x::PhasedArray2D, f, kx, ky; fs=nothing, c=c_0)
    k = 2π*f/c # wavenumber
    ζ = [kx/k, ky/k] # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

function steerk(x::PhasedArray3D, f, kx, ky, kz; fs=nothing, c=c_0)
    k = 2π*f/c # wavenumber
    ζ = [kx/k, ky/k, kz/k] # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2) + α[3].*getindex.(x.elements, 3)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

function dsb_weights(x::PhasedArray1D, f, ϕ; fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f, ϕ; fs=fs, c=c, direction=direction)/length(x.elements)
end

function dsb_weights_k(x::PhasedArray1D, f, kx; fs=nothing,  c=c_0)
    return steerk(x, f, kx; fs=fs, c=c)/length(x.elements)
end

function dsb_weights(x::PhasedArray2D, f, ϕ, θ; fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)/length(x.elements)
end

function dsb_weights_k(x::PhasedArray2D, f, kx, ky; fs=nothing,  c=c_0)
    return steerk(x, f, kx, ky; fs=fs, c=c)/length(x.elements)
end

function dsb_weights(x::PhasedArray3D, f, ϕ, θ; fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)/length(x.elements)
end

function dsb_weights_k(x::PhasedArray3D, f, kx, ky, kz; fs=nothing,  c=c_0)
    return steerk(x, f, kx, ky, kz; fs=fs, c=c)/length(x.elements)
end

function mvdr_weights(x::PhasedArray1D, Snn, f, ϕ; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    v = steerphi(x, f, ϕ; fs=fs, c=c, direction=direction)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

function mvdr_weights_k(x::PhasedArray1D, Snn, f, kx; fs=nothing, c=c_0)
    v = steerk(x, f, kx; fs=fs, c=c)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

function mvdr_weights(x::PhasedArray2D, Snn, f, ϕ, θ; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    v = steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

function mvdr_weights_k(x::PhasedArray2D, Snn, f, kx, ky; fs=nothing, c=c_0)
    v = steerk(x, f, kx, ky; fs=fs, c=c)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

function mvdr_weights(x::PhasedArray3D, Snn, f, ϕ, θ; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    v = steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

function mvdr_weights_k(x::PhasedArray3D, Snn, f, kx, ky, kz; fs=nothing, c=c_0)
    v = steerk(x, f, kx, ky, kz; fs=fs, c=c)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

mpdr_weights(x::PhasedArray1D, Sxx, f, ϕ; fs=nothing, c=c_0, direction::WaveDirection=Incoming) = mvdr_weights(x, Sxx, f, ϕ; fs=fs, c=c, direction=direction)
mpdr_weights_k(x::PhasedArray1D, Sxx, f, kx; fs=nothing, c=c_0) = mvdr_weights_k(x, Sxx, f, kx; fs=fs, c=c)
mpdr_weights(x::PhasedArray2D, Sxx, f, ϕ, θ; fs=nothing, c=c_0, direction::WaveDirection=Incoming) = mvdr_weights(x, Sxx, f, ϕ, θ; fs=fs, c=c, direction=direction)
mpdr_weights_k(x::PhasedArray2D, Sxx, f, kx, ky; fs=nothing, c=c_0) = mvdr_weights_k(x, Sxx, f, kx, ky; fs=fs, c=c)
mpdr_weights(x::PhasedArray3D, Sxx, f, ϕ, θ; fs=nothing, c=c_0, direction::WaveDirection=Incoming) = mvdr_weights(x, Sxx, f, ϕ, θ; fs=fs, c=c, direction=direction)
mpdr_weights_k(x::PhasedArray3D, Sxx, f, kx, ky, kz; fs=nothing, c=c_0) = mvdr_weights_k(x, Sxx, f, kx, ky, kz; fs=fs, c=c)

const capon_weights = mpdr_weights
const capon_weights_k = mpdr_weights_k

function whitenoise(x::PhasedArray, σ²)
    return σ²*I(length(x.elements))
end

function diffnoise(x::PhasedArray, σ², f, c=c_0)
    ω = 2π*f
    k = ω/c
    p(x, i) = [e for e in x.elements[i]] 
    si(x) = sinc(x/π)
    Γ(x, n, m, k) = si(k*norm(p(x,m)-p(x,n)))
    n = 1:length(x.elements)
    return σ²*Γ.(Ref(x), n, n', Ref(k))
end

end