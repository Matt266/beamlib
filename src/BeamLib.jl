module BeamLib

import Base.convert
using Unitful, UnitfulAngles
using PhysicalConstants.CODATA2018: c_0

@enum WaveDirection begin
    Incoming = -1
    Outgoing = 1
end

abstract type PhasedArray end

struct PhasedArray1D{T <: Number} <: PhasedArray
    elements::Vector{Tuple{T}}
end

PhasedArray1D(elements::Tuple{T}...) where {T <: Number} = PhasedArray1D{T}([e for e in elements])
PhasedArray1D(elements::Vector{T}) where {T <: Number} = PhasedArray1D{T}(map((x) -> (x,), elements))
PhasedArray1D(elements::T...) where {T <: Number} = PhasedArray1D{T}([Tuple(e) for e in elements])

struct PhasedArray2D{T <: Number} <: PhasedArray
    elements::Vector{Tuple{T, T}}
end

PhasedArray2D(elements::Tuple{T, T}...) where {T <: Number} = PhasedArray2D{T}([e for e in elements])
PhasedArray2D(x::PhasedArray1D{T})  where {T <: Number} = PhasedArray2D{T}([(e[1], Base.convert(typeof(e[1]),0)) for e in x.elements])
Base.convert(::Type{PhasedArray2D}, x::PhasedArray1D{T}) where {T <: Number} = PhasedArray2D(x)

struct PhasedArray3D{T <: Number} <: PhasedArray
    elements::Vector{Tuple{T, T, T}}
end

PhasedArray3D(elements::Tuple{T, T, T}...) where {T <: Number} = PhasedArray3D{T}([e for e in elements])
PhasedArray3D(x::PhasedArray1D{T})  where {T <: Number} = PhasedArray3D{T}([(e[1], Base.convert(typeof(e[1]),0), Base.convert(typeof(e[1]),0)) for e in x.elements])
PhasedArray3D(x::PhasedArray2D{T})  where {T <: Number} = PhasedArray3D{T}([(e[1], e[2], Base.convert(typeof(e[1]),0)) for e in x.elements])
Base.convert(::Type{PhasedArray3D}, x::PhasedArray1D{T}) where {T <: Number} = PhasedArray3D(x)
Base.convert(::Type{PhasedArray3D}, x::PhasedArray2D{T}) where {T <: Number} = PhasedArray3D(x)

function steerphi(x::PhasedArray1D; f, ϕ, fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    ζ = cos(ϕ)
    ζ = ζ*Int(direction) # propagation direction
    α = ζ/c # slowness vector
    Δ = α.*getindex.(x.elements, 1)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*uconvert.(NoUnits, Δ*2π*f))
end

function steerphi(x::PhasedArray2D; f, ϕ, fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    ζ = [cos(ϕ), sin(ϕ)] 
    ζ = ζ*Int(direction) # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*uconvert.(NoUnits, Δ*2π*f))
end

function steerphi(x::PhasedArray3D; f, ϕ, θ, fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    ζ = [cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)] 
    ζ = ζ*Int(direction) # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2) + α[3].*getindex.(x.elements, 3)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*uconvert.(NoUnits, Δ*2π*f))
end

function steerk(x::PhasedArray1D; f, kx, fs=nothing, c=c_0)
    k = 2π*f/c # wavenumber
    ζ = kx/k # propagation direction
    α = ζ/c # slowness vector
    Δ = α.*getindex.(x.elements, 1)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*uconvert.(NoUnits, Δ*2π*f))
end

function steerk(x::PhasedArray2D; f, kx, ky, fs=nothing, c=c_0)
    k = 2π*f/c # wavenumber
    ζ = [kx/k, ky/k] # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*uconvert.(NoUnits, Δ*2π*f))
end

function steerk(x::PhasedArray3D; f, kx, ky, kz, fs=nothing, c=c_0)
    k = 2π*f/c # wavenumber
    ζ = [kx/k, ky/k, kz/k] # propagation direction
    α = ζ/c # slowness vector
    Δ = α[1].*getindex.(x.elements, 1) + α[2].*getindex.(x.elements, 2) + α[3].*getindex.(x.elements, 3)
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*uconvert.(NoUnits, Δ*2π*f))
end

function dsb_weights(x::PhasedArray1D; f, ϕ, fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f=f, ϕ=ϕ; fs=fs, c=c, direction=direction)/length(x.elements)
end

function dsb_weights(x::PhasedArray2D; f, ϕ, fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f=f, ϕ=ϕ; fs=fs, c=c, direction=direction)/length(x.elements)
end

function dsb_weights(x::PhasedArray3D; f, ϕ, θ, fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f=f, ϕ=ϕ, θ=θ; fs=fs, c=c, direction=direction)/length(x.elements)
end

end