module BeamLib

import Base.convert

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

function steerphi(x::PhasedArray1D, f, phi; fs=nothing, c=299792458, direction::WaveDirection=Incoming)
    zeta = cosd(phi)
    zeta = zeta*Int(direction) # propagation direction
    alpha = zeta/c # slowness vector
    delta = alpha.*getindex.(x.elements, 1)
    if(!isnothing(fs))
        delta = round.(delta*fs)/fs
    end
    return exp.(-1im*delta*2*pi*f)
end

function steerphi(x::PhasedArray2D, f, phi; fs=nothing, c=299792458, direction::WaveDirection=Incoming)
    zeta = [cosd(phi), sind(phi)] 
    zeta = zeta*Int(direction) # propagation direction
    alpha = zeta/c # slowness vector
    delta = alpha[1].*getindex.(x.elements, 1) + alpha[2].*getindex.(x.elements, 2)
    if(!isnothing(fs))
        delta = round.(delta*fs)/fs
    end
    return exp.(-1im*delta*2*pi*f)
end

function steerphi(x::PhasedArray3D, f, phi, theta; fs=nothing, c=299792458, direction::WaveDirection=Incoming)
    zeta = [cosd(phi)*sind(theta), sind(phi)*sind(theta), cosd(theta)] 
    zeta = zeta*Int(direction) # propagation direction
    alpha = zeta/c # slowness vector
    delta = alpha[1].*getindex.(x.elements, 1) + alpha[2].*getindex.(x.elements, 2) + alpha[3].*getindex.(x.elements, 3)
    if(!isnothing(fs))
        delta = round.(delta*fs)/fs
    end
    return exp.(-1im*delta*2*pi*f)
end

function dsb_weights(x::PhasedArray, f, phi; fs=nothing,  c=299792458, direction::WaveDirection=Incoming)
    return steerphi(x, f, phi; fs=fs, c=c, direction=direction)/length(x.elements)
end

end