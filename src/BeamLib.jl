module BeamLib
import Base.convert
import Base.length
using LinearAlgebra

export PhasedArray, IsotropicArray, ArrayManifold, NestedArray, steerphi, steerk, 
        dsb_weights, dsb_weights_k, mvdr_weights, mvdr_weights_k,
        mpdr_weights, mpdr_weights_k, capon_weights, capon_weights_k,
        whitenoise, diffnoise, esprit

c_0 = 299792458.0
@enum WaveDirection begin
    Incoming = -1
    Outgoing = 1
end

# generic phased arrays
abstract type PhasedArray end

# placeholder for future implementations
# TODO: implement
abstract type ArrayManifold <: PhasedArray end

# phased arrays with only isotropic radiators as elements 
# perfect for quick evaluation of array factors
struct IsotropicArray <: PhasedArray
    r::Matrix{<:Number}
    function IsotropicArray(r::Matrix{<:Number})
        @assert 1 <= size(r)[1] <= 3 "Found invalid shape of input matrix: $(size(r)[1])x$(size(r)[2]). Each column are the 1D, 2D, or 3D coordinates of an element. Input matrix must be of size 1xN, 2xN, or 3xN respectively"

        if size(r)[1] == 1
            return new([r; zeros(2, size(r)[2])])
        elseif size(r)[1] == 2
            return new([r; zeros(1, size(r)[2])])
        else
            return new(r)
        end 
    end
end

function IsotropicArray(r::Vector{<:Number})
    return IsotropicArray(reshape(r, 1, length(r)))
end

IsotropicArray(elements::Number...) = IsotropicArray([e for e in elements])


struct NestedArray <: PhasedArray
    elements::IsotropicArray
    subarrays::Vector{<:PhasedArray}
end

function Base.length(x::IsotropicArray)
    return size(x.r)[2]
end

function Base.length(x::NestedArray)
    return sum(length.(x.subarrays))
end

function steerphi(x::IsotropicArray, f, ϕ, θ=π/2; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    ζ = [cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)] 
    ζ = ζ*Int(direction) # propagation direction
    α = ζ/c # slowness vector
    Δ = Vector(vec(α'*x.r))
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

function steerphi(x::NestedArray, f, ϕ, θ=π/2; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    v = steerphi(x.elements, f, ϕ, θ, fs=fs, c=c, direction=direction)
    return reduce(vcat, Tuple(v[i]*steerphi(s, f, ϕ, θ, fs=fs, c=c, direction=direction) for (i,s) in enumerate(x.subarrays)))
end

function steerk(x::IsotropicArray, f, kx, ky=0, kz=0; fs=nothing, c=c_0)
    k = 2π*f/c # wavenumber
    ζ = [kx/k, ky/k, kz/k] # propagation direction
    α = ζ/c # slowness vector
    Δ =  Vector(vec(α'*x.r))
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    return exp.(-1im*Δ*2π*f)
end

function steerk(x::NestedArray, f, kx, ky=0, kz=0; fs=nothing, c=c_0)
    v = steerk(x.elements, f, kx, ky, kz, fs=fs, c=c)
    return reduce(vcat, Tuple(v[i]*steerk(s, f, kx, ky, kz; fs=fs, c=c) for (i,s) in enumerate(x.subarrays)))
end

function dsb_weights(x::PhasedArray, f, ϕ, θ=π/2; fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)/Base.length(x)
end

function dsb_weights_k(x::PhasedArray, f, kx, ky=0, kz=0; fs=nothing,  c=c_0)
    return steerk(x, f, kx, ky, kz; fs=fs, c=c)/Base.length(x)
end

function mvdr_weights(x::PhasedArray, Snn, f, ϕ, θ=π/2; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    v = steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

function mvdr_weights_k(x::PhasedArray, Snn, f, kx, ky=0, kz=0; fs=nothing, c=c_0)
    v = steerk(x, f, kx, ky, kz; fs=fs, c=c)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

mpdr_weights(x::PhasedArray, Sxx, f, ϕ, θ=π/2; fs=nothing, c=c_0, direction::WaveDirection=Incoming) = mvdr_weights(x, Sxx, f, ϕ, θ; fs=fs, c=c, direction=direction)
mpdr_weights_k(x::PhasedArray, Sxx, f, kx, ky=0, kz=0; fs=nothing, c=c_0) = mvdr_weights_k(x, Sxx, f, kx, ky, kz; fs=fs, c=c)

const capon_weights = mpdr_weights
const capon_weights_k = mpdr_weights_k

function whitenoise(x::IsotropicArray, σ²)
    return σ²*I(Base.length(x))
end

function diffnoise(x::IsotropicArray, σ², f, c=c_0)
    ω = 2π*f
    k = ω/c
    p(x, i) = [e for e in x.r[:,i]] 
    si(x) = sinc(x/π)
    Γ(x, n, m, k) = si(k*norm(p(x,m)-p(x,n)))
    n = 1:Base.length(x)
    return σ²*Γ.(Ref(x), n, n', Ref(k))
end

"""
    esprit(Z, Δ, d, f, c=c_0)

    Calculates the TLS esprit estimator for the direction of arrival.

    arguments:
    ----------
        Z: data matrix, NOT covariance matrix
        Δ: distance between both subarrays
        d: number of sources or name of the estimator
            for source detection
        f: center/operating frequency
        c: propagation speed of the wave
"""
function esprit(Z, Δ, d, f, c=c_0)
    # number of sensors in the array (p)
    # and the subarrays (ps)
    p = size(Z)[1]
    ps = Int(p/2)

    U, _ = svd(1/size(Z)[2] * Z*Z')

    #TODO: source detection
    # d = ...

    # obtain signal subspace estimate Es
    Es = U[:,1:d]
    Ex = Es[(1:ps),:]
    Ey = Es[(1:ps).+(ps),:]

    # estimate Φ by exploiting the array symmetry
    E, _ = svd([Ex';Ey']*[Ex Ey])
    E12 = E[1:d, (1:d).+d]
    E22 = E[(1:d).+d, (1:d).+d]

    # TLS
    Ψ = -E12*inv(E22)

    # LS
    #Ψ = pinv(Ex)*Ey

    U, S, _ = svd(Ψ)
    Φ = S.^2

    # calculate the directions of arrival (DoAs) from Φ
    ks = c/(2π*f*Δ)
    Θ = asin.((ks*real.(Φ)))
    return Θ
end

end