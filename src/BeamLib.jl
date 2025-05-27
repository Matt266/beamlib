module BeamLib
import Base.convert
import Base.length
using LinearAlgebra
using Convex
using SCS

export PhasedArray, IsotropicArray, ArrayManifold, NestedArray, steerphi, steerk, 
        dsb_weights, dsb_weights_k, bartlett, mvdr_weights, mvdr_weights_k,
        mpdr_weights, mpdr_weights_k, capon_weights, capon_weights_k, capon,
        whitenoise, diffnoise, esprit, music, unitary_esprit, lasso, omp, bpdn

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

function steerphi(x::IsotropicArray, f, ϕ, θ=0; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    if direction == Incoming
        ζ = [-cos(ϕ)*cos(θ), -sin(ϕ)*cos(θ), -sin(θ)]
    else
        ζ = [cos(ϕ)*cos(θ), sin(ϕ)*cos(θ), sin(θ)]
    end

    # Matlab orientation
    # ζ = [-cos(ϕ)*cos(θ), -sin(ϕ)*cos(θ), -sin(θ)]

    α = ζ/c # slowness vector
    Δ = Vector(vec(α'*x.r))
    if(!isnothing(fs))
        Δ = round.(Δ*fs)/fs
    end
    v = exp.(-1im*Δ*2π*f)
    return v
end

function steerphi(x::NestedArray, f, ϕ, θ=0; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
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

function dsb_weights(x::PhasedArray, f, ϕ, θ=0; fs=nothing,  c=c_0, direction::WaveDirection=Incoming)
    return steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)/Base.length(x)
end

function dsb_weights_k(x::PhasedArray, f, kx, ky=0, kz=0; fs=nothing,  c=c_0)
    return steerk(x, f, kx, ky, kz; fs=fs, c=c)/Base.length(x)
end

function mvdr_weights(x::PhasedArray, Snn, f, ϕ, θ=0; fs=nothing, c=c_0, direction::WaveDirection=Incoming)
    v = steerphi(x, f, ϕ, θ; fs=fs, c=c, direction=direction)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

function mvdr_weights_k(x::PhasedArray, Snn, f, kx, ky=0, kz=0; fs=nothing, c=c_0)
    v = steerk(x, f, kx, ky, kz; fs=fs, c=c)
    return (inv(Snn)*v)/(v'*inv(Snn)*v)
end

mpdr_weights(x::PhasedArray, Sxx, f, ϕ, θ=0; fs=nothing, c=c_0, direction::WaveDirection=Incoming) = mvdr_weights(x, Sxx, f, ϕ, θ; fs=fs, c=c, direction=direction)
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
bartlett(pa::PhasedArray, Rxx, f, ϕ, θ=0; w=nothing, fs=nothing, c=c_0)

Calculates the bartlett spectrum for direction of arrival estimation.
This is identically to steering a bartlett (delay-and-sum) beamformer and measuring the 
output power. Directly outputing the power spectrum for a given angle 
and data is just more convenient for DoA estimation.

arguments:
----------
    pa: PhasedArray to calculate the MUSIC spectrum for
    Rxx: covariance matrix of the array which is used for estimation
    f: center/operating frequency
    ϕ: azimuth angle for which spectrum is evaluated 
    θ: elevation angle for which spectrum is evaluated
    w: vector of taper weights for the array (e.g., chebyshev window) 
    fs: sampling frequency of the steering vector to quantize phase shifts
    c: propagation speed of the wave
"""
function bartlett(pa::PhasedArray, Rxx, f, ϕ, θ=0; w=nothing, fs=nothing, c=c_0)
    
    if isnothing(w)
        W = Matrix(I, Base.length(pa), Base.length(pa))
    else 
        assert(Base.length(w) == Base.length(pa))
        W = diagm(vec(w))
    end

    a = steerphi(pa, f, ϕ, θ; fs=fs, c=c, direction=Incoming)
    P = a'*W*Rxx*W'*a
    return P
end


"""
capon(pa::PhasedArray, Rxx, f, ϕ, θ=0; fs=nothing, c=c_0)

Calculates the capon spectrum for direction of arrival estimation.
This is identically to steering a capon beamformer and measuring the 
output power. Directly outputing the power spectrum for a given angle 
and data is just more convenient for DoA estimation.  

arguments:
----------
    pa: PhasedArray to calculate the MUSIC spectrum for
    Rxx: covariance matrix of the array which is used for estimation
    f: center/operating frequency
    ϕ: azimuth angle for which spectrum is evaluated 
    θ: elevation angle for which spectrum is evaluated
    fs: sampling frequency of the steering vector to quantize phase shifts
    c: propagation speed of the wave
"""
function capon(pa::PhasedArray, Rxx, f, ϕ, θ=0; fs=nothing, c=c_0)
    a = steerphi(pa, f, ϕ, θ; fs=fs, c=c, direction=Incoming)
    P = 1/(a'*inv(Rxx)'*a)
    return P
end

"""
esprit(Rzz, Δ, d, f, c=c_0)

Calculates the TLS esprit estimator for the direction of arrival.

arguments:
----------
    Rzz: covariance matrix of total array (both subarrays vertically concatenated),
    Δ: distance between both subarrays
    d: number of sources
    f: center/operating frequency
    c: propagation speed of the wave
    TLS: calculates total least squares solution if 'true' (default),
        least squares if 'false'
"""
function esprit(Rzz, Δ, d, f; c=c_0, TLS = true)
    # number of sensors in the array (p)
    # and the subarrays (ps)
    p = size(Rzz)[1]
    ps = Int(p/2)

    U = eigvecs(Rzz, sortby= λ -> -abs(λ))

    Es = U[:,1:d]
    Ex = Es[(1:ps),:]
    Ey = Es[(1:ps).+(ps),:]


    # estimate Φ by exploiting the array symmetry
    if(TLS)
        # TLS-ESPRIT
        E = eigvecs([Ex Ey]'*[Ex Ey], sortby= λ -> -abs(λ))
        E12 = E[1:d, (1:d).+d]
        E22 = E[(1:d).+d, (1:d).+d]
        Ψ = -E12*inv(E22)
    else
        # LS-ESPRIT
        Ψ = pinv(Ex)*Ey
    end

    Φ = eigvals(Ψ, sortby= λ -> -abs(λ))

    # calculate the directions of arrival (DoAs) from Φ
    k = (2π*f)/c
    Θ = asin.(angle.(Φ)/(k*Δ))
    return Θ
end

"""
unitary_esprit(X, J1, Δ, d, f; c=c_0, TLS = true)

Calculates the DoAs using the unitary esprit.
Requires a centrosymmetric array geometry.

arguments:
----------
    X: data matrix of the array (NO CONCATENATION of subarrays)
    J1: selection matrix for the first subarray
    Δ: distance between both subarrays
    d: number of sources
    f: center/operating frequency
    c: propagation speed of the wave
    TLS: calculates total least squares solution if 'true' (default),
        least squares if 'false'
"""
function unitary_esprit(X, J1, Δ, d, f; c=c_0, TLS = true)
    # NxN exchange matrix
    II(N) = begin
        return rotl90(Matrix(I,N,N))
    end
    
    # unitary matrices
    Q(N) = begin
        @assert N>=2
        n = Int(floor(N/2))
        if N%2 == 0
            # N even
            [I(n) 1im*I(n); II(n) -1im*II(n)]/sqrt(2)
        else
            # N odd
            [I(n) 0 1im*I(n); zeros(1,n) sqrt(2) zeros(1, n); II(n) 0 -1im*II(n)]/sqrt(2)
        end
    end

    m = size(J1)[1] # number elements in subarray
    M = size(J1)[2] # number elements in array
    K1 = Q(m)'*(J1+II(m)*J1*II(M))*Q(M)
    K2 = Q(m)'*1im*(J1-II(m)*J1*II(M))*Q(M)

    Y = Q(M)'*X
    U, _ = svd([real(Y) imag(Y)])
    Es = U[:,1:d]

    C1 = K1*Es
    C2 = K2*Es

    if(TLS)
        # TLS solution
        E, _ = svd([C1 C2]'*[C1 C2])
        E12 = E[1:d, (1:d).+d]
        E22 = E[(1:d).+d, (1:d).+d]
        Ψ = -E12*inv(E22)
    else
        # LS solution
        Ψ = pinv(C1)*C2
    end

    Φ = eigvals(Ψ, sortby= λ -> -abs(λ))

    # calculate the directions of arrival (DoAs) from Φ
    Μ = 2atan.(real(Φ))
    k = (2π*f)/c
    Θ = asin.(Μ/(k*Δ))
    return Θ
end

"""
music(pa::PhasedArray, Rxx, d, f, ϕ, θ=0; fs=nothing, c=c_0)

Calculates the MUSIC spectrum for direction of arrival estimation.

arguments:
----------
    pa: PhasedArray to calculate the MUSIC spectrum for
    Rxx: covariance matrix of the array which is used for estimation
    d: number of sources
    f: center/operating frequency
    ϕ: azimuth angle for which spectrum is evaluated 
    θ: elevation angle for which spectrum is evaluated
    fs: sampling frequency of the steering vector to quantize phase shifts
    c: propagation speed of the wave
"""
function music(pa::PhasedArray, Rxx, d, f, ϕ, θ=0; fs=nothing, c=c_0)
    U = eigvecs(Rxx, sortby= λ -> -abs(λ))

    Un = U[:, d+1:size(U)[2]]

    a = steerphi(pa, f, ϕ, θ; fs=fs, c=c, direction=Incoming)

    P = a'*a/(a'*Un*Un'*a)
    return P
end


"""
lasso(Y, A, λ=1e-2)

LASSO DOA estimation. Returns a vector representing the estimated, on-grid, spatial power spectrum of the signals. Estimated 
DOAs are the grid positions for which the spectrum crosses a certain threshold, as shown in the 'LASSO.ipynb' example.    

arguments:
----------
    Y: Data matrix of the array
    A: Dictionary matrix of array response vectors from the angle grid 
    λ: Regularization parameter for the LASSO problem
"""
function lasso(Y, A, λ=1e-2)
    X = ComplexVariable(size(A)[2], size(Y)[2])
    p = minimize(λ*sum([norm(X[i, :], 2) for i in axes(X, 1)]) + 0.5*sumsquares(A*X-Y))
    solve!(p, SCS.Optimizer)
    return norm.(eachrow(evaluate(X)),2).^2
end

"""
bpdn(Y, A, η=1e-2)

Basis Pursuit Denoising (BPDN) DOA estimation. Returns a vector representing the estimated, on-grid, spatial power spectrum of the signals. Estimated 
DOAs are the grid positions for which the spectrum crosses a certain threshold, as shown in the 'BPDN.ipynb' example.    

arguments:
----------
    Y: Data matrix of the array
    A: Dictionary matrix of array response vectors from the angle grid 
    η: Constraint and upper bound on the noise energy (η≥||E||_2). 
"""
function bpdn(Y, A, η=1e-2)
    X = ComplexVariable(size(A)[2], size(Y)[2])
    p = minimize(sum([norm(X[i, :], 2) for i in axes(X, 1)]), norm(A*X-Y, 2) <= η)
    solve!(p, SCS.Optimizer)
    return norm.(eachrow(evaluate(X)),2).^2
end

"""
omp(Y, A, d)

Orthogonal Matching Pursuit (OMP) DOA estimation. Returns a vector representing the estimated, on-grid, sparse, spatial power spectrum of the signals. Estimated 
DOAs are the angles corresponding to indices of the non-zero values of the output spectrum.

arguments:
----------
    Y: Data matrix of the array
    A: Dictionary matrix of array response vectors from the angle grid 
    d: number of sources
"""
function omp(Y, A, d)
    r = copy(Y)
    Λ = Int[]

    for _ in 1:d
        corr = A'*r
        idx = argmax(norm.(eachrow(corr)))
        push!(Λ, idx)

        Ψ = A[:, Λ]
        X = Ψ \ Y
        r = Y - Ψ*X
    end
    
    Ψ = A[:, Λ]
    s = zeros(size(A,2))
    s[Λ] .= norm.(eachrow(Ψ \ Y)).^2
    return s
end

end