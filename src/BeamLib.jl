module BeamLib
import Base.convert
import Base.length
using LinearAlgebra
using StatsBase
using Convex
using SCS
using Optimization
using OptimizationOptimJL
using Optim
using ProximalAlgorithms
using ProximalOperators

export PhasedArray, IsotropicArray, ArrayManifold, NestedArray, steer, 
        dsb_weights, bartlett, mvdr_weights, mpdr_weights, capon_weights, capon,
        whitenoise, diffnoise, esprit, music, unitary_esprit, lasso, omp, bpdn,
        aic, mdl, wsf, unconditional_signals

c_0 = 299792458.0

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

"""
steer(x::IsotropicArray, f, angles; c=c_0, dir=:rx, coords=:azel)

Calculates the steering vector/matrix for given azimuth and elevation angles. 

arguments:
----------
    f: Frequency of the signal in Hz.
    angles: Matrix with angles in the format specified in 'coords'
    c: Propagation speed of the wave (default: c_0).
    dir: Direction for beamforming:
         - ':rx': Receiving (steer for incoming wave)
         - ':tx': Transmitting (steer for outgoing wave)
         (default: ':rx').
    coords: Coordinate system of 'angles':
         - ':azel' (default): interpret angles as azimuth/elevation pairs [az; el]
         - ':k': interpret as wavevector [kx; ky; kz]
returns:
--------
    Complex steering matrix of size MxD, where M is the number of array elements and D is the number of angle pairs.
"""
function steer(x::IsotropicArray, f, angles; c=c_0, dir=:rx, coords=:azel)
    k = 2π * f / c

    if coords == :azel
        # angels = [azimtuh, elevation] 2xD matrix

        if ndims(angles) == 0
            angles = [angles; 0]
        end

        # only when single az/el pair - transpose changes ndims
        # ndims(transpose([3, 4])) is 2
        # ndims([3, 4]) is 1
        if ndims(angles) == 1
            angles = reshape(angles, 2, 1)
        end

        M, D = size(angles)

        az = transpose(angles[1, :])
        el = M == 1 ? zeros(1, D) : transpose(angles[2, :])

        ζ = [cos.(el) .* cos.(az);
             cos.(el) .* sin.(az);
             sin.(el)]

        if dir == :tx
            φ = (k .* (x.r' * ζ))
        elseif dir == :rx
            φ = -(k .* (x.r' * ζ))
        else
            throw(ArgumentError("dir must be ':rx' or ':tx'; got: '$(dir)'"))
        end 

        return exp.(-1im .* φ)

    elseif coords == :k
        # angels = [kx; ky; kz] 3xD matrix

        if ndims(angles) == 0
            angles = [angles; 0; 0]
        end

        # see quvialent part for mode == :azel
        if ndims(angles) == 1
            angles = reshape(angles, 3, 1)
        end

        M, D = size(angles)

        if M == 1
            angles = [angles; zeros(2, D)]
        elseif M==2
            angles = [angles; zeros(1, D)]
        end 
           
        ζ = angles ./ (2π*f/c)
        φ = k .* (x.r' * ζ)
        return exp.(-1im .* φ) 
    else
        throw(ArgumentError("coords must be ':azel' or ':k'; got: '$(mode)'"))
    end
end

function steer(x::NestedArray, f, angles; c=c_0, dir=:rx, coords=:azel)
    v_super = steer(x.elements, f, angles; c=c, dir=dir, coords=coords)
    v_sub = steer.(x.subarrays, Ref(f), Ref(angles); c=c, dir=dir, coords=coords)
    return reduce(vcat, map((sup_row, sub_mat) -> sub_mat .* reshape(sup_row, 1, :), eachrow(v_super), v_sub))
end

function dsb_weights(x::PhasedArray, f, angles; c=c_0, dir=:rx, coords=:azel)
    return steer(x, f, angles; c=c, dir=dir, coords=coords)/Base.length(x)
end

function mvdr_weights(x::PhasedArray, Rnn, f, angles; c=c_0, dir=:rx, coords=:azel)
    v = steer(x, f, angles; c=c, dir=dir, coords=coords)
    return (inv(Rnn)*v)/(v'*inv(Rnn)*v)
end

mpdr_weights(x::PhasedArray, Rxx, f, angles; c=c_0, dir=:rx, coords=:azel) = mvdr_weights(x, Rxx, f, angles; c=c, dir=dir, coords=coords)

const capon_weights = mpdr_weights

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
bartlett(pa::PhasedArray, Rxx, angles; w=nothing, c=c_0)

Calculates the bartlett spectrum for direction of arrival estimation.
This is identically to steering a bartlett (delay-and-sum) beamformer and measuring the 
output power. Directly outputing the power spectrum for a given angle 
and data is just more convenient for DoA estimation.

arguments:
----------
    pa: PhasedArray to calculate the estimator for
    Rxx: covariance matrix of the array which is used for estimation
    f: center/operating frequency
    angles: 1xD or 2xD matrix of steering directions.
        - If 1xD: azimuth angles in radians, elevation assumed zero.
        - If 2xD: [azimuth; elevation] for D directions in radians.
    w: vector of taper weights for the array (e.g., chebyshev window) 
    c: propagation speed of the wave
"""
function bartlett(pa::PhasedArray, Rxx, f, angles; w=nothing, c=c_0, coords=:azel)
    
    if isnothing(w)
        W = Matrix(I, Base.length(pa), Base.length(pa))
    else 
        assert(Base.length(w) == Base.length(pa))
        W = diagm(vec(w))
    end

    A = steer(pa, f, angles; c=c, dir=:rx, coords=coords)

    # a'*W*Rxx*W'*a
    P = vec(sum(conj(A) .* (W*Rxx*W' * A), dims=1))
    return real(P)
end


"""
capon(pa::PhasedArray, Rxx, f, ϕ, θ=0; fs=nothing, c=c_0)

Calculates the capon spectrum for direction of arrival estimation.
This is identically to steering a capon beamformer and measuring the 
output power. Directly outputing the power spectrum for a given angle 
and data is just more convenient for DoA estimation.  

arguments:
----------
    pa: Array to evaluate for
    Rxx: Covariance matrix (MxM)
    angles: 1xD or 2xD matrix of [azimuth; elevation] angles in radians
    c: Propagation speed (default: c_0)

returns:
--------
    P: Power spectrum evaluated at each direction
"""
function capon(pa::PhasedArray, Rxx, f, angles; c=c_0, coords=:azel)
    A = steer(pa, f, angles; c=c, dir=:rx, coords=coords)
    #P = 1/(a'*inv(Rxx)*a)
    P = vec(1 ./ sum(conj(A) .* (inv(Rxx) * A), dims=1))
    return real(P)
end

"""
esprit(Rzz, Δ, d, f; c=c_0, TLS = true, side = :left)

Calculates the TLS esprit estimator for the direction of arrival.
Returns a vector of tuples with each tuple holding the two ambigues 
DOAs corresponding to a source. 

arguments:
----------
    Rzz: covariance matrix of total array (both subarrays vertically concatenated),
    Δ: displacement vector between both subarrays
    d: number of sources
    f: center/operating frequency
    c: propagation speed of the wave
    TLS: calculates total least squares solution if 'true' (default),
        least squares if 'false'
    side: choose angles on the left (':left'), right (':right'), or both (':both') sides
        of the displacement vector to decide between the two ambigues angles per source.
"""
function esprit(Rzz, Δ, d, f; c=c_0, TLS = true, side = :left)
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

    # orientation of displacement vector
    ϕ0 = atan(Δ[2], Δ[1])  

    # angle estimates to the left of Δ
    Θ1 = ϕ0 .+ acos.(angle.(Φ) ./ (k * norm(Δ)))
    Θ1 =  mod.(Θ1 .+ π, 2π) .- π

    # angle estimates to the right of Δ
    Θ2 = ϕ0 .- acos.(angle.(Φ) ./ (k * norm(Δ)))
    Θ2 = mod.(Θ2 .+ π, 2π) .- π

    # select ambigues angles left or right
    # respective to displacement vector Δ
    if side == :left
        return Θ1
    elseif side == :right
        return Θ2
    elseif side == :both
        return Θ1, Θ2
    else
       error("Invalid symbol for side: ':$(side)'. Valid options are: ':left', ':right', ':both'")
    end
    # for displacement along y-axis:
    # asin.(angle.(Φ)/(k*norm(Δ)))
end

"""
unitary_esprit(X, J1, Δ, d, f; c=c_0, TLS = true, side = :left)

Calculates the DoAs using the unitary esprit.
Requires a centrosymmetric array geometry.

arguments:
----------
    X: data matrix of the array (NO CONCATENATION of subarrays)
    J1: selection matrix for the first subarray
    Δ: displacement vector between both subarrays
    d: number of sources
    f: center/operating frequency
    c: propagation speed of the wave
    TLS: calculates total least squares solution if 'true' (default),
        least squares if 'false'
    side: choose angles on the left (':left'), right (':right'), or both (':both') sides
        of the displacement vector to decide between the two ambigues angles per source.
"""
function unitary_esprit(X, J1, Δ, d, f; c=c_0, TLS = true, side = :left)
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
            [I(n) zeros(n,1) 1im*I(n); zeros(1,n) sqrt(2) zeros(1, n); II(n) zeros(n,1) -1im*II(n)]/sqrt(2)
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

    # orientation of displacement vector
    ϕ0 = atan(Δ[2], Δ[1])  

    # angle estimates to the left of Δ
    Θ1 = ϕ0 .+ acos.(Μ ./ (k * norm(Δ)))
    Θ1 =  mod.(Θ1 .+ π, 2π) .- π

    # angle estimates to the right of Δ
    Θ2 = ϕ0 .- acos.(Μ ./ (k * norm(Δ)))
    Θ2 = mod.(Θ2 .+ π, 2π) .- π

    # select ambigues angles left or right
    # respective to displacement vector Δ
    if side == :left
        return Θ1
    elseif side == :right
        return Θ2
    elseif side == :both
        return Θ1, Θ2
    else
       error("Invalid symbol for side: ':$(side)'. Valid options are: ':left', ':right', ':both'")
    end

    # for displacement along y-axis:
    # Θ = asin.(Μ/(k*Δ))
end

"""
music(pa::PhasedArray, Rxx, d, f, angles; c=c_0)

Calculates the MUSIC spectrum for direction of arrival (DoA) estimation.

arguments:
----------
    pa: Array for which the MUSIC spectrum is computed
    Rxx: Covariance matrix of the received signals
    d: Number of signal sources (model order)
    f: Center/operating frequency
    angles: 1xD vector or 2xD matrix of azimuth and elevation angles. For 1xD input, elevation is assumed zero.
    c: Propagation speed of the wave (default: c_0)
"""
function music(pa::PhasedArray, Rxx, d, f, angles; c=c_0, coords=:azel)
    U = eigvecs(Rxx, sortby= λ -> -abs(λ))

    Un = U[:, d+1:size(U)[2]]

    A = steer(pa, f, angles; c=c, dir=:rx, coords=coords)

    #P = a'*a/(a'*Un*Un'*a)
    P = vec(sum(abs2, A; dims=1) ./ sum(abs2, Un' * A; dims=1))
    return real(P)
end

"""
wsf(pa::PhasedArray, Rxx, d, ϕ0, f; fs=nothing, c=c_0, optimizer=NelderMead(), maxiters=1e3)

DoA estimation using weighted subspace fitting (WSF) .

arguments:
----------
    pa: PhasedArray to calculate the MUSIC spectrum for
    Rxx: covariance matrix of the array which is used for estimation
    d: number of sources
    ϕ0: vector of initial DoAs as starting point for WSF
    f: center/operating frequency
    fs: sampling frequency of the steering vector to quantize phase shifts
    c: propagation speed of the wave
    optimizer: used optimizer to solve the WSF problem
    maxiters: maximum optimization iterations
"""
function wsf(pa::PhasedArray, Rxx, d, ϕ0, f; c=c_0, coords=:azel, optimizer=NelderMead(), maxiters=1e3)
    p = pa, Rxx, d, f, c
    wsf_cost = function(ϕ, p)
        pa, Rxx, d, f, c = p
        Λ, U = eigen(Rxx, sortby= λ -> -abs(λ))

        Λs = diagm(Λ[1:d])
        Λn = diagm(Λ[d+1:length(Λ)])

        Us = U[:, 1:d]
        Un = U[:, d+1:size(U, 2)]

        # estimate noise variance as mean of noise eigenvalues
        σ² = mean(Λn)
        Λest = Λs - σ²*I
        W = Λest^2*inv(Λs)

        #A = hcat(steerphi.(Ref(pa), f, ϕ; fs=fs, c=c, direction=Incoming)...)
        A = steer(pa, f, ϕ'; c=c, dir=:rx, coords=coords)
        PA = A*inv(A'*A)*A'

        cost =  tr((I-PA)*Us*W*Us')
        return real(cost)
    end
    
    f = OptimizationFunction(wsf_cost, Optimization.AutoForwardDiff())
    p = OptimizationProblem(f, ϕ0, p)
    s = solve(p, optimizer; maxiters=maxiters)
    return s.u
end


"""
aic(Rxx, N)

Estimates number of sources using the Akaike information criterion (AIC).

arguments:
----------
    Rxx: covariance matrix of the array which is used for estimation
    N: sample size
"""
function aic(Rxx, N)
    p = size(Rxx, 1)
    λ = eigvals(Rxx, sortby= λ -> -abs(λ))

    aic = zeros(p)

    for k in 0:p-1
        λ_n = λ[k+1:end]
        L = log(StatsBase.geomean(λ_n)/mean(λ_n))
        aic[k+1] = -L^((p-k)N) + k*(2p-k)
    end

    # filter out invalid (Inf, -Inf, NaN) results
    orders = (0:p-1)[map(!, (isnan.(aic) .|| isinf.(aic)))]
    aic = aic[map(!, (isnan.(aic) .|| isinf.(aic)))]
    return orders[argmin(aic)]
end

"""
mdl(Rxx, N)

Estimates number of sources using the Minimum Description Length (MDL) model selection criteria.

arguments:
----------
    Rxx: covariance matrix of the array which is used for estimation
    N: sample size
"""
function mdl(Rxx, N)
    p = size(Rxx, 1)
    λ = eigvals(Rxx, sortby= λ -> -abs(λ))
    mdl = zeros(p)

    for k in 0:p-1
        λ_n = λ[k+1:end]
        L = log(StatsBase.geomean(λ_n)/mean(λ_n))
        mdl[k+1] = -L^((p-k)N) + 0.5k*(2p-k)log(N)
    end

    # filter out invalid (Inf, -Inf, NaN) results
    orders = (0:p-1)[map(!, (isnan.(mdl) .|| isinf.(mdl)))]
    mdl = mdl[map(!, (isnan.(mdl) .|| isinf.(mdl)))]
    return orders[argmin(mdl)]
end

# for solving lasso with LeastSquares from ProximalOperators.jl
function ProximalAlgorithms.value_and_gradient(f::ProximalOperators.LeastSquares, X)
    val = f(X)
    grad = similar(X)
    gradient!(grad, f, X)
    return val, grad
end

"""
lasso(Y, A, λ=1e-2; max_iter=300, tol=1e-6)

LASSO DOA estimation. Returns a vector representing the estimated, on-grid, spatial power spectrum of the signals. Estimated 
DOAs are the grid positions for which the spectrum crosses a certain threshold, as shown in the 'LASSO.ipynb' example.    

arguments:
----------
    Y: Data matrix of the array
    A: Dictionary matrix of array response vectors from the angle grid 
    λ: Regularization parameter for the LASSO problem
    maxit: maximum iterations for optimization
    tol: tolerance for optimization 
"""
function lasso(Y, A, λ=1e-2; maxit=100, tol=1e-6)
    f = LeastSquares(A, Y)
    g = NormL21(λ, 2)
    X0 = zeros(ComplexF64, size(A,2), size(Y,2))
    ffb = ProximalAlgorithms.FastForwardBackward(maxit=maxit, tol=tol)
    solution, _ = ffb(x0=X0, f=f, g=g)
    return norm.(eachrow(solution), 2).^2
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
    Convex.solve!(p, SCS.Optimizer)
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


"""
unconditional_signals(Rss, N; norm=true)

Generates signals corresponding to the unconditional signal model (stochastic signals) but with given spatial correlation

arguments:
----------
    Rss: Covariance matrix of the signals
    N: number of snapshots to generate
    norm: normalize so source power adds up to unit power
"""
function unconditional_signals(Rss, N; norm=true)
    # number signals
    d = size(Rss, 1)

    # generate random signals
    w = (randn(d, N) + 1im*randn(d, N))/sqrt(2)

    # normalize so source power adds up to unit power
    if norm
        Rss = Rss/tr(Rss)
    end

    # correlate sources 
    s = cholesky(Rss).L * w
    return s
end

end