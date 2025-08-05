module BeamLib
using LinearAlgebra
using StatsBase
using ForwardDiff
using Roots
using Convex
using SCS
using Optimization
using OptimizationOptimJL
using Optim
using ProximalAlgorithms
using ProximalOperators
using Interpolations
using PRIMA

export PhasedArray, IsotropicArray, PhasedArray, NestedArray, steer, IsotropicArrayManifold, SampledArrayManifold,
        AzEl, WaveVec,
        dsb_weights, bartlett, mvdr_weights, mpdr_weights, capon_weights, capon,
        whitenoise, diffnoise, esprit, music, unitary_esprit, lasso, omp, ols, bpdn,
        aic, mdl, wsf, dml, sml, unconditional_signals, find_doas

c_0 = 299792458.0

abstract type ArrayManifold end

abstract type AbstractPhasedArray end

struct PhasedArray <: AbstractPhasedArray
    manifold::ArrayManifold
end

struct NestedArray <: AbstractPhasedArray
    elements::PhasedArray
    subarrays::Vector{<:AbstractPhasedArray}
end

struct IsotropicArrayManifold <: ArrayManifold
    r::AbstractMatrix
    function IsotropicArrayManifold(r::AbstractMatrix)
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

function IsotropicArrayManifold(r::AbstractVector)
    return IsotropicArrayManifold(reshape(r, 1, length(r)))
end

IsotropicArrayManifold(elements::Number...) = IsotropicArrayManifold([e for e in elements])

IsotropicArray(args...; kwargs...) = PhasedArray(IsotropicArrayManifold(args...; kwargs...))  

function Base.length(a::IsotropicArrayManifold)
    return size(a.r)[2]
end

function Base.length(pa::PhasedArray)
    return length(pa.manifold)
end

function Base.length(pa::NestedArray)
    return sum(length.(pa.subarrays))
end

abstract type Wavefront end
abstract type PlaneWave <: Wavefront end
abstract type SphericalWave <: Wavefront end

struct AzEl <: PlaneWave
    coords::AbstractMatrix
    function AzEl(coords)
         # coords = [azimtuh, elevation] 2xD matrix

        if ndims(coords) == 0
            coords = [coords; 0]
        end

        # only when single az/el pair
        # transpose changes ndims 
        # az-only: [az0, az2]', az/el-pair: [az, el]
        # ndims(transpose([3, 4])) is 2
        # ndims([3, 4]) is 1
        if ndims(coords) == 1
            coords = reshape(coords, 2, 1)
        end

        M, D = size(coords)

        if M == 1
            coords = [coords; zeros(1, D)]
        end

        return new(coords)
    end
end

struct WaveVec <: PlaneWave
    coords::AbstractMatrix
    function WaveVec(coords)
        # coords = [kx; ky; kz] 3xD matrix

        if ndims(coords) == 0
            coords = [coords; 0; 0]
        end

        # only when single k vector
        # transpose changes ndims
        # single k vector: [kx, ky, kz]'
        # multiple kx: [kx, kx, kx]
        if ndims(coords) == 1
            coords = reshape(coords, 3, 1)
        end

        M, D = size(coords)

        if M == 1
            coords = [coords; zeros(2, D)]
        elseif M==2
            coords = [coords; zeros(1, D)]
        end 

        return new(coords)
    end
end

function steer(pa::PhasedArray, args...; kwargs...)
    return pa.manifold(args...; kwargs...)
end

function steer(pa::NestedArray, args...; kwargs...)
    v_super = steer(pa.elements, args...; kwargs...)
    v_sub = map(sa -> steer(sa, args...; kwargs...), pa.subarrays)
    return reduce(vcat, map((sup_row, sub_mat) -> sub_mat .* reshape(sup_row, 1, :), eachrow(v_super), v_sub))
end

"""
References:
-----------
H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
"""

function (a::IsotropicArrayManifold)(f, angles::AzEl; c=c_0)
    k = 2π * f / c

    az = transpose(angles.coords[1, :])
    el = transpose(angles.coords[2, :])

    ζ = [cos.(el) .* cos.(az);
         cos.(el) .* sin.(az);
         sin.(el)]

    φ = -(k .* (a.r' * ζ))

    return exp.(-1im .* φ)
end

"""
References:
-----------
H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
"""
function (a::IsotropicArrayManifold)(f, angles::WaveVec; c=c_0)
    #k = 2π * f / c  
    #ζ = angles.coords ./ k
    #φ = k .* (a.r' * ζ)
    φ = a.r' * angles.coords
    return exp.(-1im .* φ)
end

"""
steer(x::IsotropicArray, f, angles; c=c_0, coords=:azel)

Calculates the steering vector/matrix for given azimuth and elevation angles. 

arguments:
----------
    f: Frequency of the signal in Hz.
    angles: Matrix with angles in the format specified in 'coords'
    c: Propagation speed of the wave (default: c_0).
    coords: Coordinate system of 'angles':
         - ':azel' (default): interpret angles as azimuth/elevation pairs [az; el]
         - ':k': interpret as wavevector [kx; ky; kz]
returns:
--------
    Complex steering matrix of size MxD, where M is the number of array elements and D is the number of angle pairs.

References:
-----------
H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
"""
function (a::ArrayManifold)(f, angles; c=c_0, coords=:azel)
    if coords == :azel
        return a(f, AzEl(angles); c=c)
    elseif coords == :k
        return a(f, WaveVec(angles); c=c)
    else
        throw(ArgumentError("coords must be ':azel' or ':k'; got: '$(mode)'"))
    end
end

"""
Creates an ArrayManifold from recorded samples.
'responses' array holding the complex amplitudes should have following dimension order:
    propagation_speeds - frequencies - spatial coordinates - array elements

E.g., for AzEl this would be (check implemented Wavefront types for supported formats):
    propagation_speeds - frequencies - azimuths - elevations - array elements

Note that each dimension of size 1 (or less) can be dropped. So, if only azimuths for a
single frequency are recorded, responses reduces to:
    azimuths - array elements

You must sample over at least one dimension. If other dependencies (e.g., temperature) are 
be needed, either create an array of SampledArrayManifolds or implement a custom ArrayManifold subtype.
"""
struct SampledArrayManifold <: ArrayManifold
    c_grid::AbstractVector
    f_grid::AbstractVector
    coords_grid::Wavefront
    num_elements::Int
    responses::AbstractArray
    keep_axes::AbstractVector
    itp_mag::AbstractInterpolation
    itp_phase::AbstractInterpolation
    function SampledArrayManifold(responses::AbstractArray; c_grid=[0], f_grid=[0], coords_grid=AzEl(0))
        num_elements = size(responses)[end]

        # sort all axes
        coords_perms = sortperm.(unique.(eachrow(coords_grid.coords)))
        coords_axes = map(i -> unique(eachrow(coords_grid.coords)[i])[coords_perms[i]], 1:length(coords_perms))

        c_perms = sortperm(c_grid)
        c_grid = c_grid[c_perms]

        f_perms = sortperm(f_grid)
        f_grid = f_grid[f_perms]

        element_perms = Vector(1:num_elements)
        element_axes = Vector(1:num_elements)

        # tuple with all axes of length >1
        grid_perms = [c_perms, f_perms, coords_perms..., element_perms]
        grid_axes = [c_grid, f_grid, coords_axes..., element_axes]
        keep_axes = length.(grid_axes) .> 1
        grid_perms = (grid_perms[keep_axes]...,)
        grid_axes = (grid_axes[keep_axes]...,)

        # responses with all axes of length >1
        flattened_responses = dropdims(responses; dims=Tuple(findall(size(responses) .<= 1)))

        # permute responsens to match sorted axes again
        flattened_responses = flattened_responses[grid_perms...]
        
        itp_mag = extrapolate(interpolate(grid_axes, abs.(flattened_responses), Gridded(Interpolations.Linear())),Interpolations.Flat())
        itp_phase = extrapolate(interpolate(grid_axes, angle.(flattened_responses), Gridded(Interpolations.Linear())),Interpolations.Flat())
        return new(c_grid, f_grid, coords_grid, num_elements, flattened_responses, keep_axes, itp_mag, itp_phase)
    end
end

# TODO: implement Wavefront type conversion logic.
# Currently will fail if called with other format than the sampled one,
# which for now is the intended behaviour 
# TODO: probably in each Wavefront constructer: constrain values (e.g., azimuth to -π...+π and elevation to 0...π).
function (a::SampledArrayManifold)(f, angles::Wavefront; c=c_0)
    angles = convert(typeof(a.coords_grid), angles)

    # matrix of tuples: each row an element index e, each col an column from the angles 
    # all axes with size <= 1 will be dropped from the points
    query_points = hcat(map(col -> map(e -> Tuple([c, f, col..., e][a.keep_axes]), 1:a.num_elements), eachcol(angles.coords))...)
    return map(query_point -> (a.itp_mag(query_point...)*exp(1im*a.itp_phase(query_point...))), query_points)
end

function Base.length(a::SampledArrayManifold)
    return a.num_elements
end

"""
References:
-----------
H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
"""
function dsb_weights(pa::AbstractPhasedArray, f, angles; c=c_0, coords=:azel)
    return steer(pa, f, angles; c=c, coords=coords)/length(pa)
end

"""
References:
-----------
H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
"""
function mvdr_weights(pa::AbstractPhasedArray, Rnn, f, angles; c=c_0, coords=:azel)
    v = steer(pa, f, angles; c=c, coords=coords)
    return (inv(Rnn)*v)/(v'*inv(Rnn)*v)
end

mpdr_weights(pa::AbstractPhasedArray, Rxx, f, angles; c=c_0, coords=:azel) = mvdr_weights(pa, Rxx, f, angles; c=c, coords=coords)

const capon_weights = mpdr_weights

function whitenoise(pa::AbstractPhasedArray, σ²)
    return σ²*I(length(pa))
end

"""
References:
-----------
W. Herbordt, Sound capture for human / machine interfaces, 2005th ed. Berlin, Germany: Springer, 2005.
"""
function diffnoise(pa::AbstractPhasedArray, σ², f, c=c_0)
    ω = 2π*f
    k = ω/c
    p(x, i) = [e for e in x.manifold.r[:,i]] 
    si(x) = sinc(x/π)
    Γ(x, n, m, k) = si(k*norm(p(x,m)-p(x,n)))
    n = 1:length(pa)
    return σ²*Γ.(Ref(pa), n, n', Ref(k))
end

"""
bartlett(pa::AbstractPhasedArray, Rxx, angles; w=nothing, c=c_0)

Calculates the bartlett spectrum for direction of arrival estimation.
This is identically to steering a bartlett (delay-and-sum) beamformer and measuring the 
output power. Directly outputing the power spectrum for a given angle 
and data is just more convenient for DoA estimation.

arguments:
----------
    pa: AbstractPhasedArray to calculate the estimator for
    Rxx: covariance matrix of the array which is used for estimation
    f: center/operating frequency
    angles: 1xD or 2xD matrix of steering directions.
        - If 1xD: azimuth angles in radians, elevation assumed zero.
        - If 2xD: [azimuth; elevation] for D directions in radians.
    w: vector of taper weights for the array (e.g., chebyshev window) 
    c: propagation speed of the wave

References:
-----------
H. Krim and M. Viberg, ‘Two decades of array signal processing research: the parametric approach’, IEEE Signal Process. Mag., vol. 13, no. 4, pp. 67–94, Jul. 1996.
"""
function bartlett(pa::AbstractPhasedArray, Rxx, f, angles; w=nothing, c=c_0, coords=:azel)
    
    if isnothing(w)
        W = Matrix(I, Base.length(pa), Base.length(pa))
    else 
        assert(Base.length(w) == Base.length(pa))
        W = diagm(vec(w))
    end

    A = steer(pa, f, angles; c=c, coords=coords)

    # a'*W*Rxx*W'*a
    P = vec(sum(conj(A) .* (W*Rxx*W' * A), dims=1))
    return real(P)
end


"""
capon(pa::AbstractPhasedArray, Rxx, f, ϕ, θ=0; fs=nothing, c=c_0)

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

References:
-----------
H. Krim and M. Viberg, ‘Two decades of array signal processing research: the parametric approach’, IEEE Signal Process. Mag., vol. 13, no. 4, pp. 67–94, Jul. 1996.
"""
function capon(pa::AbstractPhasedArray, Rxx, f, angles; c=c_0, coords=:azel)
    A = steer(pa, f, angles; c=c, coords=coords)
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

References:
-----------
R. Roy and T. Kailath, ‘ESPRIT-estimation of signal parameters via rotational invariance techniques’, IEEE Trans. Acoust., vol. 37, no. 7, pp. 984–995, Jul. 1989.

H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
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
        Ψ = Ex \ Ey
    end

    Φ = eigvals(Ψ, sortby= λ -> -abs(λ))

    # calculate the directions of arrival (DoAs) from Φ
    k = (2π*f)/c

    # orientation of displacement vector
    ϕ0 = atan(Δ[2], Δ[1])  

    # angle estimates to the left of Δ
    Θ1 = ϕ0 .+ acos.(min.(max.(angle.(Φ) ./ (k * norm(Δ)), -1), 1))
    Θ1 =  mod.(Θ1 .+ π, 2π) .- π

    # angle estimates to the right of Δ
    Θ2 = ϕ0 .- acos.(min.(max.(angle.(Φ) ./ (k * norm(Δ)), -1), 1))
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

References:
-----------
M. Haardt and J. A. Nossek, ‘Unitary ESPRIT: how to obtain increased estimation accuracy with a reduced computational burden’, IEEE Trans. Signal Process., vol. 43, no. 5, pp. 1232–1242, May 1995.
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
        Ψ = C1 \ C2
    end

    Φ = eigvals(Ψ, sortby= λ -> -abs(λ))

    # calculate the directions of arrival (DoAs) from Φ
    Μ = 2atan.(real(Φ))
    k = (2π*f)/c

    # orientation of displacement vector
    ϕ0 = atan(Δ[2], Δ[1])  

    # angle estimates to the left of Δ
    Θ1 = ϕ0 .+ acos.(min.(max.(Μ ./ (k * norm(Δ)), -1), 1))
    Θ1 =  mod.(Θ1 .+ π, 2π) .- π

    # angle estimates to the right of Δ
    Θ2 = ϕ0 .- acos.(min.(max.(Μ ./ (k * norm(Δ)), -1), 1))
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
music(pa::AbstractPhasedArray, Rxx, d, f, angles; c=c_0)

Calculates the MUSIC spectrum for direction of arrival (DoA) estimation.

arguments:
----------
    pa: Array for which the MUSIC spectrum is computed
    Rxx: Covariance matrix of the received signals
    d: Number of signal sources (model order)
    f: Center/operating frequency
    angles: 1xD vector or 2xD matrix of azimuth and elevation angles. For 1xD input, elevation is assumed zero.
    c: Propagation speed of the wave (default: c_0)

References:
-----------
H. Krim and M. Viberg, ‘Two decades of array signal processing research: the parametric approach’, IEEE Signal Process. Mag., vol. 13, no. 4, pp. 67–94, Jul. 1996.
"""
function music(pa::AbstractPhasedArray, Rxx, d, f, angles; c=c_0, coords=:azel)
    U = eigvecs(Rxx, sortby= λ -> -abs(λ))

    Un = U[:, d+1:size(U)[2]]

    A = steer(pa, f, angles; c=c, coords=coords)

    #P = a'*a/(a'*Un*Un'*a)
    P = vec(sum(abs2, A; dims=1) ./ sum(abs2, Un' * A; dims=1))
    return real(P)
end

"""
wsf(pa::AbstractPhasedArray, Rxx, d, DoAs, f; fs=nothing, c=c_0, optimizer=NelderMead(), maxiters=1e3)

DoA estimation using weighted subspace fitting (WSF) .

arguments:
----------
    pa: AbstractPhasedArray to calculate the wsf estimation for
    Rxx: covariance matrix of the array which is used for estimation
    DoAs: vector/matrix of initial DoAs as starting point for WSF
    f: center/operating frequency
    c: propagation speed of the wave
    optimizer: used optimizer to solve the WSF problem
    maxiters: maximum optimization iterations

References:
-----------
M. Viberg and B. Ottersten, ‘Sensor array processing based on subspace fitting’, IEEE Trans. Signal Process., vol. 39, no. 5, pp. 1110–1121, May 1991.

B. Ottersten and M. Viberg, ‘Analysis of subspace fitting based methods for sensor array processing’, in International Conference on Acoustics, Speech, and Signal Processing, Glasgow, UK, 2003.

H. Krim and M. Viberg, ‘Two decades of array signal processing research: the parametric approach’, IEEE Signal Process. Mag., vol. 13, no. 4, pp. 67–94, Jul. 1996.

M. Pesavento, M. Trinh-Hoang, and M. Viberg, ‘Three More Decades in Array Signal Processing Research: An optimization and structure exploitation perspective’, IEEE Signal Process. Mag., vol. 40, no. 4, pp. 92–106, Jun. 2023.
"""
function wsf(pa::AbstractPhasedArray, Rxx, DoAs, f; c=c_0, coords=:azel, optimizer=:prima, maxiters=1e3)
    p = pa, Rxx, f, c
    wsf_cost = function(angles, p)
        pa, Rxx, f, c = p
        d = size(angles, 2)
        Λ, U = eigen(Rxx, sortby= λ -> -abs(λ))

        Λs = diagm(Λ[1:d])
        Λn = diagm(Λ[d+1:length(Λ)])

        Us = U[:, 1:d]
        Un = U[:, d+1:size(U, 2)]

        # estimate noise variance as mean of noise eigenvalues
        # optimal weights as in Ottersten and Viberg ‘Analysis of subspace fitting based methods for sensor array processing’
        σ² = mean(Λn)
        Λest = Λs - σ²*I
        W = Λest^2*inv(Λs)

        #A = hcat(steerphi.(Ref(pa), f, ϕ; fs=fs, c=c, direction=Incoming)...)
        A = steer(pa, f, angles; c=c, coords=coords)
        PA = A*inv(A'*A)*A'

        cost =  tr((I-PA)*Us*W*Us')
        return real(cost)
    end

    if optimizer == :prima
        # prima does not require parameters 
        # but starting points must be a vector
        shape = size(DoAs)
        function obj_func(angles)
            return wsf_cost(reshape(angles, shape), p)
        end
        result, _ = prima(obj_func, vec(DoAs))
        return reshape(result, shape)
    else
        # e.g., optimizer=NelderMead()
        f = OptimizationFunction(wsf_cost, Optimization.AutoForwardDiff())
        p = OptimizationProblem(f, DoAs, p)
        s = solve(p, optimizer; maxiters=maxiters)
        return s.u
    end
end


"""
dml(pa::AbstractPhasedArray, Rxx, DoAs, f; c=c_0, coords=:azel, optimizer=NelderMead(), maxiters=1e3)

DoA estimation using Deterministic Maximum Likelihood (DML).

arguments:
----------
    pa: AbstractPhasedArray to calculate the dml estimator for
    Rxx: covariance matrix of the array which is used for estimation
    DoAs: vector/matrix of initial DoAs as starting point
    f: center/operating frequency
    c: propagation speed of the wave
    optimizer: used optimizer to solve the DML problem
    maxiters: maximum optimization iterations

References:
-----------
H. Krim and M. Viberg, ‘Two decades of array signal processing research: the parametric approach’, IEEE Signal Process. Mag., vol. 13, no. 4, pp. 67–94, Jul. 1996.
"""
function dml(pa::AbstractPhasedArray, Rxx, DoAs, f; c=c_0, coords=:azel, optimizer=:prima, maxiters=1e3)
    p = pa, Rxx, f, c
    dml_cost = function(angles, p)
        pa, Rxx, f, c = p

        A = steer(pa, f, angles; c=c, coords=coords)
        PA = A*inv(A'*A)*A'

        cost =  tr((I-PA)*Rxx)
        return real(cost)
    end

    if optimizer == :prima
        # prima does not require parameters 
        # but starting points must be a vector
        shape = size(DoAs)
        function obj_func(angles)
            return dml_cost(reshape(angles, shape), p)
        end
        result, _ = prima(obj_func, vec(DoAs))
        return reshape(result, shape)
    else
        # e.g., optimizer=NelderMead()
        f = OptimizationFunction(dml_cost, Optimization.AutoForwardDiff())
        p = OptimizationProblem(f, DoAs, p)
        s = solve(p, optimizer; maxiters=maxiters)
        return s.u
    end
end

"""
sml(pa::AbstractPhasedArray, Rxx, DoAs, f; c=c_0, coords=:azel, optimizer=NelderMead(), maxiters=1e3)

DoA estimation using Stochastic Maximum Likelihood (SML).
(Also known as Unconditional Maximum Likelihood (UML))

arguments:
----------
    pa: AbstractPhasedArray to calculate the sml estimator for
    Rxx: covariance matrix of the array which is used for estimation
    DoAs: vector/matrix of initial DoAs as starting point
    f: center/operating frequency
    c: propagation speed of the wave
    optimizer: used optimizer to solve the SML problem
    maxiters: maximum optimization iterations

References:
-----------
H. Krim and M. Viberg, ‘Two decades of array signal processing research: the parametric approach’, IEEE Signal Process. Mag., vol. 13, no. 4, pp. 67–94, Jul. 1996.

H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
"""
function sml(pa::AbstractPhasedArray, Rxx, DoAs, f; c=c_0, coords=:azel, optimizer=:prima, maxiters=1e3)
    p = pa, Rxx, f, c
    sml_cost = function(angles, p)
        pa, Rxx, f, c = p

        A = steer(pa, f, angles; c=c, coords=coords)
        PA = A*inv(A'*A)*A'

        M = size(Rxx, 1)
        d = size(angles, 2)

        # asymptotic Maximum Likelihood (AML) estimator
        cost = log(det(PA*Rxx*PA+tr((I-PA)*Rxx)*(I-PA)/(M-d)))
        return real(cost)
    end

    if optimizer == :prima
        # prima does not require parameters 
        # but starting points must be a vector
        shape = size(DoAs)
        function obj_func(angles)
            return sml_cost(reshape(angles, shape), p)
        end
        result, _ = prima(obj_func, vec(DoAs))
        return reshape(result, shape)
    else
        # e.g., optimizer=NelderMead()
        f = OptimizationFunction(sml_cost, Optimization.AutoForwardDiff())
        p = OptimizationProblem(f, DoAs, p)
        s = solve(p, optimizer; maxiters=maxiters)
        return s.u
    end
end


"""
aic(Rxx, K)

Estimates number of sources using the Akaike information criterion (AIC).

arguments:
----------
    Rxx: covariance matrix of the array which is used for estimation
    K: sample size

References:
-----------
M. Wax and T. Kailath, ‘Detection of signals by information theoretic criteria’, IEEE Trans. Acoust., vol. 33, no. 2, pp. 387–392, Apr. 1985.

H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
"""
function aic(Rxx, K)
    N = size(Rxx, 1)
    λ = real(eigvals(Rxx, sortby= λ -> -abs(λ)))
    λ = max.(λ, 0)

    aic = zeros(N)

    for d in 0:N-1
        λ_n = λ[d+1:end]
        L = ((N-d)K)*log(mean(λ_n)/StatsBase.geomean(λ_n))
        aic[d+1] = L + d*(2N-d)
    end

    # filter out invalid (Inf, -Inf, NaN) results
    orders = (0:N-1)[map(!, (isnan.(aic) .|| isinf.(aic)))]
    aic = aic[map(!, (isnan.(aic) .|| isinf.(aic)))]
    return orders[argmin(aic)]
end

"""
mdl(Rxx, K)

Estimates number of sources using the Minimum Description Length (MDL) model selection criteria.

arguments:
----------
    Rxx: covariance matrix of the array which is used for estimation
    K: sample size

References:
-----------
M. Wax and T. Kailath, ‘Detection of signals by information theoretic criteria’, IEEE Trans. Acoust., vol. 33, no. 2, pp. 387–392, Apr. 1985.

H. L. Van Trees, Optimum array processing. Nashville, TN: John Wiley & Sons, 2002.
"""
function mdl(Rxx, K)
    N = size(Rxx, 1)
    λ = real(eigvals(Rxx, sortby= λ -> -abs(λ)))
    λ = max.(λ, 0)
    mdl = zeros(N)

    for d in 0:N-1
        λ_n = λ[d+1:end]
        L = ((N-d)K)*log(mean(λ_n)/StatsBase.geomean(λ_n))
        mdl[d+1] = L + 0.5*(d*(2N-d)+1)log(K)
    end

    # filter out invalid (Inf, -Inf, NaN) results
    orders = (0:N-1)[map(!, (isnan.(mdl) .|| isinf.(mdl)))]
    mdl = mdl[map(!, (isnan.(mdl) .|| isinf.(mdl)))]
    return orders[argmin(mdl)]
end

# for solving lasso with LeastSquares from ProximalOperators.jl
function value_and_gradient(f::ProximalOperators.LeastSquares, X)
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

References:
-----------
Z. Yang, J. Li, P. Stoica, and L. Xie, ‘Sparse methods for direction-of-arrival estimation’, arXiv [cs.IT], 30-Sep-2016.
"""
function lasso(Y, A, λ=1e-2; maxit=100, tol=1e-6, solver=:sfista)
    f = LeastSquares(A, Y)
    g = NormL21(λ, 2)
    X0 = zeros(ComplexF64, size(A,2), size(Y,2))
    if solver == :ffb
        solve = ProximalAlgorithms.FastForwardBackward(maxit=maxit, tol=tol)
    elseif solver == :sfista
        solve = ProximalAlgorithms.SFISTA(maxit=maxit, tol=tol, Lf = opnorm(A)^2)
    else
        throw(ArgumentError("unsupported solver; got: '$(solver)'"))
    end
    solution, _ = solve(x0=X0, f=f, g=g)
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

References:
-----------
Z. Yang, J. Li, P. Stoica, and L. Xie, ‘Sparse methods for direction-of-arrival estimation’, arXiv [cs.IT], 30-Sep-2016.
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

References:
-----------
V. N. Xuan, K. Hartmann, W. Weihs, and O. Loffeld, ‘Modified orthogonal matching pursuit for multiple measurement vector with joint sparsity in super-resolution compressed sensing’, in 2017 51st Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, USA, 2017.

J. Chen and X. Huo, ‘Sparse representations for multiple measurement vectors (MMV) in an over-complete dictionary’, in Proceedings. (ICASSP ’05). IEEE International Conference on Acoustics, Speech, and Signal Processing, 2005, Philadelphia, Pennsylvania, USA, 2006.
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
ols(Y, A, d)

Orthogonal Least Squares (OLS) DOA estimation. Returns a vector representing the estimated, on-grid, sparse, spatial power spectrum of the signals. Estimated 
DOAs are the angles corresponding to indices of the non-zero values of the output spectrum.

arguments:
----------
    Y: Data matrix of the array
    A: Dictionary matrix of array response vectors from the angle grid 
    d: number of sources

References:
-----------
A. Hashemi and H. Vikalo, "Sparse linear regression via generalized orthogonal least-squares," 2016 IEEE Global Conference on Signal and Information Processing (GlobalSIP), Washington, DC, USA, 2016, pp. 1305-1309, doi: 10.1109/GlobalSIP.2016.7906052.

S. F. Cotter, B. D. Rao, Kjersti Engan and K. Kreutz-Delgado, "Sparse solutions to linear inverse problems with multiple measurement vectors," in IEEE Transactions on Signal Processing, vol. 53, no. 7, pp. 2477-2488, July 2005, doi: 10.1109/TSP.2005.849172.
"""
function ols(Y, A, d)
    r = copy(Y)
    Λ = Int[]

    cost(i) = begin
        Ψ = A[:, [Λ; i]]
        PΨ = Ψ*inv(Ψ'*Ψ)*Ψ'
        return norm((I-PΨ)*Y)^2
    end

    for _ in 1:d
        candidates = setdiff(1:size(A, 2), Λ)
        
        # select N indices that give the smallest cost and add them to the support
        idx = candidates[argmin(cost.(candidates))]
        push!(Λ, idx...)

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

References:
------------
A. J. Barabell, J. Capon, D. F. DeLong, K. D. Senne, and J. R. Johnson, ‘Performance Comparison of Superresolution Array Processing Algorithms’, MIT Lincoln Laboratory, Lexington, MA, May 1984.
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

"""
Automates spectral search for maxima corresponding to DoAs like encountered in MUSIC.
Define a function func(coords...) that returns the power spectrum (or any real valued spectrum)
for the specified coordinates. The d highest maxima will be returned. You must give 
an initial grid for each coordinate axes that is fine enough to resolve each peak. 
"""
function find_doas(func::Function, d::Int, grids...; merge_distance = 0.01)
    function gridsearch(func::Function, grids::Tuple)
        function filter_plateaus(indices)
            N = length(indices[1])
            filtered_indices = []
            indices = Set(indices)

            while !isempty(indices)
                start_idx = pop!(indices)

                # breadth-first search (BFS) to find all connected neighbors
                queue = [start_idx]
                plateau_group = [start_idx]
                min_norm_idx = start_idx
                min_norm = norm(Tuple(start_idx))
                while !isempty(queue)
                    current_idx = popfirst!(queue)
                    for neighbor_offset in CartesianIndices(ntuple(_ -> -1:1, N))
                        # Skip the current index itself
                        if all(Tuple(neighbor_offset) .== 0)
                            continue
                        end

                        # Check if the neighbor is in the set of unvisited indices
                        neighbor_idx = current_idx + neighbor_offset
                        if neighbor_idx in indices
                            push!(plateau_group, neighbor_idx)
                            push!(queue, neighbor_idx)
                            delete!(indices, neighbor_idx)

                            # Update the index with the minimum norm
                            current_norm = norm(Tuple(neighbor_idx))
                            if current_norm < min_norm
                                min_norm = current_norm
                                min_norm_idx = neighbor_idx
                            end
                        end
                    end
                end
                # Add the minimum-norm index from the plateau to the filtered list
                push!(filtered_indices, min_norm_idx)
            end

            return filtered_indices
        end
        grid_indices = CartesianIndices(map(grid -> 1:length(grid), grids))
        grid_data = [func(map( (i, g) -> g[i], Tuple(idx), grids)...) for idx in grid_indices]

        raw_peak_indices = []
        for idx in grid_indices
            is_local_max = true
            current_value = grid_data[idx]

            for neighbor_offset in CartesianIndices(ntuple(_ -> -1:1, length(grids)))
                # Skip the current index itself
                if all(Tuple(neighbor_offset) .== 0)
                    continue
                end

                neighbor_idx = idx + neighbor_offset

                # Ensure neighbor is within grid bounds
                if checkbounds(Bool, grid_data, neighbor_idx)
                    if grid_data[neighbor_idx] > current_value
                        is_local_max = false
                        break
                    end
                end
            end

            if is_local_max
                push!(raw_peak_indices, idx)
            end
        end

        # Filter the raw indices to handle plateaus
        filtered_indices = filter_plateaus(raw_peak_indices)

        # Convert filtered indices back to coordinates and return
        peak_coords = [map((i, g) -> g[i], Tuple(idx), grids) for idx in filtered_indices]

        return filtered_indices, peak_coords
    end

    function refine_peaks(func::Function, peak_indices, grids)

        refined_peak_coords = []
        obj_func(x) = -func(x...)

        for current_index in peak_indices
            current_coords = map((idx, grid) -> grid[idx], Tuple(current_index), grids)

            lower_bounds = map(
                (j, grid) -> grid[max(1, current_index[j] - 1)], 
                1:length(grids), grids
            )
            upper_bounds = map(
                (j, grid) -> grid[min(length(grid), current_index[j] + 1)], 
                1:length(grids), grids
            )

            is_refinable = all(map((l, u) -> l != u, lower_bounds, upper_bounds))

            if is_refinable
                initial_guess = collect(current_coords)
                rhobeg = minimum(upper_bounds - lower_bounds) / 4.0
                result, _ = prima(obj_func, initial_guess; xl=lower_bounds, xu=upper_bounds, rhobeg=rhobeg)
                push!(refined_peak_coords, result)
            else 
                push!(refined_peak_coords, collect(current_coords))
            end
        end

        return refined_peak_coords
    end

    function merge_close_peaks(func::Function, peak_coords, merge_distance)
        merged_peak_coords = []

        for peak_coord in peak_coords
            peak_val = func(peak_coord...)

            is_merged = false
            for (i, (unique_peak, unique_val)) in enumerate(merged_peak_coords)
                # Check if the distance is within the tolerance
                if norm(peak_coord - unique_peak) < merge_distance
                    is_merged = true
                    # Keep the peak with the higher function value
                    if peak_val > unique_val
                        merged_peak_coords[i] = (peak_coord, peak_val)
                    end
                    break
                end
            end

            if !is_merged
                push!(merged_peak_coords, (peak_coord, peak_val))
            end
        end

        return [p[1] for p in merged_peak_coords]
    end

    function select_peaks(func::Function, d::Int, peak_coords)
        # Store peak values and their coordinates as tuples
        peak_info = [(func(p...), p) for p in peak_coords]

        # Sort the list in descending order based on the peak value (the first element of the tuple)
        sort!(peak_info, by = first, rev = true)
        return [p[2] for p in peak_info[1:min(d, end)]]
    end

    initial_peak_indices, _ = gridsearch(func, grids)

    if isempty(initial_peak_indices)
        return Matrix{Float64}(undef, length(grids), 0)
    end

    refined_peak_coords = refine_peaks(func, initial_peak_indices, grids)

    merged_peaks = merge_close_peaks(func, refined_peak_coords, merge_distance)

    top_peaks = select_peaks(func, d, merged_peaks)
    sort!(top_peaks, by = p -> Tuple(p))
    return reduce(hcat, top_peaks)
end

end