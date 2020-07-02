# Octahedron Stein Distribution
#
# Represents a uniform distribution over an octahedron on the integer lattice
# in three dimensions, i.e., a scaled l1 ball in R^3 intersected with Z^3.
# The vertices of the octahedron are at (0,0,ulim), (0,0,-ulim), (0,ulim,0), etc.

using StatsBase: wsample

type SteinOctahedron <: SteinDistribution
    ulim::Int64
    c1::Array{Float64}
    c2::Array{Float64}
    c3::Array{Float64}
end

SteinOctahedron(ulim::Int64) = SteinOctahedron(
    ulim,
    [1.0 for i=1:3],
    [1.0 for i=1:3],
    [1.0 for i=1:3]
);

# Draw n independent samples from distribution via rejection sampling
function rand(d::SteinOctahedron, n::Int64)
    x = zeros(n,3);
    i = 0;
    while i < n
        y = rand(-d.ulim:d.ulim,3);
        if sum(abs(y)) <= d.ulim
            i += 1;
            x[i,:] = y;
        end
    end
    x
end

# Joint distribution evaluated at a single point (unnormalized)
function jointdistribution(d::SteinOctahedron, y::Array{Float64})
    sum(abs(y)) <= d.ulim ? 1.0 : 0.0
end

# Conditional distribution of jth coordinate of y, given all other coordinates,
# represented as a probability vector
function conddistribution(d::SteinOctahedron, y::Array{Float64,1}, j::Int64)
    conddist = [jointdistribution(d, [y[1:(j-1)], yj, y[(j+1):end]])
                for yj in -d.ulim:d.ulim]
    conddist / sum(conddist)
end

function supportlowerbound(d::SteinOctahedron, j::Int64)
    -d.ulim
end

function supportupperbound(d::SteinOctahedron, j::Int64)
    d.ulim
end

function numdimensions(d::SteinOctahedron)
    3
end
