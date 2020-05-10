using ArgParse
using DelimitedFiles # :readdlm   # CSVFiles
using IterTools


# Create Distance Matrix from points
function dist(x1, y1, x2, y2) 
    return sqrt((x2 - x1)^2 + (y2 - y1)^2)
end

function arr_points_to_dist_matrix(points)
    # initialize dist matrix of approp size 
    numpoints = size(points)[1]
    dist_matrix = zeros(Float64, numpoints, numpoints)

    # calc distance for each pair of points
    for (i, pt1) in enumerate(eachrow(points))
        for (j, pt2) in enumerate(eachrow(points))
            dist_matrix[i,j] = dist(pt1[1], pt1[2], pt2[1], pt2[2])
        end
    end
    return dist_matrix
end

function calc_tour_dist(tour, dist_matrix)
    total_dist = sum([dist_matrix[i[1], i[2]] for i in zip(tour[1:end - 1], tour[2:end])])
    total_dist += dist_matrix[tour[1], tour[end]]
end

function greedy_salesman(dist_matrix, startnode = 1)
    tour = [startnode]
    numpoints = size(dist_matrix)[1]

    for i in 1:numpoints - 1  # we don't need to choose a 'next point' for our last point
        current_pt = tour[end]
        distance_to_alternatives = dist_matrix[i,:]
        # dont wanna select a point we've already visited
        distance_to_alternatives[tour] .= Inf
        # greedy - pt with min dist among possible choices
        next_pt = argmin(distance_to_alternatives)
        append!(tour, next_pt)
    end
    return tour
end

function two_opt(tour, dist_matrix)
    tour_inds = 1:length(tour)
    for swap_option in IterTools.subsets(1:length(tour), 2)
        # println("trying swap_option $swap_option")
        idx1, idx2 = swap_option
        tour2 = copy(tour)
        tour2[idx1], tour2[idx2] = tour2[idx2], tour2[idx1]
        
        old_dist = dist_around2(tour, idx1, idx2, dist_matrix)
        new_dist  = dist_around2(tour2, idx1, idx2, dist_matrix)
        dist_improvement = old_dist - new_dist  # if old<new, dist lessened (improved)

        if dist_improvement > 0
            return (tour2, dist_improvement)
        end
    end
    return nothing
end


"""
dist_around1(tour, idx, dist_matrix)

Compute the aggregate distance across both edges linked to node idx, in tour
"""
function dist_around1(tour, nodeidx, dist_matrix)

    # For modulo arithmetic where our lower bound is 1 instead of 0, we 
    # subtract one, calculate the remainder, then add the one back
    # (for prev, we also add n before % to avoid taking modulo of neg number)
    # thanks to: https://stackoverflow.com/a/3803412/1870832 for the elegant one-liners)
    n = length(tour)
    next_idx = (nodeidx - 1 + 1) % n + 1
    prev_idx = (nodeidx - 1 - 1 + n) % n + 1

    d1 = dist_matrix[tour[prev_idx], tour[nodeidx]]
    d2 = dist_matrix[tour[nodeidx], tour[next_idx]]
    return d1 + d2

end

"""
dist_around2(tour, idx1, idx2, dist_matrix)

Compute the aggregate distance across all edges linked to nodes at idx1 or idx2 in tour
"""
function dist_around2(tour, idx1, idx2, dist_matrix)

    d1 = dist_around1(tour, idx1, dist_matrix)
    d2 = dist_around1(tour, idx2, dist_matrix)
    total_dist = d1 + d2

    # If idx1, idx2 are consecutive , then d1+d2 above double-counts
    # length of edge from idx1 to idx2
    if abs(idx2 - idx1) == 1 || (idx1 == 1 && idx2 == length(tour))
        total_dist = total_dist - dist_matrix[tour[idx1], tour[idx2]]
    end

    return total_dist
end

function main()

    # Input File
    INPUT_FILE = "data/tsp_51_1" # "data/for_jl_solver" #
    points = readdlm(INPUT_FILE, ' ', Float64, skipstart = 1)


    # Calc distance matrix and generate initial (greedy) tour
    dist_matrix = arr_points_to_dist_matrix(points)
    display(dist_matrix)
    writedlm("dist_matrix_jl.csv", dist_matrix, ',')


    tour = greedy_salesman(dist_matrix)
    println("Initial greedy tour: $tour")
    println("Initial greedy tour total dist: $(calc_tour_dist(tour, dist_matrix))")

    # Improve solution with local moves
    NUM_MOVES = 1000
    n = 1
    local_move = (tour, 0)
    while n < NUM_MOVES && !isnothing(local_move)  
        local_move = two_opt(tour, dist_matrix)
        # println("local_move is $local_move")
        if isnothing(local_move)
            # println("Hit local optimum after $n moves")
        else
            tour, dist_improved = local_move
            # println("Improved tour distance by $dist_improved")
        end
        n += 1
    end

    # Format output as desired by course grader
    proved_opt = 0
    tour_dist = calc_tour_dist(tour, dist_matrix)
    tour_0idx = [t - 1 for t in tour]
    tour_fmt = join(tour_0idx, " ")

    println("$tour_dist $proved_opt")
    println("$tour_fmt")

    return nothing
end

main()

# BUG: why is tour I'm submitting worse than original greedy (should be 506)?
# ...NOTE: says initial greedy tour dist is 1313...
# TODO: check what initial greedy tour python code comes up with (need to convert 1-idx to 0-idx)
# Julia greedy  starts: [0, 33, 1,  (after converting to 0-indexed)
# Python greedy starts: [0, 33, 5,
# so third step, python goes from 33 to 5, julia goes from (0-idxd) 33 to 1

# TODO: compare dist matrices between julia and python code
#   for tsp_5_1 RESULTS - same dist matrix but python one has more decimal precision
#               UPDATE: same precision, can see when write to csv. display(m) in jl is just rounding
#   for $prob1
#     UPDATE: looks like a bug in jl greedy salesman. from 33, chooses 1, 14.036 away.
#                                                              but 5, just like in py dist matrix, is closer in jl dist matrix (both say its 9.20 away)


# TODO
    # 7) call it from python with subprocess
    # 8) submit
    # 9) add three-opt

    # DONE
# 1) get points from csv (skipping first line)
# 2) get_distance_matrix() from points
    # 2a - func for dist between two points
    # 2b - run 2b for each point against each other point
# 3) greedy_salesman(dist_matrix, startnode)
# 4) two_opt(tour, dist_matrix)
#   - requires: dist_around(tour, nodeidx, dist_matrix)
    # 5) run it 



    # Calculate distance matrix. Optionally save to csv disk for debugging
# distance_matrix = arr_points_to_dist_matrix(points)

    # get starting tour using regular greedy
# best_tour = None
# best_dist = np.inf