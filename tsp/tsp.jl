using ArgMacros
using Dates
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

    for i in 1:numpoints - 1  # -1 bc we don't need to choose a 'next point' for our last point
        current_pt = tour[end]
        distance_to_alternatives = dist_matrix[current_pt,:]
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

function three_opt(tour, dist_matrix)
    tour_inds = 1:length(tour)
    for swap_option in IterTools.subsets(1:length(tour), 3)
        idx1, idx2, idx3 = swap_option
        tour2 = copy(tour)
        tour3 = copy(tour)
        tour4 = copy(tour)
        tour5 = copy(tour)
        tour6 = copy(tour)

        tour2[idx1], tour2[idx2], tour2[idx3] = tour2[idx1], tour2[idx3], tour2[idx2]
        tour3[idx1], tour3[idx2], tour3[idx3] = tour3[idx2], tour3[idx1], tour3[idx3]
        tour4[idx1], tour4[idx2], tour4[idx3] = tour4[idx2], tour4[idx3], tour4[idx1]
        tour5[idx1], tour5[idx2], tour5[idx3] = tour5[idx3], tour5[idx1], tour5[idx2]
        tour6[idx1], tour6[idx2], tour6[idx3] = tour6[idx3], tour6[idx2], tour6[idx1]

        old_dist = dist_around3(tour, idx1, idx2, idx3, dist_matrix)
        dist2 = dist_around3(tour2, idx1, idx2, idx3, dist_matrix)
        dist3 = dist_around3(tour3, idx1, idx2, idx3, dist_matrix)
        dist4 = dist_around3(tour4, idx1, idx2, idx3, dist_matrix)
        dist5 = dist_around3(tour5, idx1, idx2, idx3, dist_matrix)
        dist6 = dist_around3(tour6, idx1, idx2, idx3, dist_matrix)

        min_dist = min(dist2, dist3, dist4, dist5, dist6)
        if min_dist < old_dist
            if dist2 == min_dist
                return (tour2, old_dist - dist2)
            elseif dist3 == min_dist
                return (tour3, old_dist - dist3)
            elseif dist4 == min_dist
                return (tour4, old_dist - dist4)
            elseif dist5 == min_dist
                return (tour5, old_dist - dist5)
            elseif dist6 == min_dist
                return (tour6, old_dist - dist6)
            end
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


function consecutive_nodes(idx1, idx2, tour_length)
    cond1 = abs(idx2 - idx1) == 1
    cond2 = (idx1 == 1 && idx2 == tour_length)
    cond3 = (idx2 == 1 && idx1 == tour_length)
    if cond1 || cond2 || cond3
        return true
    else
        return false
    end
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
    if consecutive_nodes(idx1, idx2, length(tour))
        total_dist = total_dist - dist_matrix[tour[idx1], tour[idx2]]
    end

    return total_dist
end

function dist_around3(tour, idx1, idx2, idx3, dist_matrix)

    d1 = dist_around1(tour, idx1, dist_matrix)
    d2 = dist_around1(tour, idx2, dist_matrix)
    d3 = dist_around1(tour, idx3, dist_matrix)
    total_dist = d1 + d2 + d3

    # If any pair of idx1, idx2, idx3 are consecutive , then total above 
    # double-counts length of edge of that pair
    if consecutive_nodes(idx1, idx2, length(tour))
        total_dist = total_dist - dist_matrix[tour[idx1], tour[idx2]]
    end

    if consecutive_nodes(idx1, idx3, length(tour))
        total_dist = total_dist - dist_matrix[tour[idx1], tour[idx3]]
    end

    if consecutive_nodes(idx2, idx3, length(tour))
        total_dist = total_dist - dist_matrix[tour[idx2], tour[idx3]]
    end

    return total_dist

end


function main()
    TIME_LIMIT = Minute(10)   # Minutes

    # Input File From CLI . Arg Parsing here taken exactly from 
    # example at https://juliahub.com/docs/ArgMacros/aUDMk/0.1.3/
    @beginarguments begin
        @positionalrequired String input_file "input_file"
        @argtest input_file isfile "The input must be a valid file" # Confirm that the input file really exists
    end
    points = readdlm(input_file, ' ', Float64, skipstart = 1)

    # Calc distance matrix and generate initial (greedy) tour
    dist_matrix = arr_points_to_dist_matrix(points)
    tour = greedy_salesman(dist_matrix)
    #println("Initial greedy tour: $tour")
    #println("Initial greedy tour total dist: $(calc_tour_dist(tour, dist_matrix))")

    # Improve solution with local moves
    lm3 = (tour, 0)
    start = Dates.now()
    time_elapsed = Minute(0)
    while !isnothing(lm3) && (time_elapsed < TIME_LIMIT)
        time_elapsed = Dates.now() - start
        if time_elapsed > Minute(TIME_LIMIT)
            #println("Stopping local moves due to time elapsed > $TIME_LIMIT")
        end

        lm2 = two_opt(tour, dist_matrix)
        if isnothing(lm2)
            #println("2-opt hit local optimum, proceeding to 3-opt")
            lm3 = three_opt(tour, dist_matrix)
            if isnothing(lm3)
                #println("Hit local optimum")
            else
                tour, dist_improved = lm3
                #println("3-opt improved tour distance by $dist_improved")    
            end
        else
            tour, dist_improved = lm2
            #println("2-opt improved tour distance by $dist_improved")
        end
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

# Notes:
# prob4 took about 4.5 - 5 minutes on 'time_limit=2minutes' (that would include write data and generate initial greedy solution though)
# 

# final thoughts - on e.g. prob5, hits 2-opt local optimum around a few minutes 
# in with 5 minute time limit, sets off on 3-opt, 5 minutes later no 
# more log lines. maybe for the huge problems better with 2-opt
# with more simulated annealing or restarts. for smaller problems
# maybe better to keep it going with 3-opt. killed at 18-19 minutes


# TODO
    # 12) add simulated annealing?
        # potentially take slightly negative solutions
        # more likely to reject slightly positive solutions
    #    add simultaneous processing from different starting places?
    #    show log output somewhere while running?
    #       # maybe best to just do that by having logging statements so can see log when call
    #       tsp.jl directly from shell, but those extra logging off by default when python/submit calls it

    # DONE
# 1) get points from csv (skipping first line)
# 2) get_distance_matrix() from points
    # 2a - func for dist between two points
    # 2b - run 2b for each point against each other point
# 3) greedy_salesman(dist_matrix, startnode)
# 4) two_opt(tour, dist_matrix)
#   - requires: dist_around(tour, nodeidx, dist_matrix)
    # 5) run it 
    # 7) call it from python with subprocess
    # 8) submit

    # 9) call julia from solver.py passing along input datafile arg
    # 9b) implement cli arg from julia side on input data path
    # 10) add time-limit - 29 minutes per problem - return current tour at that point
    # 11) Add 3-opt



    # Calculate distance matrix. Optionally save to csv disk for debugging
# distance_matrix = arr_points_to_dist_matrix(points)

    # get starting tour using regular greedy
# best_tour = None
# best_dist = np.inf                                                                