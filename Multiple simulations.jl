#Obtain convergence information for a specified alpha and beta. Runs many simulations to
#collect info on average market price, outputs, profit and convergence iterations
using Pkg
#Pkg.add("Random")
#Pkg.add("Plots")
#Pkg.add("Distributions")
#Pkg.add("StatsPlots")
#Pkg.add("StatsBase")
#Pkg.add("Dictionaries")
#Libraries
using Random
using Plots
using Distributions
using StatsPlots
using StatsBase
using Dictionaries

#Agent definition
mutable struct QL_Agent
    epsilon::Float64 #exploration rate. This value gets updated
    α::Float64 #step size, learning rate
    γ::Float64 #discount rate
    q_table
    strategy_matrix
    max_val_q_matrix
    action
    current_reward
    prev_state
    change
    stable_count
end

function ql_agent(epsilon, α, γ, q_table, strategy_matrix, max_val_q_matrix, action = missing, current_reward = missing, prev_state = missing, change = false, stable_count = 0)
    return QL_Agent(epsilon, α, γ, q_table, strategy_matrix, max_val_q_matrix, action, current_reward, prev_state, change, stable_count)
end

#Parameters
iterations = 10_000_000
low_demand = 290.0
high_demand = 310.0
demand_levels = [low_demand, high_demand]
min_quantity = 70
max_quantity = 105
step_size = 2.5
states = collect(low_demand - 2*max_quantity:step_size:high_demand - 2*min_quantity)
actions = collect(min_quantity:step_size:max_quantity)
α = 0.15 #step size, learning rate
γ = 0.95 #discount rate
β = 0.000004 #0.00005 for 50k reps, 0.000004 for 500k+ reps, 0.000004 for 2M
num_agents = 2
stopping_rule = 100_000 #iterations without modification of strategy matrix

#PECOMPUTATIONS TO SPEED UP CODE
#precomputed tables
output_table = zeros(length(actions), length(actions), 2)
for (index1, value1) in enumerate(actions)
    for (index2, value2) in enumerate(actions)
        for (index3, demand) in enumerate(demand_levels)
            output_table[index1, index2, index3] = demand - (value1 + value2)
        end
    end
end

profit_table = zeros(length(states), length(actions))
for (index1, state) in enumerate(states)
    for (index2, action) in enumerate(actions)
        profit_table[index1, index2] = state*action
    end
end

next_state_dict = Dict(i => states[i] for i = 1:length(states))
reverse_dic = Dict(value => key for (key, value) in next_state_dict)
#precomputed ranges
range_1_to_length_actions = 1:length_actions
range_1_to_num_agents = 1:num_agents
#other precomputations to speed up
beta_cons = exp(-β)
length_actions = length(actions)


#to collect info
profits = zeros(iterations, num_agents)
quantities = zeros(iterations, num_agents)
prices = zeros(iterations)
simulations = 10
iterations_until_convergence = fill(iterations, simulations) #if it stays in 10M, it means it didnt converge

#arguments to initialize the agent
epsilon = 1.0 #initial exploration rate. expects float
initialization = 0 #231000
q_table1 = fill(Float64(initialization), length(states), length(actions))
q_table2 = fill(Float64(initialization), length(states), length(actions))
strategy_matrix = fill(1, length(states))
max_val_q_matrix = zeros(length(states))

@time for simulation in 1:simulations
    #SIMULATION
    println("")
    println("Simulation ", simulation)
    println("")
    #initiate agents
    Agent1 = ql_agent(epsilon, α, γ, q_table1, strategy_matrix, max_val_q_matrix)
    Agent2 = ql_agent(epsilon, α, γ, q_table2, strategy_matrix, max_val_q_matrix)
    agents = [Agent1, Agent2]

    for iteration in 1:iterations
        ##### PERIOD T #####
        #all agents select output at same time
        for agent in agents
            if rand() < agent.epsilon
                #rand much faster than sample (benchmarked)
                agent.action = rand(range_1_to_length_actions)
            else
                agent.action = agent.strategy_matrix[agent.prev_state]
            end
        end
        ##### PERIOD T+1 #####

        #calculate market price
        if rand() > 0.5
            demand = 2
        else
            demand = 1
        end
        mrkt_price = output_table[Agent1.action, Agent2.action, demand]

        #state t+1
        state = reverse_dic[mrkt_price]
        #reward (profit)
        for agent in agents
            agent.current_reward = profit_table[state, agent.action]
        end

        #update Q table cell. Skip first update as prev_state is missing.
        #also update of strategy matrix, to make use of computed target
        if iteration != 1
            for agent in agents
                #UPDATE Q VALUES
                target_difference = agent.current_reward + agent.γ*agent.max_val_q_matrix[state] - agent.q_table[agent.prev_state, agent.action]
                agent.q_table[agent.prev_state, agent.action] += agent.α * (target_difference)

                #UPDATE OF STRAT MATRIX
                #PART 1
                #if the action selected was the one suggested by the strategy matrix but the q_value has decreased in value,
                #then update max_q_value for that specific state, and if it goes below the second highest, update strategy matrix
                if (agent.action == agent.strategy_matrix[agent.prev_state])
                    #instead of checking all values, we check if the current highest value is the same as the previous, and the q value
                    #has been negatively updated. Only in that case it makes sense to check whether the highest index is still the same,
                    #and if not, update strategy matrix
                    if (target_difference < 0)
                        #argmax 10 times faster... but can we really do it? It means that if there is a tie, we take the first
                        max_index = argmax(agent.q_table[agent.prev_state, :])
                        #findall
                        #max_index = sample(findall(x->x==maximum(agent.q_table[agent.prev_state, :]), agent.q_table[agent.prev_state, :]))
                        if (max_index != agent.action) #there has been a change in max action
                            agent.change = true #there has been a change in strategy_matrix. Info for stopping rule
                            #update max q value matrix and strat matrix with max index
                            agent.strategy_matrix[agent.prev_state] = max_index
                            agent.max_val_q_matrix[agent.prev_state] = agent.q_table[agent.prev_state, max_index]
                        else #max value has decreased but it is still the same action
                            agent.max_val_q_matrix[agent.prev_state] = agent.q_table[agent.prev_state, agent.action]
                        end
                    else
                        #update max q value matrix with increased q value of same action
                        agent.max_val_q_matrix[agent.prev_state] = agent.q_table[agent.prev_state, agent.action]
                    end
                    #update max q value matrix
                    agent.max_val_q_matrix[agent.prev_state] = agent.q_table[agent.prev_state, agent.action]

                #PART 2
                #if the action selected was not the one suggested by the strategy matrix, the new q update may obtain a higher
                #value than the past max q value. Only in that case update both strategy and max q table
                else
                    if agent.q_table[agent.prev_state, agent.action] > agent.max_val_q_matrix[agent.prev_state]
                        agent.change = true
                        agent.strategy_matrix[agent.prev_state] = agent.action
                        agent.max_val_q_matrix[agent.prev_state] = agent.q_table[agent.prev_state, agent.action]
                    end
                end
            end
        end

        #update class: prev_state
        for agent in agents
            agent.prev_state = state
            agent.epsilon *= beta_cons
        end

        #store information as averages
        for agent in range_1_to_num_agents
            profits[iteration, agent] += agents[agent].current_reward
            quantities[iteration, agent] += actions[agents[agent].action]
        end

        prices[iteration] += mrkt_price

        #Stopping rule
        for agent in agents
            if agent.change
                agent.stable_count = 0
            else
                agent.stable_count += 1
            end
            agent.change = false
        end

        if (Agent1.stable_count > stopping_rule) & (Agent2.stable_count > stopping_rule)
            iterations_until_convergence[simulation] = iteration
            println("Stopping rule reached at iteration ", iteration)
            break
        end
        #control flow. Stop every x iterations
        #modulus benchmarked, it only takes 1 nano second to run it 2M times
        if iteration%100000 == 0
            println("Iteration ", iteration)
            println("Agent 1 stable count: ", Agent1.stable_count)
            println("Agent 2 stable count: ", Agent2.stable_count)
        end
    end
end

#Post compute averages
#compute how many simulations have reached each iteration
simulations_completed = [count(x->(x>i),iterations_until_convergence) for i in 1:iterations]
max_simulation = findfirst(isequal(0),simulations_completed)-1
profits_red_1 = profits[1:max_simulation,1] ./ simulations_completed[1:max_simulation]
quantities_red_1 = quantities[1:max_simulation,1]  ./ simulations_completed[1:max_simulation]
prices_red = prices[1:max_simulation]  ./ simulations_completed[1:max_simulation]

#Plots
#Evolution of profits, output, prices
plot_profits = histogram2d(1:4_000_000, profits_red_1[1:4_000_000], show_empty = true, nbins = (400,600), legend=false)
plot_output = histogram2d(1:4_000_000,  quantities_red_1[1:4_000_000], show_empty = true, nbins = (600,900), legend=false, yticks=(70:2.5:90))
plot_prices = histogram2d(1:4_000_000,  prices_red[1:4_000_000], show_empty = true, nbins = (600,900), legend=false)

#Distribution of iterations until convergence
max_iter = sum(iterations_until_convergence .== 2)
d = kde(sort(iterations_until_convergence)[1:simulations-max_iter]) #remove simulations that didnt converge
plot_convergence_density = plot(d.x, d.density,
    xlabel = "Iterations",
    ylabel = "Density",
    title = "PDF: iterations until convergence",
    legend = false,
    fill = (0, :dodgerblue),
    alpha = 0.5,
    xlims=(0,10500000)
)

plot_convergence_cdf = plot(ecdf(iterations_until_convergence),
    xlabel = "Iterations",
    ylabel = "Density",
    title = "CDF: iterations until convergence",
    legend = false,
    fill = (0, :dodgerblue),
    alpha = 0.5,
    xlims=(0,10500000)
)
plot_convergence_hist = histogram(sort(iterations_until_convergence)[1:simulations-max_iter])

#Save plots
plots_to_save = [plot_profits,plot_output,plot_prices,plot_convergence_density,plot_convergence_cdf,plot_convergence_hist]
plots_names = ["plot_profits","plot_output","plot_prices","plot_convergence_density","plot_convergence_cdf","plot_convergence_hist"]

co = 0
for plot in plots_to_save
    co += 1
    savefig(plot, "/Users/rubenmartinezcuella/Desktop/Unibo/2 year/Master thesis/Dissertation/Julia code/Results simulation/$(plots_names[co]).png")
end
