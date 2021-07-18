#Obtain convergence information for a specified alpha and beta. Runs a single simulation
#and colects info on the evolution of the actions of the agents

#Libraries
using Random
using Plots
using Distributions
using StatsPlots
using StatsBase
using DataStructures
using DataFrames

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
iterations = 5_000_000
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
beta_cons = exp(-β)
length_actions = length(actions)
stopping_rule = Inf #we want to plot all evolution

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

#How many grpahs
simulations = 1

#collect info

actions_done = zeros(length(actions), num_agents)
actions_vs_state = zeros(iterations+1, num_agents + 1)
iterations_until_convergence = NaN #if it stays in 10M, it means it didnt converge


#arguments to define the agent
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
    #Agent1 = init_agent(copy(trained_algos[1,1]))
    #Agent2 = init_agent(copy(trained_algos[2,1]))
    agents = [Agent1, Agent2]
    #begin game
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
        actions_vs_state[iteration+1, 1] = state

        for agent in range_1_to_num_agents
            actions_done[agents[agent].action, agent] += 1
            actions_vs_state[iteration, agent+1] = agents[agent].action
        end

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
            iterations_until_convergence = iteration
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

#Animations actions and states
#technical details
to_merge = [collect(1:15) zeros(15)]

#animation all simulation
frames = 500
n_points_per_frame = Int(iterations / frames)
anims_all = @animate for i = 0:(frames-1) #plot 3.5M iterations
    #iterator
    origin = 1 + i*n_points_per_frame
    final = n_points_per_frame * (i+1)
    #title
    title = plot(title = "Evolution of actions: $(i*n_points_per_frame+10000)", grid = false, showaxis = false, bottom_margin = -50Plots.px)
    #actions vs state
    to_plot_st_vs_ac = actions_vs_state[origin:final,:]
    hists = histogram2d(to_plot_st_vs_ac[:,1], [to_plot_st_vs_ac[:,2],to_plot_st_vs_ac[:,3]], nbins = (1:length(states), 1:length(actions)), layout = (2,1), legend = [false false], show_empty=true)
    yticks!(collect(1:2:15),["70","75","80","85","90","95","100","105"])
    xticks!(collect(1:4:37),["80","90","100","110","120","130","140","150","160", "170"])
    xlabel!("Market price")
    ylabel!("Output")
    #actions
    #agent 1
    partial_dic_1 = counter(actions_vs_state[origin:final,2])
    partial_1 = Array(hcat([[key, val] for (key, val) in partial_dic_1]...)')
    merged = copy(to_merge)
    c=0
    for j in partial_1[:,1]
        c += 1
        merged[Int(j), 2] = partial_1[c,2]
    end
    t = merged[sortperm(merged[:, 1]), :]
    bar_1 = bar(t, orientation=:h, xticks=(0:1000:n_points_per_frame), yticks=(1:1:15), ylims = (1,15), xlims = (0,n_points_per_frame), xaxis=false, grid=false, xguidefont = font(:white))
    yticks!(collect(1:2:15),["70","75","80","85","90","95","100","105"])
    xlabel!("This is to align with left plot")
    #agent 2
    partial_dic_2 = counter(actions_vs_state[origin:final,3])
    partial_2 = Array(hcat([[key, val] for (key, val) in partial_dic_2]...)')
    merged = copy(to_merge)
    c=0
    for j in partial_2[:,1]
        c += 1
        merged[Int(j), 2] = partial_2[c,2]
    end
    t = merged[sortperm(merged[:, 1]), :]
    bar_2 = bar(t, orientation=:h, xticks=(0:1000:n_points_per_frame), yticks=(1:1:15), ylims = (1,15), xlims = (0,n_points_per_frame), xaxis=false, grid=false, xguidefont = font(:white))
    yticks!(collect(1:2:15),["70","75","80","85","90","95","100","105"])
    xlabel!("This is to align with left plot")
    #combine agents
    acts = plot(bar_1, bar_2, layout = (2,1), legend = [false false])

    #all in 1 graph
    plot(title, hists, acts, layout = @layout([A{0.07h}; [B C]]))
end
gif(anims_all)
gif(anims_all, "/Users/rubenmartinezcuella/Desktop/Unibo/2 year/Master thesis/Dissertation/Julia code/Results action selection/evolutionppt.gif", fps = 10)
