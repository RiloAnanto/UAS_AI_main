import matplotlib.pyplot as plt
from numpy import mean
import numpy as np
import ga

no_variables = 2
pop_size = 2
crossover_rate = 2
mutation_rate = 2
no_generations = 20
lower_bounds = [0, 0]
upper_bounds = [100, 100]
step_size = 2
rate = 2
pop = np.zeros((pop_size,no_variables))
for s in range(pop_size):
    for h in range(no_variables):
        pop[s,h] = np.random.uniform(lower_bounds[h],upper_bounds[h])

extended_pop = np.zeros((pop_size+crossover_rate+mutation_rate+2*no_variables*rate,pop.shape[1]))
while extended_pop.all() > 100:
    extended_pop = np.zeros((pop_size + crossover_rate + mutation_rate + 2 * no_variables * rate, pop.shape[1]))

#visualization
fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title('Economic Growth')
plt.xlabel("Iteration")
plt.ylabel("Growth Percentage")
A = []
B = []
a=2 #adaptive restart
g=0
global_best = pop
k=0
while g <= no_generations:
    for i in range(no_generations):
        offspring1 = ga.crossover(pop, crossover_rate)
        offspring2 = ga.mutation(pop, mutation_rate)
        fitness = ga.objective_function(pop)
        offspring3 = ga.local_search(pop, fitness, lower_bounds, upper_bounds, step_size, rate)
        step_size = step_size*0.98
        if step_size < 1:
            step_size = 1
        extended_pop[0:pop_size] = pop
        extended_pop[pop_size:pop_size+crossover_rate] = offspring1
        extended_pop[pop_size+crossover_rate:pop_size+crossover_rate+mutation_rate] = offspring2
        extended_pop[pop_size+crossover_rate+mutation_rate:pop_size+crossover_rate+mutation_rate+2*no_variables*rate] = offspring3
        fitness = ga.objective_function(extended_pop)
        index = np.argmax(fitness)
        current_best = extended_pop[index]
        pop = ga.selection(extended_pop,fitness,pop_size)

        print("Generation: ", g, ", growth percentage: ", max(fitness)-100)

        A.append(max(fitness)-100)
        B.append(mean(fitness)-100)
        g +=1

        if i >= a:
            if sum(abs(np.diff(B[g-a:g])))<=0.01:
                fitness = ga.objective_function(pop)
                index = np.argmax(fitness)
                current_best = pop[index]
                pop = np.zeros((pop_size, no_variables))
                for s in range(pop_size):
                    for h in range(no_variables):
                        pop[s, h] = np.random.uniform(lower_bounds[h], upper_bounds[h])
                pop[0] = current_best #keep the best
                step_size = 0.2
                global_best[k] = current_best
                k +=1
                break

        #Visualization
        # ax.plot(A, color='r')
        ax.plot(B, color='b')
        fig.canvas.draw()
        ax.set_xlim(left=max(0, g - no_generations), right=g+3)
        if g > no_generations:
            break
    if g > no_generations:
        break

plt.show()
