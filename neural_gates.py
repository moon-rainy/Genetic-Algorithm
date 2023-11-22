from random import randrange, random, uniform


'''

		单个神经元: 两个输入值

                  bias
                   |
                   |
(Input1)*-----*(Hidden1)*      bias
         \   /           \      |
          \ /             \     |
           *      bias     *(Output1)
          / \      |      /
         /   \     |     /
(Input2)*-----*(Hidden2)*



	参数的定义:

 Hidden Weights                                          Hidden Biases                   Output Weights              Output Biases
 ======================================================  ==============================  ==========================  ==============
[<Hidden1_W1>, <Hidden1_W2>, <Hidden2_W1>, <Hidden2_W2>, <Hidden1_Bias>, <Hidden2_Bias>, <Output1_W1>, <Output1_W2>, <Output1_Bias>]

'''


def initialization(size: int) -> list[list[float]]:
	'''
	使用随机或启发式方法创建潜在解决方案（个体）的初始种群。
	'''
	if size % 2 != 0:
		raise ValueError("Size must evenly divisible by 2.")
	# Create population with random network values as small floats.
	population = []
	for i in range(size):
		individual = []
		for f in range(9):
			individual.append(uniform(-1, 1))
		population.append(individual)
	return population


def forward_prop(network: list[float], inputs: list[float]) -> float:
	hidden = [0.0, 0.0]
	# 	计算隐藏层神经元的激活函数:
	#           ReLu(    bias +       (inputs * weights                               ) )
	#           -------- ------------ ------------------------------------------------- -
	hidden[0] = max(0.0, network[4] + (inputs[0] * network[0] + inputs[1] * network[1]) )
	hidden[1] = max(0.0, network[5] + (inputs[0] * network[2] + inputs[1] * network[3]) )
	# 	计算输出神经元的激活函数:
	#      ReLu(    bias +       (inputs * weights                               ) )
	#      -------- ------------ ------------------------------------------------- -
	return max(0.0, network[8] + (hidden[0] * network[6] + hidden[1] * network[7]) )


def fitness_func(individual: list[float], inputs: list[list[float]], outputs: list[float]) -> float:
	# 类似于批量学习，运行所有输入以获得总误差。
	error = 0.0
	for input, expected  in zip(inputs, outputs):
		output = forward_prop(individual, input)
		error += abs(output - expected)
	return error


def fitness_evaluation(population: list[list[float]], inputs: list[list[float]], outputs: list[float]) -> None:
	'''
	根据定义的适应性函数评估种群中每个个体的适应性，该函数衡量它们解决问题的能力有多好。
	'''
	# Sort the population from most to least fit (small error = fit).
	population.sort(key=lambda i: fitness_func(i, inputs, outputs))


def selection(population: list[list[float]]) -> list[list[float]]:
	'''
	根据个体的适应性从当前种群中选择个体。适应性值较高的个体更有可能被选中。
	'''
	# Select the first half of the population (most fit individuals).
	return population[:len(population) // 2]


def crossover(parents: list[list[float]]) -> list[list[float]]:
	'''
	通过结合选定父母的基因信息创建新的个体（后代）。在基因材料中选择交叉点以交换信息。
	'''
	# Create equal number of offspring from parents.
	offspring = []
	for p in range(len(parents)):
		# 随机选择父代.
		male = parents[randrange(0, len(parents))]
		female = parents[randrange(0, len(parents))]
		# 随机进行交叉操作.
		point = randrange(1,8)
		genetics = male[:point] + female[point:]
		offspring.append(genetics)
	return offspring


def mutation(offspring: list[list[float]], rate: float) -> None:
	'''
	对所选个体的基因材料进行小范围的随机变化，生成新的解决方案（后代）。
	'''
	for i in range(len(offspring)):
		# Create new mutated genetics for each offspring.
		for v in range(9):
			# Randomly mutate offspring vals at the given rate.
			if random() <= rate:
				offspring[i][v] += uniform(-1, 1)


def replacement(population: list[int], offspring: list[int]) -> None:
	'''
	用新创建的后代替换当前种群中的一些个体，以保持种群的规模。
	'''
	# Replace second half of the population (least fit individuals).
	population[len(population) // 2:] = offspring


def termination(population: list[list[float]], inputs: list[list[float]], outputs: list[float], error: float) -> tuple[float, bool]:
	'''
	检查是否满足终止准则，例如达到最大的迭代次数或达到令人满意的解决方案。
	'''
	# Return population fitness is at or above a certain level.
	fitness = sum([fitness_func(i, inputs, outputs) for i in population]) / len(population)
	return fitness, fitness <= error


def genetic_algorithm(size: int, generations: int, inputs: list[list[float]], outputs: list[float], rate: float, error: float):
	gen = 0
	term = False
	# Step 1:
	population = initialization(size)
	while gen < generations and not term:
		# Step 2:
		fitness_evaluation(population, inputs, outputs)
		# Step 3:
		selected = selection(population)
		# Step 4:
		offspring = crossover(selected)
		# step 5:
		mutation(selected, rate)
		# Step 6:
		replacement(population, offspring)
		# Step 7:
		err, term = termination(population, inputs, outputs, error)
		print("Generation %d, Error: %f" % (gen, err))
		gen += 1
		for input, expected in zip(inputs, outputs):
			output = forward_prop(population[1], input)
			print("%f %f = %f (%f)" % (input[0], input[1], output, expected))
	if err < error:
		print(f"由于误差为{err:.5f}小于{error},所以提前退出，迭代代数为{gen}代")


inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

OR = [0.0, 1.0, 1.0, 1.0]
NOR = [1.0, 0.0, 0.0, 0.0]
AND = [0.0, 0.0, 0.0, 1.0]
NAND = [1.0, 1.0, 1.0, 0.0]
#genetic_algorithm(512, 1024, inputs, OR, 0.05, 0.2)

XOR = [0.0, 1.0, 1.0, 0.0]
genetic_algorithm(512, 1024, inputs, AND, 0.05, 0.2)
