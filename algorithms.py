from organism import *
from population import *
import numpy as np
import math
from PIL import Image, ImageOps
import random
import csv

class Algorithm():
	def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory):
		self.goal = goal
		self.goalpx = np.array(goal)
		self.w = w
		self.h = h
		self.num_poly = num_poly
		self.num_vertex = num_vertex
		self.comparison_method = comparison_method
		self.data = []
		self.savepoints = savepoints
		self.outdirectory = outdirectory

	def save_data(self, row):
		# function to be called every generation to remember data on that generation
		self.data.append(row)

	def write_data(self):
		# writes all collected data to a file
		with open(self.outdirectory + '/data.csv', 'a', newline='') as f:
			writer = csv.writer(f)
			for row in self.data:
				writer.writerow(row)

class Hillclimber(Algorithm):
	def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory, iterations):
		super().__init__(goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory)
		self.iterations = iterations	

		# initializing organism
		self.best = Organism(0,0,None, self.w, self.h)
		self.best.initialize_genome(self.num_poly, num_vertex)
		self.best.genome_to_array()
		self.best.calculate_fitness_mse(self.goalpx)

		# define data header for hillclimber
		self.data.append(["Polygons", "Generation", "MSE"])

	def run(self):
		for i in range(0, self.iterations):
			james = Organism(0, 0, None, self.w, self.h)
			james.genome = self.best.deepish_copy_genome()
			james.id = i
			james.random_mutation(1)
			james.genome_to_array()

			if self.comparison_method == "MSE":
				james.calculate_fitness_mse(self.goalpx)
		
			if james.fitness <= self.best.fitness:
				self.best = deepcopy(james)
				# print(self.best.fitness)
				# best.save_img()

			if i in self.savepoints:
				self.best.save_img(self.outdirectory)
			
			self.save_data([self.num_poly, i, self.best.fitness])

		self.best.save_img(self.outdirectory)

class SA(Algorithm):
	# https://am207.github.io/2017/wiki/lab4.html
	def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory, iterations):
		super().__init__(goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory)
		self.iterations = iterations

		# initializing organism
		self.best = Organism(0,0,None, self.w, self.h)
		self.best.initialize_genome(self.num_poly, num_vertex)
		self.best.genome_to_array()
		self.best.calculate_fitness_mse(self.goalpx)

		self.current = deepcopy(self.best)

		# define data header for SA
		self.data.append(["Polygons", "Generation", "bestMSE", "currentMSE"])


	def acceptance_probability(self, dE, T):
		return math.exp(-dE/T)

	def cooling_geman(self, i):
		# what to pick for C?
		# d is usually set to one according to Nourani & Andresen (1998)
		c = 195075
		d = 1

		return c/math.log(i + d)

	def run(self):
		for i in range(1, self.iterations):
			james = Organism(0, i, None, self.w, self.h)
			james.genome = self.current.deepish_copy_genome()
			james.random_mutation(1)
			james.genome_to_array()

			james.calculate_fitness_mse(self.goalpx)


			dE = james.fitness - self.current.fitness
			T = self.cooling_geman(i)

			acceptance = self.acceptance_probability(dE, T)

			if random() < acceptance: 
				self.current.genome = james.deepish_copy_genome()
				self.current.fitness = james.fitness

			if self.current.fitness < self.best.fitness:
				self.best = deepcopy(self.current)

			if i in self.savepoints:
				self.best.generation = i
				self.best.save_img(self.outdirectory)

			self.save_data([self.num_poly, i, self.best.fitness, self.current.fitness])

		# self.best.save_img(self.outdirectory)


class PPA(Algorithm):
	def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory, iterations, pop_size, nmax, mmax):
		super().__init__(goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory)
		self.iterations = iterations
		self.pop_size = pop_size
		self.nmax = nmax
		self.mmax = mmax
		self.evaluations = 0
		self.pop = Population(self.pop_size)
		self.best = None
		self.worst = None

		# define data header for hillclimber
		self.data.append(["Polygons", "Generation", "Evaluations", "bestMSE", "worstMSE", "medianMSE", "meanMSE"])

		# fill population with random polygon drawings
		for i in range(self.pop_size):
			alex = Organism(0, i, None, self.w, self.h)
			alex.initialize_genome(self.num_poly, self.num_vertex)
			alex.genome_to_array()
			alex.calculate_fitness_mse(self.goalpx)
			self.pop.add_organism(alex)

	def calculate_random_runners(self):
		# if the populations max and min fitness are equal, this function generates random runners and distance for all organisms
		for organism in self.pop.organisms:
			organism.nr = random.randint(1, self.nmax)
			organism.d = random.randint(1, self.mmax)
		
	def calculate_runners(self):
		# default runner calculation for all organisms in the population
		for organism in self.pop.organisms:
			organism.scale_fitness(self.worst.fitness, self.best.fitness)
			organism.calculate_runners(self.nmax, self.mmax)

	def generation(self, gen):
		counter = 0
		for organism in self.pop.organisms[:]:
			for i in range(organism.nr):
				james = Organism(gen, counter, organism.name(), self.w, self.h)
				james.genome = organism.deepish_copy_genome()
				james.random_mutation(organism.d)
				james.genome_to_array()
				james.calculate_fitness_mse(self.goalpx)

				self.pop.add_organism(james)
				counter += 1

		self.evaluations += counter


	def run(self):
		gen = 1

		while self.evaluations < self.iterations:
			self.pop.sort_by_fitness()
			self.best = self.pop.return_best()
			self.worst = self.pop.return_worst()

			if gen in self.savepoints:
				self.best.save_img(self.outdirectory)
		
			if self.best.fitness != self.worst.fitness:
				self.calculate_runners()
				self.generation(gen)
			else: 
				self.calculate_random_runners()
				self.generation(gen)

			gen += 1


			self.pop.eliminate()
			best, worst, median, mean = self.pop.return_data()			
			self.save_data([self.num_poly, gen, self.evaluations, best.fitness, worst.fitness, median, mean])	


		self.pop.sort_by_fitness()
		self.best = self.pop.return_best()
		self.best.save_img(self.outdirectory)