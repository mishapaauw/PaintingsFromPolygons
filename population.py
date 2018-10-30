import numpy as np

class Population:
	def __init__(self, size):
		self.size = size
		self.organisms = []

	def add_organism(self, organism):
		self.organisms.append(organism)

	def sort_by_fitness(self):
		self.organisms.sort(key=lambda organism: organism.fitness)

	def eliminate(self):
		self.sort_by_fitness()
		self.organisms = self.organisms[:self.size]

	def return_best(self):
		return self.organisms[0]

	def return_worst(self):
		return self.organisms[self.size - 1]

	def return_data(self):
		self.sort_by_fitness()

		best = self.return_best()
		worst = self.return_worst()

		fitnesses = [organism.fitness for organism in self.organisms]
		median = fitnesses[int(self.size/2)]
		mean = np.mean(fitnesses)

		return best, worst, median, mean


	def dump_imgs(self):
		for organism in self.organisms:
			organism.save_img()
		
	def __repr__(self):
		return repr((self.organisms))