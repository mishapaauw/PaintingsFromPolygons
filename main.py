from random import randint, choice, shuffle
from PIL import Image
import numpy as np
from algorithms import Algorithm, Hillclimber, SA, PPA
import time
import math
import os
import csv

# from globals import *

# goal

# im_goal = Image.open("paintings/bach-240-180.png")
# im_goal = Image.open("paintings/dali-240-180.png")
im_goal = Image.open("paintings/monalisa-240-180.png")
# im_goal = Image.open("paintings/pollock-240-180.png")
# im_goal = Image.open("paintings/mondriaan2-180-240.png")


goal = np.array(im_goal)
h, w = np.shape(goal)[0], np.shape(goal)[1]
method = "MSE"
outdirx = "test/"

# genome size settings
polygons = 250
vertices = 1000


# ppa specific settings
population_size = 30
nmax = 5 # max number of runners for the best indidiviual within a population



# SA = SA(goal, w, h, polygons, vertices, method, savepoints, outdirx, 100000)
# SA.run()
# SA.write_data()

def experiment(name, paintings, repetitions, polys, iterations, savepoints):
	# get date/time
	now = time.strftime("%c")

	# create experiment directory with log .txt file
	if not os.path.exists(name):
		os.makedirs(name)

	total_runs = len(polys) * len(paintings) * repetitions

	# logging a lot of metadata
	logfile = name+"/"+name+"-LOG.txt"
	with open(logfile, 'a') as f:
		f.write("EXPERIMENT " + name + " LOG\n")
		f.write("DATE " + now + "\n\n")
		f.write("STOP CONDITION " +str(iterations)+ " iterations\n\n")
		f.write("LIST OF PAINTINGS (" + str(len(paintings)) +")\n")
		for painting in paintings:
			f.write(painting + "\n")
		f.write("\n")
		f.write("POLYS " + str(len(polys)) + " " + str(polys) + "\n\n")
		f.write("REPETITIONS " +str(repetitions) + "\n\n")
		f.write("RESULTING IN A TOTAL OF " + str(total_runs) + " RUNS\n\n")
		f.write("STARTING EXPERIMENT NOW!\n")

	# initializing the main datafile
	datafile = name+"/"+name + "-DATA.csv"
	header = ["Painting", "Vertices", " Replication", "MSE"]
	with open(datafile, 'a', newline = '') as f:
		writer = csv.writer(f)
		writer.writerow(header)


	# main experiment, looping through repetitions, poly numbers, and paintings:
	exp = 1
	for painting in paintings:
		painting_name = painting.split("/")[1].split("-")[0]
		for poly in polys:
			for repetition in range(repetitions):
				tic = time.time()
				# make a directory for this run, containing the per iteration data and a selection of images
				outdir = name + "/" + str(exp) + "-" + str(repetition) + "-" + str(poly) + "-" + painting_name
				os.makedirs(outdir)
				
				# run the hillclimber
				im_goal = Image.open(painting)
				goal = np.array(im_goal)
				h, w = np.shape(goal)[0], np.shape(goal)[1]


				# the number of mutable parameters depends on the number of vertices and polygons
				nparam = (poly * 4 * 2) + (poly * 4) + poly

				# longest runners mutate 20% of their parameters
				mmax = math.ceil(nparam * 0.10)


				ppa = PPA(goal, w, h, poly, poly*4, "MSE", savepoints, outdir, iterations, population_size, nmax, mmax)
				ppa.run()
				ppa.write_data()
				# sa = SA(goal, w, h, poly, poly * 4, "MSE", savepoints, outdir, iterations)
				# sa.run()
				
				# sa.write_data()

				# save best value in maindata sheet
				bestMSE = ppa.best.fitness
				datarow = [painting_name, str(poly * 4), str(repetition), bestMSE]

				with open(datafile, 'a', newline = '') as f:
					writer = csv.writer(f)
					writer.writerow(datarow)

				toc = time.time()
				now = time.strftime("%c")
				with open(logfile, 'a') as f:
					f.write(now + " finished run " + str(exp) + "/" + str(total_runs) + " n: " + str(repetition) + " poly: " + str(poly) + " painting: " + painting_name + " in " + str((toc - tic)/60) + " minutes\n")

				exp += 1



name = "PPA"
paintins = ["paintings/monalisa-240-180.png", "paintings/bach-240-180.png", "paintings/dali-240-180.png", "paintings/mondriaan2-180-240.png", "paintings/pollock-240-180.png", "paintings/starrynight-240-180.png"]
# paintins = ["paintings/mondriaan2-180-240.png"]
savepoints = list(range(0, 15000, 250)) + list(range(15000, 100000, 5000))
repetitions = 5
# polys = [25]
polys = [5, 25, 75, 125, 175, 250]
iterations = 100000
# define a list of savepoints, more in the first part of the run, and less later.
savepoints = list(range(0, 2500, 50)) + list(range(2500, 10000, 500))

population_size = 30
nmax = 5


experiment(name, paintins, repetitions, polys, iterations, savepoints)





