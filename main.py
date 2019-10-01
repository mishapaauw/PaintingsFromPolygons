from random import randint, choice, shuffle
from PIL import Image
import numpy as np
from algorithms import Algorithm, Hillclimber, SA, PPA
import time
import math
import os
import csv
from multiprocessing import Process, current_process


# im_goal = Image.open("paintings/bach-240-180.png")
# im_goal = Image.open("paintings/dali-240-180.png")
im_goal = Image.open("paintings/monalisa-240-180.png")
# im_goal = Image.open("paintings/pollock-240-180.png")
# im_goal = Image.open("paintings/mondriaan2-180-240.png")


goal = np.array(im_goal)
h, w = np.shape(goal)[0], np.shape(goal)[1]
method = "MSE"
# outdirx = "test/"

# genome size settings
polygons = 250
vertices = 1000


# ppa specific settings
population_size = 30
nmax = 5 # max number of runners for the best indidiviual within a population


def experiment(name, algorithm, paintings, repetitions, polys, iterations, savepoints):
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

				if algorithm == "PPA":
					nparam = (poly * 4 * 2) + (poly * 4) + poly
					mmax = math.ceil(nparam * 0.20)

					solver = PPA(goal, w, h, poly, poly*4, "MSE", savepoints, outdir, iterations, population_size, nmax, mmax)

				elif algorithm == "HC":
					solver = Hillclimber(goal, w, h, poly, poly * 4, "MSE", savepoints, outdir, iterations)

				elif algorithm =="SA":
					solver = SA(goal, w, h, poly, poly * 4, "MSE", savepoints, outdir, iterations)

				# run the solver with selected algorithm
				solver.run()
				solver.write_data()
				bestMSE = solver.best.fitness

				# save best value in maindata sheet
				datarow = [painting_name, str(poly * 4), str(repetition), bestMSE]

				with open(datafile, 'a', newline = '') as f:
					writer = csv.writer(f)
					writer.writerow(datarow)

				toc = time.time()
				now = time.strftime("%c")
				with open(logfile, 'a') as f:
					f.write(now + " finished run " + str(exp) + "/" + str(total_runs) + " n: " + str(repetition) + " poly: " + str(poly) + " painting: " + painting_name + " in " + str((toc - tic)/60) + " minutes\n")

				exp += 1



name = "1miltest.x2"
paintings_files = [["paintings/monalisa-240-180.png"], ["paintings/bach-240-180.png"], ["paintings/dali-240-180.png"], ["paintings/mondriaan2-180-240.png"], ["paintings/pollock-240-180.png"], ["paintings/starrynight-240-180.png"], ["paintings/kiss-180-240.png"]]
# paintin = ["paintings/kiss-180-240.png"]
savepoints = list(range(0, 21000, 50))
repetitions = 5
# polys = [25]
polys = [5, 25, 75, 125, 175, 250]
iterations = 1000000
# define a list of savepoints, more in the first part of the run, and less later.
savepoints = list(range(0, 100000, 2500)) + list(range(100000, 1000000, 10000))

population_size = 30
nmax = 5


# args = (name, paintings_files, repetitions, polys, iterations, savepoints)

names = ["mona", "bach","dali", "mondriaanSA", "pollock", "starrynight", "kiss"]
# names = ["kiss1", "kiss2","kiss3","kiss4","kiss5"]
#experiment(name, "HC" paintings_files, repetitions, polys, iterations, savepoints)

# parallelize stuff

if __name__ == '__main__':
	worker_count = 7
	worker_pool = []
	for i in range(worker_count):
		# print(str(paintings_files[i]), paintin)
		args = (names[i], "SA", paintings_files[i], repetitions, polys, iterations, savepoints)
		p = Process(target=experiment, args=args)
		p.start()
