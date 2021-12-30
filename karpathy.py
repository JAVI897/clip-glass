import argparse
import os
import torch
import numpy as np
import pickle
from pymoo.optimize import minimize
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_algorithm, get_decision_making, get_decomposition
from pymoo.visualization.scatter import Scatter
from pre_trained_transformers_tf import get_captions
from config import get_config
from problem import GenerationProblem
from operators import get_operators
import time
from gpt2.encoder import get_encoder
import os
import random
from itertools import cycle
import numpy as np
import json
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--config", type=str, default="DeepMindBigGAN512")
parser.add_argument("--generations", type=int, default=500)
parser.add_argument("--save-each", type=int, default=50)
parser.add_argument("--tmp-folder", type=str, default="./tmp")

def save_callback(algorithm):
    global iteration
    global config

    iteration += 1
    if iteration % config.save_each == 0 or iteration == config.generations:
        if config.problem_args["n_obj"] == 1:
            sortedpop = sorted(algorithm.pop, key=lambda p: p.F)
            X = np.stack([p.X for p in sortedpop])  
        else:
            X = algorithm.pop.get("X")
        
        ls = config.latent(config)
        ls.set_from_population(X)

        with torch.no_grad():
            generated = algorithm.problem.generator.generate(ls, minibatch=config.batch_size)
            if config.task == "txt2img":
                ext = "jpg"
            elif config.task == "img2txt":
                ext = "txt"
            name = "genetic-it-%d.%s" % (iteration, ext) if iteration < config.generations else "genetic-it-final.%s" % (ext, )
            algorithm.problem.generator.save(generated, os.path.join(config.tmp_folder, name))


def main():

	output_predictions = 'karpathy_test_predictions_generations_{}.csv'.format(parser.parse_args().generations)
	#Get ground truth captions validation
	with open('data/coco/karpathy_validation_captions.json') as json_file:
		captions_valid_test = json.load(json_file)

	if not os.path.isfile(output_predictions):
		captions_final = []
		predictions = []
		file = open('data/coco/karpathy_valid_images.txt','r')
		c = 0
		time_ini_pred = time.time()
		for test_img in file.readlines():
			global iteration
			global config
			iteration = 0
			config = parser.parse_args()
			vars(config).update(get_config(config.config))
			file_path, number_instance = test_img.split()
			_, name_img = file_path.split('/')
			name_img = 'data/coco/val2014/'+ name_img # image path
			config.target = name_img
			caption_img = captions_valid_test[number_instance][:5]

			problem = GenerationProblem(config)
			operators = get_operators(config)

			if not os.path.exists(config.tmp_folder): os.mkdir(config.tmp_folder)

			# generate initial population
			enc = get_encoder(config)
			ini_t = time.time()
			captions = get_captions(config.target)
			end_t = time.time()
			print('[INFO] Generated captions. Time: {}'.format(end_t - ini_t))

			captions_tokenized = [ enc.encode(caption) for caption in captions ]

			vecs = []
			for vec in captions_tokenized:
			    new_vec = vec
			    while len(new_vec) < config.dim_z:
			        new_vec.append(random.randint(0, config.encoder_size))
			    vecs.append(new_vec)

			while len(vecs) < config.batch_size:
			    new_vec = []
			    while len(new_vec) < config.dim_z:
			        new_vec.append(random.randint(0, config.encoder_size))
			    vecs.append(new_vec)

			initial_solutions = np.array(vecs)

			algorithm = get_algorithm(
			    config.algorithm,
			    pop_size=config.pop_size,
			    sampling= initial_solutions,
			    crossover=operators["crossover"],
			    mutation=operators["mutation"],
			    eliminate_duplicates=True,
			    callback=save_callback,
			    **(config.algorithm_args[config.algorithm] if "algorithm_args" in config and config.algorithm in config.algorithm_args else dict())
			)

			res = minimize(
			    problem,
			    algorithm,
			    ("n_gen", config.generations),
			    save_history=False,
			    verbose=True,
			    seed = 1
			)

			pickle.dump(dict(
			    X = res.X,
			    F = res.F,
			    G = res.G,
			    CV = res.CV,
			), open(os.path.join(config.tmp_folder, "genetic_result"), "wb"))

			if config.problem_args["n_obj"] == 2:
			    plot = Scatter(labels=["similarity", "discriminator",])
			    plot.add(res.F, color="red")
			    plot.save(os.path.join(config.tmp_folder, "F.jpg"))


			if config.problem_args["n_obj"] == 1:
			    sortedpop = sorted(res.pop, key=lambda p: p.F)
			    X = np.stack([p.X for p in sortedpop])
			else:
			    X = res.pop.get("X")

			ls = config.latent(config)
			ls.set_from_population(X)

			torch.save(ls.state_dict(), os.path.join(config.tmp_folder, "ls_result"))

			if config.problem_args["n_obj"] == 1:
			    X = np.atleast_2d(res.X)
			else:
			    try:
			        result = get_decision_making("pseudo-weights", [0, 1]).do(res.F)
			    except:
			        print("Warning: cant use pseudo-weights")
			        result = get_decomposition("asf").do(res.F, [0, 1]).argmin()

			    X = res.X[result]
			    X = np.atleast_2d(X)

			ls.set_from_population(X)

			with torch.no_grad():
			    generated = problem.generator.generate(ls)
			print(generated)
			
			caption_img.append(generated[0])
			captions_final.append(caption_img)
			#print(captions)
			# debugging
			c+=1
			if c % 9 == 0:
				print('[INFO] Evaluated {} out of {}'.format(c, 5000))
				break

		time_end_pred = time.time()
		total_time = time_end_pred - time_ini_pred
		print('[INFO] Total time: {}'.format(total_time))
		df = pd.DataFrame(captions_final, columns = ['caption 1', 'caption 2', 'caption 3', 'caption 4', 'caption 5', 'prediction'])
		print('\nWriting predictions to file "{}".'.format(output_predictions))
		df.to_csv(output_predictions)


if __name__ == '__main__':
	main()