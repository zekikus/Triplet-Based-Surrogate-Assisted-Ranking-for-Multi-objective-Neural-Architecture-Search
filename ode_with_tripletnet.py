import os
import copy
import torch
import random
import pickle
import numpy as np
from model import Model
from ops import OPS_Keys
from utils.distances import *
from triplet_network import *
from torchmetrics import Accuracy
from utils.bcnb_dataset import BCNB_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from xgboost import XGBClassifier
from torchinfo import summary

"""
    - Opposition-Based Differential Evolution
"""
class ODE():
    
    def __init__(self, pop_size = None, 
                 mutation_factor = None, 
                 crossover_prob = None, 
                 boundary_fix_type = 'random', 
                 seed = None,
                 mutation_strategy = 'rand1',
                 crossover_strategy = 'bin'):

        # DE related variables
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.mutation_strategy = mutation_strategy
        self.crossover_strategy = crossover_strategy
        self.boundary_fix_type = boundary_fix_type

        # Global trackers
        self.P_G = []
        self.P0 = [] # P0 population
        self.OP0 = [] # Opposite of the P0 population
        self.history = []
        self.allModels = dict()
        self.archive = [] # Potentially Pareto Optimal List
        self.best_arch = None
        self.S_train = [] # Training Set
        self.Triplets = [] # set of (anchor, pos, neg)
        self.triplet_model = None # TripletNetwork
        self.scaler = None # for TripletNetwork
        self.seed = seed
        self.predictor = XGBClassifier(n_estimators=100, max_depth=3, random_state=self.seed)

        # CONSTANTS
        self.MAX_SOL = 250
        self.START_NODE = 1
        self.END_NODE = 6
        self.MAX_PATH = 3
        self.INPUT_SIZE = (1, 3, 128, 128)
        self.NUM_VERTICES = self.END_NODE - self.START_NODE + 1
        self.GRAPH = {j : [i for i in range(int(j) + 1, self.END_NODE + 1)] for j in [k for k in range(self.START_NODE, self.END_NODE + 1)]}
        self.paths = self.find_all_paths(self.GRAPH, self.START_NODE , self.END_NODE)
        self.PATH_DICT = {idx: path for idx, path in enumerate(self.paths)}
        self.NUM_POSSIBLE_PATH = len(self.paths)
        self.DIMENSIONS = self.MAX_PATH + (self.END_NODE - self.START_NODE - 1) + 3 # 3 for path 5 for operation 1 for cell number 1 for stack number 1 for filter number
        self.MAX_STACK = 3
        self.MAX_NUM_CELL = 3
        self.JUMPING_RATE = 0.3
        self.STACKS = [i for i in range(1, self.MAX_STACK + 1)] # 1, 2, 3
        self.CELLS = [i for i in range(1, self.MAX_NUM_CELL + 1)] # 1, 2, 3
        self.NBR_FILTERS = [2**i for i in range(5, 8)] # 32, 64, 128
        self.OPS = copy.deepcopy(OPS_Keys)
    
    
    def reset(self):
        self.best_arch = None
        self.P_G = []
        self.P0 = []
        self.OP0 = []
        self.allModels = dict()
        self.history = []
        self.init_rnd_nbr_generators()
    
    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.init_pop_rnd = np.random.RandomState(self.seed)
        self.jumping_rnd = np.random.RandomState(self.seed)

    def get_model_FLOPs(self, model):
        results = summary(model, self.INPUT_SIZE, verbose=0)
        return results.to_megabytes(results.total_mult_adds)

    def seed_torch(self, seed=42):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
    
    def train_predictor(self):
        print("Training Predictor...")
        print(f"Training set size: {len(self.S_train)}")
        X = []
        y = []
        for idx1, X1 in enumerate(self.S_train):
            for idx2, X2 in enumerate(self.S_train):
                if idx1 == idx2:
                    continue
                
                X1_embedding = evaluate_embeddings(model = self.triplet_model, 
                                                   data = X1, 
                                                   scaler = self.scaler, 
                                                    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')) 
                X2_embedding = evaluate_embeddings(model = self.triplet_model, 
                                                   data = X2, 
                                                   scaler = self.scaler, 
                                                   device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')) 
                X.append(np.concatenate((X1_embedding, X2_embedding),  axis=1))
                if X1[2] < X2[2]: # if X1.f1 > X2.f1 then
                    y.append(1)
                else:
                    y.append(0)

        X = np.array(X).squeeze()
        y = np.array(y)
        self.predictor.fit(X[-1e6:], y[-1e6:])

    def writePickle(self, data, name, path=None):
        # Write History
        with open(f"results/{path}/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

    def circle_map(self, phi_z):         
        return np.mod(phi_z + 0.2 - (0.5 / (2 * np.pi)) * np.sin(2 * np.pi * phi_z), 1.0 )

    # Generate population P_0 with CODE
    def init_P0_population(self, pop_size = None):
        i = 0
        Z = 10
        D = self.DIMENSIONS
        
        for i in range(pop_size):
            x_i = np.zeros(D)
            for d in range(D):
                z = 0
                phi_z = random.random()
                while z <= Z:
                    phi_z = self.circle_map(phi_z) 
                    z += 1
                
                x_i[d] = phi_z * (1 - 0) + 0
            
            chromosome = x_i.copy()
            config = self.vector_to_config(chromosome)
            model = Model(chromosome, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES, path_dict=self.PATH_DICT, num_vertices=self.NUM_VERTICES, max_path=self.MAX_PATH)

            # Same Solution Check
            isSame, _ = self.checkSolution(model)
            if not isSame:
                model.solNo = self.solNo
                model.nbr_params = self.get_model_FLOPs(model)
                self.solNo += 1
                self.allModels[model.solNo] = {"org_matrix": model.org_matrix.astype("int8"), 
                                               "org_ops": model.org_ops,
                                               "chromosome": model.chromosome,
                                               "fitness": model.fitness,
                                               "cost": model.cost,
                                               "num_params": model.nbr_params}                                               
                self.P0.append(model)
                self.writePickle(model, model.solNo, result_path)
                i += 1
    
    def get_opposite_model(self, model, a = 0, b = 1):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            opposite_chromosome = np.array([a[idx] + b[idx] - c for idx, c in enumerate(model.chromosome)])
        else:
            opposite_chromosome = np.array([a + b - c for c in model.chromosome])
        
        config = self.vector_to_config(opposite_chromosome)
        opposite_model = Model(opposite_chromosome, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES, num_vertices=self.NUM_VERTICES, path_dict=self.paths, max_path=self.MAX_PATH)
        
        return opposite_model

    def init_OP0_population(self):
        counter = 0
        while counter < len(self.P0):
            opposite_model = self.get_opposite_model(self.P0[counter])
            opposite_model.nbr_params = self.get_model_FLOPs(opposite_model)
            # Same Solution Check
            isSame, _ = self.checkSolution(opposite_model)
            if not isSame:
                self.solNo += 1
                opposite_model.solNo = self.solNo
                self.allModels[opposite_model.solNo] = {"org_matrix": opposite_model.org_matrix.astype("int8"), 
                                                        "org_ops": opposite_model.org_ops,
                                                        "chromosome": opposite_model.chromosome,
                                                        "fitness": opposite_model.fitness,
                                                        "cost": opposite_model.cost,
                                                        "num_params": opposite_model.nbr_params}
                self.OP0.append(opposite_model)
                self.writePickle(opposite_model, opposite_model.solNo, result_path)
            counter += 1

    def is_weakly_dominate(self, model1, model2):
        """
        Check if vector 'a' is weakly dominate to vector 'b'
        for a multi-objective optimization problem.

        Args:
        a (tuple or list): first vector representing a solution in the objective space
        b (tuple or list): second vector representing a solution in the objective space

        Returns:
        bool: True if 'a' is weakly dominate to 'b', False otherwise
        """
        if model1.nbr_params is None:
            model1.nbr_params = self.get_model_FLOPs(model1)

        if model2.nbr_params is None:
            model2.nbr_params = self.get_model_FLOPs(model2)

        a = [model1.fitness, model1.nbr_params]
        b = [model2.fitness, model2.nbr_params]

        dominates = True
        atLeastOneBetter = False

        for i in range(len(a)):
            if(a[i] > b[i]):
                dominates = False
                break
            elif a[i] < b[i]:
                atLeastOneBetter = True

        return dominates and atLeastOneBetter
    
    def gde3_selection(self, sol1, sol2):
        
        """
        Case 1: In the case of infeasible vectors, the trial vector is selected if it weakly
        dominates the old vector in constraint violation space, otherwise the
        old vector is selected.
        """
        if sol1.isFeasible == False and sol2.isFeasible == False:
            if self.is_weakly_dominate(sol2, sol1):
                del sol1
                return [sol2]
            elif self.is_weakly_dominate(sol1, sol2):
                del sol2
                return [sol1]
            else:
                del sol1
                del sol2
        
        elif sol1.isFeasible == False or sol2.isFeasible == False:
            """
            Case 2: In the case of the feasible and infeasible vectors, the feasible vector is
            selected.
            """
            if sol1.isFeasible:
                del sol2
                return [sol1]
            else:
                del sol1
                return [sol2]
        
        elif sol1.isFeasible and sol2.isFeasible:
            
            if self.is_weakly_dominate(sol2, sol1):
                """
                If both vectors are feasible, then the trial is selected if it weakly
                dominates the old vector in the objective function space.
                """
                del sol1
                return [sol2]
            elif self.is_weakly_dominate(sol1, sol2):
                """
                If the old vector dominates the trial vector, then the old vector is
                selected.
                """
                del sol2
                return [sol1]
            else:
                """
                If neither vector dominates each other in the objective function space,
                then both vectors are selected for the next generation.
                """
                return [sol1, sol2]
            
    def checkSolution(self, model):
        model_dict = {"org_matrix": model.org_matrix.astype("int8"), 
                      "org_ops": model.org_ops}
        for i in self.allModels.keys():
            model_2 = self.allModels[i]
            D = jackard_distance_caz(model_dict, model_2)
            if D == 0:
                return True, model_2
        
        return False, None          
    
    def sample_population(self, size = None):
        '''Samples 'size' individuals'''

        selection = self.sample_pop_rnd.choice(np.arange(len(self.P_G)), size, replace=False)
        return self.P_G[selection]
    
    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        projection == The invalid value is truncated to the nearest limit
        random == The invalid value is repaired by computing a random number between its established limits
        reflection == The invalid value by computing the scaled difference of the exceeded bound multiplied by two minus

        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        
        if self.boundary_fix_type == 'projection':
            vector = np.clip(vector, 0.0, 1.0)
        elif self.boundary_fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        elif self.boundary_fix_type == 'reflection':
            vector[violations] = [0 - v if v < 0 else 2 - v if v > 1 else v for v in vector[violations]]

        return vector
    
    def find_all_paths(self, graph, start, end, path=[]):
        """
        Recursively find all paths in a directed acyclic graph (DAG) from start to end.

        :param graph: Dictionary representing the adjacency list of the graph
        :param start: Starting node
        :param end: Ending node
        :param path: Current path (used during recursion)
        :return: List of all paths from start to end
        """
        path = path + [start]  # Add the current node to the path

        # If the start node is the same as the end node, return the path
        if start == end:
            return [path]

        # If the start node is not in the graph, return an empty list
        if start not in graph:
            return []

        # List to store all paths
        paths = []

        # Recur for all neighbors of the current node
        for neighbor in graph[start]:
            if neighbor not in path:  # Avoid cycles (though DAGs shouldn't have any)
                new_paths = self.find_all_paths(graph, neighbor, end, path)
                for p in new_paths:
                    paths.append(p)

        return paths
    
    def get_param_value(self, value, step_size):
        ranges = np.arange(start=0, stop=1, step=1/step_size)
        return np.where((value < ranges) == False)[0][-1]

    def vector_to_config(self, vector):
        '''Converts numpy array to discrete values'''

        try:
            config = np.zeros(self.DIMENSIONS, dtype='uint8')

            # Paths:
            for idx in range(self.MAX_PATH):
                config[idx] = self.get_param_value(vector[idx], self.NUM_POSSIBLE_PATH) # Indexes of choosen paths

            # To reach paths follow the steps:
            # paths = self.find_all_paths(self.GRAPH, self.START_NODE , self.END_NODE)  
            # paths[config[0]], paths[config[1]], paths[config[2]] 

            # Vertices - Ops
            for idx in range(self.MAX_PATH, self.MAX_PATH + self.NUM_VERTICES - 2):
                config[idx] = self.get_param_value(vector[idx], len(self.OPS))

            # Number of Cells
            idx = self.MAX_PATH + self.NUM_VERTICES - 2
            config[idx] = self.get_param_value(vector[idx], len(self.CELLS))
            
            # Number of Stacks
            config[idx + 1] = self.get_param_value(vector[idx + 1], len(self.STACKS))

            # Number of Filters
            config[idx + 2] = self.get_param_value(vector[idx + 2], len(self.NBR_FILTERS))
        except:
            print("HATA...", vector)

        return config

    def f_objective(self, model):
        if model.isFeasible == False: # Feasibility Check
            return 1, 1e10
        
        # Else  
        fitness, cost, log = model.evaluate(train_loader, val_loader, loss_fn, metric_fn, device) 
        torch.cuda.empty_cache()
        model.fitness = fitness
        model.cost = cost
        model.nbr_params = self.get_model_FLOPs(model)
        model.fitness_type = "ACTUAL"

        if fitness != 1:
            self.S_train.append((model.solNo, model.chromosome, model.fitness, model.nbr_params))
            # self.S_train.append((model.chromosome, fitness))
            self.totalTrainedModel += 1
            self.allModels.setdefault(model.solNo, dict())
            self.allModels[model.solNo]["fitness"] = fitness
            self.allModels[model.solNo]["cost"] = cost
            self.allModels[model.solNo]["num_params"] = model.nbr_params
            if log is not None:
                with open(f"results/model_{model.solNo}.txt", "w") as f:
                    f.write(log)
            else:
                print(f"Log error on model {model.sol_no}")
        return fitness, cost

    def init_eval_pop(self):
        '''
            Creates new population of 'pop_size' and evaluates individuals.
        '''
        print("Start Initialization...")

        self.init_P0_population(self.pop_size)
        self.init_OP0_population()

        for model in self.P0:
            model.fitness, model.cost = self.f_objective(model)
            self.writePickle(model, model.solNo, result_path)
         
        for model in self.OP0:
            model.fitness, model.cost = self.f_objective(model)
            self.writePickle(model, model.solNo, result_path)

        

        # from line 22 to 24
        self.Triplets = construct_triplets_with_one_objectives(self.S_train) # 22 Triplets ← ConstructTriplets(Strain)
        input_dim = self.Triplets[0][0][1].shape[0]
        self.triplet_model, losses, self.scaler = train_triplet_network(
                                                triplets = self.Triplets,
                                                input_dim=input_dim,
                                                epochs=20,
                                                batch_size=1024,
                                                learning_rate=0.001,
                                                embedding_dim=64,
                                                margin=1.0,
                                                device=device
                                            ) # 23 NetworkTriplet ← Train Triplet Network with Triplets
       
        # Train the predictor
        print("Training Predictor...")
        self.train_predictor()  # 24 Predictor ← Train Predictor(NetworkTriplet, Strain)

        P_T = []
        for i in range(len(self.P0)):
            selected_models = self.gde3_selection(self.P0[i], self.OP0[i])
            for model in selected_models:
                P_T.append(model)

        self.P_G = sorted(P_T, key = lambda x: x.fitness, reverse=False)[:self.pop_size]
        self.best_arch = self.P_G[0]

        del self.P0
        del self.OP0
        
        return np.array(self.P_G)

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation(self, current=None, best=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand1(r1.chromosome, r2.chromosome, r3.chromosome)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5)
            mutant = self.mutation_rand2(r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome, r5.chromosome)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_rand1(best.chromosome, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4)
            mutant = self.mutation_rand2(best.chromosome, r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome)
        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_currenttobest1(current, best.chromosome, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_currenttobest1(r1.chromosome, best.chromosome, r2.chromosome, r3.chromosome)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.DIMENSIONS) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.DIMENSIONS)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''
            Performs the exponential crossover of DE
        '''
        n = self.crossover_rnd.randint(0, self.DIMENSIONS)
        L = 0
        while ((self.crossover_rnd.rand() < self.crossover_prob) and L < self.DIMENSIONS):
            idx = (n+L) % self.DIMENSIONS
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''
            Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring
    
    def readPickleFile(self, file):
        with open(f"results/model_{file}.pkl", "rb") as f:
            data = pickle.load(f)
        
        return data


    def update_archive(self, population):

        self.archive.extend(population)

        # Remove duplicated ones
        temp_idxs = []
        unique_sols = set()
        for idx, sol in enumerate(self.archive):
            if sol.solNo not in unique_sols:
                temp_idxs.append(idx)
            
            unique_sols.add(sol.solNo)

        self.archive = np.array(self.archive)[temp_idxs].tolist()
        # Remove duplicated ones

        # Reset Archive solution domination count
        for a in self.archive:
            a.domination_count = 0

        for i, _ in enumerate(self.archive):
            for j, _ in enumerate(self.archive):
                if i == j: continue
                if self.is_weakly_dominate(self.archive[i], self.archive[j]):
                    self.archive[j].domination_count += 1

        temp_archive = []
        for a in self.archive:
            if a.domination_count < 1:
                temp_archive.append(a)

        return temp_archive
    
    def generate_candidate(self, X_i, generation_best):
        X_i = X_i.chromosome
        X_i = copy.deepcopy(X_i)
        V_i = self.mutation(current = X_i, best = generation_best)
        U_i = self.crossover(X_i, V_i)
        U_i = self.boundary_check(U_i)
        config = self.vector_to_config(U_i)
        
        model = Model(U_i, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES, num_vertices=self.NUM_VERTICES, path_dict=self.paths, max_path=self.MAX_PATH)
        self.solNo += 1
        model.solNo = self.solNo

        return model
    
    def generate_candidate_with_predictor(self, X_i, generation_best):
        Xi_chromosome = X_i.chromosome
        Xi_chromosome = copy.deepcopy(Xi_chromosome)
        Xi_data = (X_i.solNo, X_i.chromosome)
        Xi_embedding = evaluate_embeddings(model = self.triplet_model, 
                                                   data = Xi_data, 
                                                   scaler = self.scaler, 
                                                   device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')) 
        mutation_strategies = ['rand1', 'rand2', 'best1', 'best2', 'currenttobest1', 'randtobest1']
        inital_mutation_strategy = self.mutation_strategy
        better_mutations = [] # mutation models better than X_i
        for mutation_strategy in mutation_strategies:
            self.mutation_strategy = mutation_strategy
            C = self.mutation(current = Xi_chromosome, best = generation_best)
            C = self.boundary_check(C)
            config = self.vector_to_config(C)
            C_model = Model(C, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES, num_vertices=self.NUM_VERTICES, path_dict=self.paths, max_path=self.MAX_PATH)
            self.solNo += 1
            C_model.solNo = self.solNo
            C_data = (C_model.solNo, C_model.chromosome)
            C_embedding = evaluate_embeddings(model = self.triplet_model, 
                                                   data = C_data, 
                                                   scaler = self.scaler, 
                                                   device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')) 

            concat_embedding = np.concatenate((C_embedding, Xi_embedding),axis=1)
            C_model.fitness = 1 - self.predictor.predict(concat_embedding)[0]
            C_model.fitness_type = "PREDICTED"
            C_model.nbr_params = self.get_model_FLOPs(C_model)
            if C_model.fitness == 0:
                better_mutations.append(C_model)

        self.mutation_strategy = inital_mutation_strategy 
        if better_mutations:
            best_candidate = min(better_mutations, key=lambda m: m.nbr_params)
            return best_candidate
        else:
            return X_i

    def evolve_generation(self):
        '''
            Performs a complete DE evolution: mutation -> crossover -> selection
        '''
        G = 1
        N = 5

        while self.totalTrainedModel <= self.MAX_SOL:
            P_T = []
            generationBest = min(self.P_G, key=lambda x: x.fitness)
            if G < N:
                trials = []
                for j in range(self.pop_size):
                    U_iG = self.generate_candidate(self.P_G[j], generationBest) # line 37, 38, 39
                    trials.append(U_iG)
                
                trials = np.array(trials)

                # selection
                for j in range(self.pop_size):
                    target = self.P_G[j]
                    mutant = trials[j]

                    isSameSolution, sol = self.checkSolution(mutant)
                    if isSameSolution:
                        print("SAME SOLUTION")
                        cfg = self.vector_to_config(sol["chromosome"])
                        mutant = Model(sol["chromosome"], cfg, self.CELLS[cfg[-3]], self.STACKS[cfg[-2]], self.NBR_FILTERS[cfg[-1]], NUM_CLASSES, num_vertices= self.NUM_VERTICES, path_dict=self.paths, max_path=self.MAX_PATH)
                        mutant.fitness = sol["fitness"]
                    else:
                        # 41 Ui,G.fitnesstype ← ”ACTUAL”
                        # 42 Ui,G.f2 ← f2(Ui,G) (cost)
                        # 43 Strain ← Strain ∪ Ui,G (in f_objective function)
                        self.f_objective(mutant) # line 41, 42, 43 
                        self.writePickle(mutant, mutant.solNo, result_path)
                        self.allModels[mutant.solNo] = {"org_matrix": mutant.org_matrix.astype("int8"), 
                                                        "org_ops": mutant.org_ops,
                                                        "chromosome": mutant.chromosome,
                                                        "fitness": mutant.fitness,
                                                        "cost": mutant.cost,
                                                        "num_params": mutant.nbr_params}

                    # Check Termination Condition
                    if self.totalTrainedModel > self.MAX_SOL: 
                        return

                    selected_models = self.gde3_selection(target, mutant)
                    for model in selected_models:
                        if model.fitness <= self.best_arch.fitness:
                            self.best_arch = model
                        P_T.append(model) # 44 PT ← PT ∪ GDE3(Xi,G−1, Ui,G)
                    
            else:
                j = 0
                while len(P_T) < self.pop_size:
                    U_iG = self.generate_candidate_with_predictor(self.P_G[j], generationBest)
                    if U_iG is not None:
                        isSameSolution, sol = self.checkSolution(U_iG)
                        if isSameSolution:
                            print("SAME SOLUTION 2")
                            cfg = self.vector_to_config(sol["chromosome"])
                            U_iG = Model(sol["chromosome"], cfg,
                                        self.CELLS[cfg[-3]], self.STACKS[cfg[-2]], self.NBR_FILTERS[cfg[-1]],
                                        NUM_CLASSES, num_vertices=self.NUM_VERTICES, path_dict=self.paths, max_path=self.MAX_PATH)
                            U_iG.fitness = sol["fitness"]
                            U_iG.fitness_type = "ACTUAL"  
                        selected_models = self.gde3_selection(self.P_G[j], U_iG)
                    else:
                        selected_models = [self.P_G[j]]

                    for sol in selected_models:
                        P_T.append(sol)

                    j += 1
            
                for X_i in P_T:
                    if X_i.fitness_type == "PREDICTED":
                        self.f_objective(X_i)
                        self.writePickle(X_i, X_i.solNo, result_path)
                        #self.S_train.append((X_i.chromosome, X_i.fitness))
                        self.allModels[X_i.solNo] = {"org_matrix": X_i.org_matrix.astype("int8"), 
                                                        "org_ops": X_i.org_ops,
                                                        "chromosome": X_i.chromosome,
                                                        "fitness": X_i.fitness,
                                                        "cost": X_i.cost,
                                                        "num_params": X_i.nbr_params}

            
            self.P_G = P_T
            self.P_G = np.array(self.P_G)
            G += 1

            # Update Archive
            self.archive = self.update_archive(self.P_G)

            if G >= N:
                # Construct triplets with the new training set 
                self.Triplets = construct_triplets_with_one_objectives(self.S_train) # 61 Triplets ← ConstructTriplets(Strain)
                input_dim = self.Triplets[0][0][1].shape[0]
                self.triplet_model, losses, self.scaler = train_triplet_network(
                                                        triplets = self.Triplets,
                                                        input_dim=input_dim,
                                                        epochs=20,
                                                        batch_size=2**15,
                                                        learning_rate=0.001,
                                                        embedding_dim=64,
                                                        margin=1.0
                                                    ) # 62 NetworkTriplet ← Train Triplet Network with Triplets
            
                # Train the predictor
                print("Training Predictor...")
                self.train_predictor()  # 63 Predictor ← Train Predictor(NetworkTriplet, Strain)


            print(f"Generation:{G}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")


    def run(self, seed):
        self.seed = seed
        self.solNo = 0
        self.generation = 0
        self.totalTrainedModel = 0
        print(self.mutation_strategy)
        
        self.reset()
        self.seed_torch()

        self.P_G = self.init_eval_pop()

        # Update Archive
        self.archive = self.update_archive(self.P_G) # 31 A ← updateFront(PG)

        self.evolve_generation()

        # Write Pareto Front
        for model in self.archive:
            self.writePickle(model, model.solNo, "BCNB_MODE_Triplet/pareto_front") 
        

if __name__ == "__main__":
    device = torch.device('cuda:1')

    if os.path.exists("results/BCNB_MODE_Triplet/") == False:
        os.makedirs("results/BCNB_MODE_Triplet/")
    if os.path.exists("results/BCNB_MODE_Triplet/pareto_front/") == False:
        os.makedirs("results/BCNB_MODE_Triplet/pareto_front/")

    result_path = f"BCNB_MODE_Triplet" 
    random.seed(42)
    download = True
    NUM_CLASSES = 2
    BATCH_SIZE = 64

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load the train and val data for NAS
    train_dataset = BCNB_Dataset('data/train_patches.txt', percentage=0.1, nas_stage=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = BCNB_Dataset('data/validation_patches.txt', percentage=0.1, nas_stage=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

    de = ODE(pop_size=20, mutation_factor=0.5, crossover_prob=0.5, seed=42)
    de.run(42)