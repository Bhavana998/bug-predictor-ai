"""
Genetic Algorithm Optimizer for Hyperparameter Tuning
"""

import numpy as np
from deap import base, creator, tools, algorithms
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

from config import GA_PARAMS

class GeneticOptimizer:
    """Enhanced Genetic Algorithm for hyperparameter optimization"""
    
    def __init__(self, X_train, y_train, cv_folds=5):
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []
        
        # Expanded parameter spaces
        self.param_spaces = {
            'rf': {
                'n_estimators': (100, 2000),
                'max_depth': (10, 200),
                'min_samples_split': (2, 50),
                'min_samples_leaf': (1, 20),
                'max_features': ['sqrt', 'log2', None],
                'criterion': ['gini', 'entropy']
            },
            'svm': {
                'C': (0.1, 1000),
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'degree': (2, 5)  # for poly kernel
            },
            'lr': {
                'C': (0.01, 100),
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'l1_ratio': (0, 1)  # for elasticnet
            },
            'xgb': {
                'n_estimators': (100, 2000),
                'max_depth': (3, 50),
                'learning_rate': (0.001, 0.3),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'colsample_bylevel': (0.5, 1.0),
                'min_child_weight': (1, 20),
                'gamma': (0, 5),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10)
            },
            'lgb': {
                'n_estimators': (100, 2000),
                'max_depth': (5, 100),
                'num_leaves': (10, 300),
                'learning_rate': (0.001, 0.3),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'min_child_samples': (5, 50),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
                'min_split_gain': (0, 1)
            },
            'cat': {
                'iterations': (100, 2000),
                'depth': (4, 16),
                'learning_rate': (0.001, 0.3),
                'l2_leaf_reg': (1, 10),
                'border_count': (32, 255),
                'bagging_temperature': (0, 1)
            }
        }
        
        # Setup DEAP
        self.setup_genetic_algorithm()
    
    def setup_genetic_algorithm(self):
        """Setup DEAP framework"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                            self.toolbox.attr_float, self.get_individual_length())
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=GA_PARAMS['tournament_size'])
        self.toolbox.register("evaluate", self.evaluate_individual)
    
    def get_individual_length(self):
        """Get length of individual based on parameter spaces"""
        length = 0
        for model, params in self.param_spaces.items():
            length += len(params)
        return length
    
    def decode_individual(self, individual):
        """Enhanced decoding with adaptive ranges"""
        params = {}
        idx = 0
        
        # Decode Random Forest
        params['rf'] = {
            'n_estimators': int(self.param_spaces['rf']['n_estimators'][0] + 
                               individual[idx] * (self.param_spaces['rf']['n_estimators'][1] - 
                                                 self.param_spaces['rf']['n_estimators'][0])),
            'max_depth': int(self.param_spaces['rf']['max_depth'][0] + 
                            individual[idx+1] * (self.param_spaces['rf']['max_depth'][1] - 
                                                self.param_spaces['rf']['max_depth'][0])),
            'min_samples_split': int(self.param_spaces['rf']['min_samples_split'][0] + 
                                    individual[idx+2] * (self.param_spaces['rf']['min_samples_split'][1] - 
                                                        self.param_spaces['rf']['min_samples_split'][0])),
            'min_samples_leaf': int(self.param_spaces['rf']['min_samples_leaf'][0] + 
                                   individual[idx+3] * (self.param_spaces['rf']['min_samples_leaf'][1] - 
                                                       self.param_spaces['rf']['min_samples_leaf'][0])),
            'max_features': self.param_spaces['rf']['max_features'][int(individual[idx+4] * 2)],
            'criterion': self.param_spaces['rf']['criterion'][int(individual[idx+5])]
        }
        idx += 6
        
        # Decode SVM
        svm_gamma_options = self.param_spaces['svm']['gamma']
        svm_kernel_options = self.param_spaces['svm']['kernel']
        
        params['svm'] = {
            'C': 10 ** (individual[idx] * 4 - 1),  # Range: 0.1 to 1000
            'gamma': svm_gamma_options[int(individual[idx+1] * (len(svm_gamma_options)-1))],
            'kernel': svm_kernel_options[int(individual[idx+2] * (len(svm_kernel_options)-1))],
            'degree': int(self.param_spaces['svm']['degree'][0] + 
                         individual[idx+3] * (self.param_spaces['svm']['degree'][1] - 
                                             self.param_spaces['svm']['degree'][0]))
        }
        idx += 4
        
        # Decode Logistic Regression
        lr_solver_options = self.param_spaces['lr']['solver']
        lr_penalty_options = self.param_spaces['lr']['penalty']
        
        params['lr'] = {
            'C': 10 ** (individual[idx] * 4 - 2),  # Range: 0.01 to 100
            'solver': lr_solver_options[int(individual[idx+1] * (len(lr_solver_options)-1))],
            'penalty': lr_penalty_options[int(individual[idx+2] * (len(lr_penalty_options)-1))] 
                      if individual[idx+2] < 1 else None,
            'l1_ratio': individual[idx+3] if 'elasticnet' in params['lr'].get('penalty', '') else None
        }
        idx += 4
        
        # Decode XGBoost
        params['xgb'] = {
            'n_estimators': int(self.param_spaces['xgb']['n_estimators'][0] + 
                               individual[idx] * (self.param_spaces['xgb']['n_estimators'][1] - 
                                                 self.param_spaces['xgb']['n_estimators'][0])),
            'max_depth': int(self.param_spaces['xgb']['max_depth'][0] + 
                            individual[idx+1] * (self.param_spaces['xgb']['max_depth'][1] - 
                                                self.param_spaces['xgb']['max_depth'][0])),
            'learning_rate': self.param_spaces['xgb']['learning_rate'][0] + 
                            individual[idx+2] * (self.param_spaces['xgb']['learning_rate'][1] - 
                                                self.param_spaces['xgb']['learning_rate'][0]),
            'subsample': self.param_spaces['xgb']['subsample'][0] + 
                        individual[idx+3] * (self.param_spaces['xgb']['subsample'][1] - 
                                            self.param_spaces['xgb']['subsample'][0]),
            'colsample_bytree': self.param_spaces['xgb']['colsample_bytree'][0] + 
                               individual[idx+4] * (self.param_spaces['xgb']['colsample_bytree'][1] - 
                                                   self.param_spaces['xgb']['colsample_bytree'][0]),
            'colsample_bylevel': self.param_spaces['xgb']['colsample_bylevel'][0] + 
                                individual[idx+5] * (self.param_spaces['xgb']['colsample_bylevel'][1] - 
                                                    self.param_spaces['xgb']['colsample_bylevel'][0]),
            'min_child_weight': int(self.param_spaces['xgb']['min_child_weight'][0] + 
                                   individual[idx+6] * (self.param_spaces['xgb']['min_child_weight'][1] - 
                                                       self.param_spaces['xgb']['min_child_weight'][0])),
            'gamma': self.param_spaces['xgb']['gamma'][0] + 
                    individual[idx+7] * (self.param_spaces['xgb']['gamma'][1] - 
                                        self.param_spaces['xgb']['gamma'][0]),
            'reg_alpha': self.param_spaces['xgb']['reg_alpha'][0] + 
                        individual[idx+8] * (self.param_spaces['xgb']['reg_alpha'][1] - 
                                            self.param_spaces['xgb']['reg_alpha'][0]),
            'reg_lambda': self.param_spaces['xgb']['reg_lambda'][0] + 
                         individual[idx+9] * (self.param_spaces['xgb']['reg_lambda'][1] - 
                                             self.param_spaces['xgb']['reg_lambda'][0])
        }
        idx += 10
        
        # Decode LightGBM
        params['lgb'] = {
            'n_estimators': int(self.param_spaces['lgb']['n_estimators'][0] + 
                               individual[idx] * (self.param_spaces['lgb']['n_estimators'][1] - 
                                                 self.param_spaces['lgb']['n_estimators'][0])),
            'max_depth': int(self.param_spaces['lgb']['max_depth'][0] + 
                            individual[idx+1] * (self.param_spaces['lgb']['max_depth'][1] - 
                                                self.param_spaces['lgb']['max_depth'][0])),
            'num_leaves': int(self.param_spaces['lgb']['num_leaves'][0] + 
                             individual[idx+2] * (self.param_spaces['lgb']['num_leaves'][1] - 
                                                 self.param_spaces['lgb']['num_leaves'][0])),
            'learning_rate': self.param_spaces['lgb']['learning_rate'][0] + 
                            individual[idx+3] * (self.param_spaces['lgb']['learning_rate'][1] - 
                                                self.param_spaces['lgb']['learning_rate'][0]),
            'subsample': self.param_spaces['lgb']['subsample'][0] + 
                        individual[idx+4] * (self.param_spaces['lgb']['subsample'][1] - 
                                            self.param_spaces['lgb']['subsample'][0]),
            'colsample_bytree': self.param_spaces['lgb']['colsample_bytree'][0] + 
                               individual[idx+5] * (self.param_spaces['lgb']['colsample_bytree'][1] - 
                                                   self.param_spaces['lgb']['colsample_bytree'][0]),
            'min_child_samples': int(self.param_spaces['lgb']['min_child_samples'][0] + 
                                    individual[idx+6] * (self.param_spaces['lgb']['min_child_samples'][1] - 
                                                        self.param_spaces['lgb']['min_child_samples'][0])),
            'reg_alpha': self.param_spaces['lgb']['reg_alpha'][0] + 
                        individual[idx+7] * (self.param_spaces['lgb']['reg_alpha'][1] - 
                                            self.param_spaces['lgb']['reg_alpha'][0]),
            'reg_lambda': self.param_spaces['lgb']['reg_lambda'][0] + 
                         individual[idx+8] * (self.param_spaces['lgb']['reg_lambda'][1] - 
                                             self.param_spaces['lgb']['reg_lambda'][0]),
            'min_split_gain': self.param_spaces['lgb']['min_split_gain'][0] + 
                             individual[idx+9] * (self.param_spaces['lgb']['min_split_gain'][1] - 
                                                 self.param_spaces['lgb']['min_split_gain'][0])
        }
        idx += 10
        
        # Decode CatBoost
        params['cat'] = {
            'iterations': int(self.param_spaces['cat']['iterations'][0] + 
                             individual[idx] * (self.param_spaces['cat']['iterations'][1] - 
                                               self.param_spaces['cat']['iterations'][0])),
            'depth': int(self.param_spaces['cat']['depth'][0] + 
                        individual[idx+1] * (self.param_spaces['cat']['depth'][1] - 
                                            self.param_spaces['cat']['depth'][0])),
            'learning_rate': self.param_spaces['cat']['learning_rate'][0] + 
                            individual[idx+2] * (self.param_spaces['cat']['learning_rate'][1] - 
                                                self.param_spaces['cat']['learning_rate'][0]),
            'l2_leaf_reg': self.param_spaces['cat']['l2_leaf_reg'][0] + 
                          individual[idx+3] * (self.param_spaces['cat']['l2_leaf_reg'][1] - 
                                              self.param_spaces['cat']['l2_leaf_reg'][0]),
            'border_count': int(self.param_spaces['cat']['border_count'][0] + 
                               individual[idx+4] * (self.param_spaces['cat']['border_count'][1] - 
                                                   self.param_spaces['cat']['border_count'][0])),
            'bagging_temperature': self.param_spaces['cat']['bagging_temperature'][0] + 
                                  individual[idx+5] * (self.param_spaces['cat']['bagging_temperature'][1] - 
                                                      self.param_spaces['cat']['bagging_temperature'][0])
        }
        
        return params
    
    def evaluate_individual(self, individual):
        """Enhanced evaluation with ensemble fitness"""
        try:
            params = self.decode_individual(individual)
            
            # Create models with decoded parameters
            models = [
                RandomForestClassifier(**params['rf'], random_state=42, n_jobs=-1),
                SVC(**params['svm'], probability=True, random_state=42),
                LogisticRegression(**params['lr'], random_state=42, max_iter=5000),
                xgb.XGBClassifier(**params['xgb'], random_state=42, 
                                 use_label_encoder=False, eval_metric='logloss'),
                lgb.LGBMClassifier(**params['lgb'], random_state=42, verbose=-1),
                CatBoostClassifier(**params['cat'], random_state=42, verbose=False)
            ]
            
            # Use stratified k-fold
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            # Evaluate each model and combine scores
            model_scores = []
            for model in models:
                try:
                    scores = cross_val_score(model, self.X_train, self.y_train, 
                                           cv=cv, scoring='accuracy', n_jobs=-1)
                    model_scores.append(np.mean(scores))
                except Exception as e:
                    model_scores.append(0)
            
            # Weighted fitness (give more weight to best models)
            weights = [0.2, 0.15, 0.1, 0.25, 0.2, 0.1]  # XGB gets highest weight
            fitness = np.average(model_scores, weights=weights)
            
            return (fitness,)
        
        except Exception as e:
            return (0.0,)
    
    def optimize(self):
        """Run enhanced genetic algorithm optimization"""
        print("\n" + "="*60)
        print("🧬 GENETIC ALGORITHM OPTIMIZATION")
        print("="*60)
        
        # Create population
        population = self.toolbox.population(n=GA_PARAMS['population_size'])
        
        # Evaluate entire population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Hall of Fame
        hof = tools.HallOfFame(1)
        
        print(f"\nGeneration 0: Best={max(fitnesses)[0]:.4f}, Avg={np.mean(fitnesses):.4f}")
        
        # Run evolution
        for gen in range(GA_PARAMS['generations']):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < GA_PARAMS['crossover_prob']:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < GA_PARAMS['mutation_prob']:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population with elitism
            elite = tools.selBest(population, GA_PARAMS['elite_size'])
            population = tools.selBest(offspring, len(population) - GA_PARAMS['elite_size']) + elite
            
            # Update Hall of Fame
            hof.update(population)
            
            # Gather statistics
            record = stats.compile(population)
            self.fitness_history.append(record)
            
            print(f"Generation {gen+1}: Max={record['max']:.4f}, "
                  f"Avg={record['avg']:.4f}, Std={record['std']:.4f}")
            
            # Early stopping if target reached
            if record['max'] >= GA_PARAMS['min_accuracy']:
                print(f"\n✅ Target accuracy reached at generation {gen+1}!")
                break
        
        # Get best individual
        self.best_individual = hof[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        
        print(f"\n✅ Genetic Algorithm Complete!")
        print(f"   Best Fitness: {self.best_fitness:.4f}")
        print(f"   Final Population: {len(population)}")
        
        # Decode best parameters
        best_params = self.decode_individual(self.best_individual)
        
        # Print best parameters
        print("\n📊 Best Parameters Found:")
        for model_name, model_params in best_params.items():
            print(f"\n{model_name.upper()}:")
            for param, value in model_params.items():
                print(f"  {param}: {value}")
        
        return best_params