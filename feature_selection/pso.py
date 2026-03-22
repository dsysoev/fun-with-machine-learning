# Implementation of Partial swarm optimization algorithm
# @inproceedings{kennedy1995particle,
#   title={Particle swarm optimization},
#   booktitle={Proceedings of ICNN'95-international conference on neural networks},
#   volume={4},
#   pages={1942--1948},
#   year={1995},
#   organization={ieee}
# }


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import pyswarms as ps


def get_fitness(mask, alpha=0.9, beta=0.5, show_metrics=False):
    """Calculates the quality of the dataset (the smaller the better)."""
    if np.sum(mask) == 0:
        return 1.0
    
    _train = X_train[:, mask == 1]
    _valid = X_valid[:, mask == 1]

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(_train, y_train)

    pred_train = clf.predict_proba(_train)[:, 1]
    pred_valid = clf.predict_proba(_valid)[:, 1]
    score_train = roc_auc_score(y_train, pred_train)
    score_valid = roc_auc_score(y_valid, pred_valid)

    score = (
        (2 * score_train * score_valid) / (score_train + score_valid) 
        - (beta * abs(score_train - score_valid)) ** 2
    )
    
    num_features = np.count_nonzero(mask)
    p = num_features / dimensions
    fitness = alpha * (1 - score) + (1 - alpha) * p

    if show_metrics:
        _test = X_test[:, mask == 1]
        pred_test = clf.predict_proba(_test)[:, 1]
        score_test = roc_auc_score(y_test, pred_test)
        selected = [feature_names[i] for i in range(len(feature_names)) if mask[i] == 1]

        return {
            "roc_train": round(score_train, 4),
            "roc_valid": round(score_valid, 4),
            "roc_test": round(score_test, 4),
            "score": round(score, 4),
            "fitness": round(fitness, 4),
            "num_features": num_features,
            "selected_features": selected,
        }
    
    return fitness


def binary_pso(n_particles=15, iterations=20):
    dim = len(feature_names)
    
    pos = np.random.randint(2, size=(n_particles, dim))
    vel = np.random.uniform(-1, 1, (n_particles, dim))
    
    pbest_pos = pos.copy()
    pbest_fitness = np.array([get_fitness(p) for p in pos])
    
    gbest_idx = np.argmin(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]
    
    w, c1, c2 = 0.5, 1.5, 1.5
    for i in range(iterations):
        for j in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            
            vel[j] = (w * vel[j] + 
                      c1 * r1 * (pbest_pos[j] - pos[j]) + 
                      c2 * r2 * (gbest_pos - pos[j]))
            
            vel[j] = np.clip(vel[j], -4, 4)
            
            sigmoid = 1 / (1 + np.exp(-vel[j]))
            pos[j] = (np.random.rand(dim) < sigmoid).astype(int)
            
            current_fit = get_fitness(pos[j])
            
            if current_fit < pbest_fitness[j]:
                pbest_fitness[j] = current_fit
                pbest_pos[j] = pos[j].copy()
                
                if current_fit < gbest_fitness:
                    gbest_fitness = current_fit
                    gbest_pos = pos[j].copy()
        
        print(f"Iteration {i+1}/{iterations}, Best fitness: {gbest_fitness:.4f}")

    return gbest_pos, gbest_fitness


def f(x):
    n_particles = x.shape[0]
    j = [get_fitness(x[i]) for i in range(n_particles)]
    return np.array(j)


df, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
df.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['embarked'] = df['embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})

feature_names = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
dimensions = len(feature_names)
X = df[feature_names].values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# %%

best_mask, best_score = binary_pso()
print(get_fitness(best_mask, show_metrics=True))

# %%

options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 7, 'p': 2}
optimizer = ps.discrete.BinaryPSO(n_particles=15, dimensions=dimensions, options=options)

best_cost, best_pos = optimizer.optimize(f, iters=20)
print(get_fitness(best_mask, show_metrics=True))
