import operator
import random

import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from deap import base
from deap import creator
from deap import tools
from ADR.preprocess import *


bgl_log_path = Path(r'data\Drain_result\bgl_10k\BGL.log_structured500k.csv')
bgl_template_path = Path(r'data\Drain_result\bgl_10k\BGL.log_templates.csv')

df_bgl_logs = pd.read_csv(bgl_log_path, sep=',', header=0, nrows=100000)
df_bgl_logs["bLabel"] = 1
df_bgl_logs.loc[df_bgl_logs["Label"]=="-", "bLabel"] = 0
list_EventIds = pd.read_csv(bgl_template_path, header=0)["EventId"].tolist()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, best=None)

def generate(size, pmin, pmax):
    part = creator.Particle(np.random.randint(low=pmin, high=pmax, size=size))
    part.speed = np.random.rand(*part.shape)
    return part

def updateParticle(part, best, omega, epsilon1, epsilon2):
    part.speed = omega * part.speed + np.random.rand() * epsilon1 * (part.best-part) + np.random.rand() * epsilon2 * (best - part)
    part = part + part.speed

def evaluate(part):
    window_size = part[0]
    step_size = 1
    identifier = "Node"
    collect_df_ECM_windows_N_sessions = pd.DataFrame()
    collect_df_ECM_windows_N_sessions_bLabels = pd.Series()

    collect_df_ECM_windows_AN_sessions = pd.DataFrame()
    collect_df_ECM_windows_AN_sessions_bLabels = pd.Series()

    for iden, df_iden in df_bgl_logs.groupby(identifier):
        if df_iden.shape[0] < window_size:
            continue
        df_iden_label = df_iden["bLabel"].astype(int).any()
        if df_iden_label == 0:
            df_iden = df_iden.reset_index(drop=True)
            df_ECM_windows, df_ECM_windows_bLabels = ECM_by_NumEventWindow(df_iden, list_EventIds=list_EventIds, col_timestamp="Timestamp", col_EventId="EventId", window_size=window_size, step_size=step_size, col_bLabel="bLabel")
            collect_df_ECM_windows_N_sessions = collect_df_ECM_windows_N_sessions.append(df_ECM_windows, sort=False)
            collect_df_ECM_windows_N_sessions_bLabels = collect_df_ECM_windows_N_sessions_bLabels.append(df_ECM_windows_bLabels)
        elif df_iden_label == 1:
            df_iden = df_iden.reset_index(drop=True)
            df_ECM_windows, df_ECM_windows_bLabels = ECM_by_NumEventWindow(df_iden, list_EventIds=list_EventIds, col_timestamp="Timestamp", col_EventId="EventId", window_size=window_size, step_size=step_size, col_bLabel="bLabel")
            collect_df_ECM_windows_AN_sessions = collect_df_ECM_windows_AN_sessions.append(df_ECM_windows, sort=False)
            collect_df_ECM_windows_AN_sessions_bLabels = collect_df_ECM_windows_AN_sessions_bLabels.append(df_ECM_windows_bLabels)

    collect_npa_ECM_windows_N_sessions = collect_df_ECM_windows_N_sessions.values
    collect_npa_ECM_windows_AN_sessions = collect_df_ECM_windows_AN_sessions.values
    rank_collect_npa_ECM_windows_N_sessions = np.linalg.matrix_rank(collect_npa_ECM_windows_N_sessions)
    rank_collect_npa_ECM_windows_AN_sessions = np.linalg.matrix_rank(collect_npa_ECM_windows_AN_sessions) if collect_npa_ECM_windows_AN_sessions.shape != (0,0) else 0
    return rank_collect_npa_ECM_windows_AN_sessions-rank_collect_npa_ECM_windows_N_sessions,

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=(1), pmin=0, pmax=len(list_EventIds))
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, epsilon1=1.49445, epsilon2=1.49445)
toolbox.register("evaluate", evaluate)

def main():
    print('=====start PSO=====')
    print('(this step may take long, but it only needs to be run one time to get the optimal window size)')
    pop = toolbox.population(n=2)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 350
    best = None

    for g in range(GEN):
        print(f"total generations: 350, current generation: {g+1}")
        for part in pop:
            omega =  0.9 - (0.9 - 0.4)*(g/GEN)**2
            part.fitness.values = toolbox.evaluate(part)
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best, omega)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
        print(f"best is {best}")
    return pop, logbook, best

if __name__ == "__main__":
    pop, logbook, best = main()