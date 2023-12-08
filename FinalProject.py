#Final Project - Intelligent Tutoring 
# Katie, Saum, and Olivia
# Offline Implementation

from re import A, S, T
import sys
import numpy as np
import pandas as pd
import time
from scipy.stats import norm
from Simulate import*

# Create class with POMDP model
class POMDP:
    def __init__(self, num_bins):
        self.num_bins = num_bins
        self.States = np.linspace(0, 1 - (1/num_bins), num_bins)
        self.Actions = [0,1,2]
        self.discount = 0.99
        self.transition = [makeTransitionMatrix(num_bins, 0), makeTransitionMatrix(num_bins, 1), np.eye(num_bins)]
        self.Observations = ([makeObservation(num_bins), np.ones((num_bins,2))])
        self.belief = (1/num_bins)*np.ones((num_bins,1)) #column vector

def reward_vec(num_bins, current_action):
    if current_action == 0:
        return np.ones((num_bins, ))*1
    elif current_action == 1:
        return np.ones((num_bins, ))*-2
    else:
        return np.arange(num_bins)*5

def makeTransitionMatrix(num_bins, action):
    transitionMatrix = []
    for i in range(num_bins):
        current_state = i/num_bins
        if action == 0:#question
            mean_update = 0.01*(current_state)
        elif action == 1:#hint
            mean_update = 0.01*(1-current_state)
        # mean_update = 0.1
        # sd = 0.1
        sd = mean_update+1e-5
        transitionMatrix.append(transition(current_state, mean_update, sd, num_bins))
    transitionMatrix = np.array(transitionMatrix)

    return transitionMatrix

def transition(current_state, mean_update, sd, num_bins = 10):
    mean = current_state+mean_update
    T= []
    bin_size = 1/num_bins
    T_sum = 0
    for i in range(num_bins):
        bin_prob = norm.cdf((i + 0.5) * bin_size, mean, sd) - norm.cdf((i - 0.5) * bin_size, mean, sd)
        T.append(bin_prob)
        T_sum += bin_prob

    for i in range(num_bins):
        T[i]/=T_sum

    return T

def alphavector_iteration(model, gamma):
    convergence = 0.001 # Can modify this value if we want
    gamma_prev = np.copy(gamma)
    # print("gamma_prev:", gamma_prev)
    gamma = update(model, gamma)

    while np.linalg.norm(gamma_prev - gamma) > convergence:
        # print("gamma:", gamma)
        gamma_prev = np.copy(gamma)
        gamma = update(model, gamma)
    
    return gamma

def evaluateAlphaVectors(model, gamma):
    b = model.belief
    dot_products = [np.transpose(gamma[:,i]).dot(b) for i in range(len(model.Actions))]
    utility = np.max(dot_products)
    i = np.argmax(dot_products)
    optAction = model.Actions[i]
    return optAction

def QMDP_Solve(num_bins):
    model = POMDP(num_bins)
    gamma = np.zeros((num_bins, 3)) 
    gamma = alphavector_iteration(model, gamma)
    return gamma 

def update(model, gamma):
    new_gamma = []
    for a in model.Actions:
        new_alpha_a = reward_vec(model.num_bins, a) + model.discount*model.transition[a]@np.max(gamma, axis = 1)
        new_gamma.append(new_alpha_a)
    return np.array(new_gamma).T


