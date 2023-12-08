import numpy as np
import random

from FinalProject import *

class Student:
    def __init__(self,initialUnderstanding):
        self.initialUnderstanding = initialUnderstanding
        self.understanding = initialUnderstanding

def makeObservation(num_bins):
    observation = np.zeros((num_bins,2))
    
    for i in range(num_bins):
        observation[i][1] = (1/num_bins)*(i)
        observation[i][0] = 1 - observation[i][1]
    
    return observation

def beliefUpdate(model, current_action, current_observation):
    S, T, O, b = model.States, model.transition, model.Observations, model.belief
    new_belief = np.empty_like(b)

    for j in range(len(S)):
        po = O[current_action][j,current_observation]
        new_belief[j] = po*sum(T[current_action][i,j]*b[i] for i in range(len(S)))
    
    new_belief = new_belief/(np.sum(new_belief) + 1e-5)

    # print("b:", list(b))
    # print("observation:", current_observation)
    # print("transition:", T[current_action])
    # print("current_action:", current_action)
    # print("new_belief:", list(new_belief))

    return new_belief


def simulatedResponse(student):
    rnd_num = random.random()

    if rnd_num <= student.understanding:
        return 1
    else:
        return 0


def updateUnderstanding(student, model, current_action):
    currentUnderstanding = student.understanding

    current_state = int((currentUnderstanding*model.num_bins)) #gives index of state

    T = model.transition

    # print(current_state)

    transitionVector = T[current_action][current_state,:]

    # print(transitionVector)

    next_state = np.random.choice(np.arange(model.num_bins), p = transitionVector)

    updUnderstanding = (next_state)/model.num_bins

    return updUnderstanding

    
