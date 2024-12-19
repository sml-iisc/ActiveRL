import random
import argparse
import time
import glob
import os
import warnings
warnings.simplefilter("ignore")
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import d4rl
import d3rlpy
import mujoco_py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from sklearn.model_selection import train_test_split
#----------------------------------------------------------------------------- [Constants] -----------------------------------------------------------------------------#
#Valid Environments
antmaze_envs = [
	'antmaze-large-diverse-v0',
	'antmaze-large-play-v0',
	'antmaze-medium-diverse-v0',
	'antmaze-medium-play-v0',
	'antmaze-umaze-v0',
	'antmaze-umaze-diverse-v0'
]

maze2d_envs = [
	'maze2d-large-v1',
	'maze2d-medium-v1',
	'maze2d-umaze-v1'
]

locomotion_envs = [
	"halfcheetah-random-v2",
	"halfcheetah-medium-v2",
	"walker2d-random-v2",
	"walker2d-medium-v2",
	"hopper-random-v2",
	"hopper-medium-v2"
]



#Antmaze constants: X,Y Limits
ANTMAZE_XLIM_LARGE = (-5, 40)
ANTMAZE_YLIM_LARGE = (-2.5, 27.5)

ANTMAZE_XLIM_MEDIUM = (-5, 25)
ANTMAZE_YLIM_MEDIUM = (-5, 25)

ANTMAZE_XLIM_UMAZE = (-2, 11)
ANTMAZE_YLIM_UMAZE = (-2, 11)

#Antmaze constants: Goal Locations
ANTMAZE_GOAL_LARGE_DIVERSE = (32.74, 24.75)
ANTMAZE_GOAL_LARGE_PLAY = (32.70, 24.77)
ANTMAZE_GOAL_MEDIUM_DIVERSE = (20.72, 20.75)
ANTMAZE_GOAL_MEDIUM_PLAY = (20.72, 20.72)
ANTMAZE_GOAL_UMAZE = (0.76, 8.74)
ANTMAZE_GOAL_UMAZE_DIVERSE = (0.745, 8.73)

#Antmaze constants: Prune Conditions
ANTMAZE_LARGE_EASY = lambda x,y: x > 25 and y > 17
ANTMAZE_LARGE_MEDIUM = lambda x,y: y > 15
ANTMAZE_LARGE_HARD = lambda x,y: y > 5

ANTMAZE_MEDIUM_EASY = lambda x,y: x > 14 and y > 10
ANTMAZE_MEDIUM_MEDIUM = lambda x,y: y > 10
ANTMAZE_MEDIUM_HARD = lambda x,y: y > 5

ANTMAZE_UMAZE_EASY = lambda x,y: x < 6 and y > 6
ANTMAZE_UMAZE_MEDIUM = lambda x,y: y > 5
ANTMAZE_UMAZE_HARD = lambda x,y: x > 6 or y > 2

#Maze2d constants: X,Y Limits
MAZE2D_XLIM_LARGE = (0, 8)
MAZE2D_YLIM_LARGE = (0, 10.5)

MAZE2D_XLIM_MEDIUM = (0,7)
MAZE2D_YLIM_MEDIUM = (0,7)

MAZE2D_XLIM_UMAZE = (0,4)
MAZE2D_YLIM_UMAZE = (0,4)

#Maze2d constants: Goal Locations
MAZE2D_GOAL_LARGE = (6.93, 8.59)
MAZE2D_GOAL_MEDIUM = (5.89, 5.88)
MAZE2D_GOAL_UMAZE = (0.94, 1.065)


#Maze2d constants: Prune Conditions
MAZE2D_LARGE_EASY = lambda x,y: x > 5.2 and y > 7
MAZE2D_LARGE_MEDIUM = lambda x,y: y > 6.2
MAZE2D_LARGE_HARD = lambda x,y: y > 3.4

MAZE2D_MEDIUM_EASY = lambda x,y: x > 4 and y > 4.5
MAZE2D_MEDIUM_MEDIUM = lambda x,y: y > 4
MAZE2D_MEDIUM_HARD = lambda x,y: y > 2

MAZE2D_UMAZE_EASY = lambda x,y: x<1.5 and y<2
MAZE2D_UMAZE_MEDIUM = lambda x,y: x<2
MAZE2D_UMAZE_HARD = lambda x,y: x<2.5 or y > 2


#----------------------------------------------------------------------------- [Functions] -----------------------------------------------------------------------------#


def get_environment_constants(env_name):
	'''
	Returns the xlim, ylim, goal based on the environment name
	Inputs:
		env_name: str
	Outputs:
		xlim: tuple
		ylim: tuple
		goal: tuple
	'''
	if 'antmaze' in env_name:
		if 'large' in env_name:
			xlim = ANTMAZE_XLIM_LARGE
			ylim = ANTMAZE_YLIM_LARGE
			goal = ANTMAZE_GOAL_LARGE_DIVERSE if 'diverse' in env_name else ANTMAZE_GOAL_LARGE_PLAY
		if 'medium' in env_name:
			xlim = ANTMAZE_XLIM_MEDIUM
			ylim = ANTMAZE_YLIM_MEDIUM
			goal = ANTMAZE_GOAL_MEDIUM_DIVERSE if 'diverse' in env_name else ANTMAZE_GOAL_MEDIUM_PLAY
		if 'umaze' in env_name:
			xlim = ANTMAZE_XLIM_UMAZE
			ylim = ANTMAZE_YLIM_UMAZE
			goal = ANTMAZE_GOAL_UMAZE_DIVERSE if 'diverse' in env_name else ANTMAZE_GOAL_UMAZE
	elif 'maze2d' in env_name:
		if 'large' in env_name:
			xlim = MAZE2D_XLIM_LARGE
			ylim = MAZE2D_YLIM_LARGE
			goal = MAZE2D_GOAL_LARGE
		if 'medium' in env_name:
			xlim = MAZE2D_XLIM_MEDIUM
			ylim = MAZE2D_YLIM_MEDIUM
			goal = MAZE2D_GOAL_MEDIUM
		if 'umaze' in env_name:
			xlim = MAZE2D_XLIM_UMAZE
			ylim = MAZE2D_YLIM_UMAZE
			goal = MAZE2D_GOAL_UMAZE
	elif env_name in locomotion_envs:
		xlim = None
		ylim = None
		goal = None
	else:
		raise ValueError('Invalid Environment Name')
	

	return xlim, ylim, goal

def get_ma(arr: np.ndarray, window_size: int):
	'''
	Calculates the moving average of a numpy array with the same shape.

	Inputs:
		arr: np.ndarray
		window_size: int

	Outputs:
		ma: np.ndarray
	'''

	ma = [arr[max(0, i - window_size):i+1].mean() for i in range(arr.shape[0])]
	ma = np.array(ma)
	return ma

def seed_everything(seed:int):
	'''
	Seed everything
	Inputs:
		seed: int
	'''
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	d3rlpy.seed(seed)

def setup_directory(path):
	if not os.path.exists(path):
		os.makedirs(path)


def get_sim_state(env, state):
	#Handle Maze2d
	if env.spec.id in maze2d_envs:
		sim_state = mujoco_py.MjSimState(time=0.0, qpos=np.array(state[:2]), qvel=np.array(state[2:]), act=None, udd_state={})
	
	#Handle AntMaze
	elif env.spec.id in antmaze_envs:
		sim_state = mujoco_py.MjSimState(time=0.0, qpos=np.array(state[:15]), qvel=np.array(state[15:]), act=None, udd_state={})
	
	#Handle Locomotion
	elif env.spec.id in locomotion_envs:
		#Hopper
		if state.shape[0]<17:
			sim_state = mujoco_py.cymj.MjSimState(time=0.0, qpos=np.concatenate(([0.0], state[:5])), qvel=state[5:], act=None, udd_state={})
		
		#HalfCheetah, Walker2d
		else:
			sim_state = mujoco_py.cymj.MjSimState(time=0.0, qpos=np.concatenate(([0.0], state[:8])), qvel=state[8:], act=None, udd_state={})	
	else:
		raise ValueError

	return sim_state
