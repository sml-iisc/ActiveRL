from utils_common import *
from utils_plotting import plot_states

#-----------------------------------------------[HELPER FUNCTIONS]-----------------------------------------------#

def dataset_from_episode_list(episodes: list):
	states = []
	actions = []
	rewards = []
	terminals = []
	episode_terminals = []

	for epi in episodes:
		states.extend(epi.observations)
		actions.extend(epi.actions)
		rewards.extend(epi.rewards)

		terminal = np.zeros(epi.observations.shape[0])
		episode_terminal = np.zeros(epi.observations.shape[0])
		
		if epi.terminal:
			terminal[-1] = 1
			episode_terminal[-1] = 1
		else:
			episode_terminal[-1] = 1

		terminals.extend(terminal)
		episode_terminals.extend(episode_terminal)

	states = np.array(states)
	actions = np.array(actions)
	rewards = np.array(rewards)
	terminals = np.array(terminals)
	episode_terminals = np.array(episode_terminals)
	
	dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminals, episode_terminals)
	return dataset

def reduce_dataset(dataset: d3rlpy.dataset.MDPDataset,target_size: int):
	print("[UTILS] Reducing Dataset...")
	avg_epi_len = dataset.observations.shape[0] / dataset.size()
	retain_percentage = (target_size/avg_epi_len)/dataset.size()
	retain_percentage = min(retain_percentage, 0.9)

	_, retained_episode = train_test_split(dataset.episodes, test_size=retain_percentage)
	reduced_dataset = dataset_from_episode_list(retained_episode)
	return reduced_dataset


#-----------------------------------------------[POSITION BASED PRUNING]-----------------------------------------------#
#Works for AntMaze and Maze2D
def get_pruned_dataset(dataset: d3rlpy.dataset.MDPDataset, prune_condition):
    states = []
    actions = []
    rewards = []
    terminals = []
    episode_terminals = []

    for episode in tqdm(dataset.episodes):
        #Skip the episode if the initial state is in the medium region
        state_x, state_y = episode.observations[0][:2]
        if prune_condition(state_x, state_y):
            continue

        #Add the transitions to the dataset
        for transition in episode:
            #Check if current transition is in the medium region
            state_x, state_y = transition.next_observation[:2]
            if prune_condition(state_x, state_y):
                if len(episode_terminals)>0:
                    episode_terminals[-1] = 1.0
                
                break
            
            states.append(transition.observation)
            actions.append(transition.action)
            rewards.append(transition.reward)
            terminals.append(transition.terminal)
            episode_terminals.append(transition.terminal)

        #Last Transition
        episode_terminals[-1] = 1.0

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    episode_terminals = np.array(episode_terminals)

    pruned_dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminals, episode_terminals)
    return pruned_dataset

def prune_navigation(env_name, easy_cond, medium_cond, hard_cond):
    dataset, env = d3rlpy.datasets.get_d4rl(env_name)
    
    #Plotting Constants
    xlim, ylim, goal = get_environment_constants(env_name)

    #Remove Version Name from Env Name
    env_name = env_name[:-3]

    #Setup Directories
    setup_directory(f"DATASETS/{env_name}/")
    setup_directory(f"DATASETS_INFO/{env_name}/")

    pruned_dataset_easy = get_pruned_dataset(dataset, easy_cond)
    pruned_dataset_medium = get_pruned_dataset(dataset, medium_cond)
    pruned_dataset_hard = get_pruned_dataset(dataset, hard_cond)


    pruned_dataset_easy.dump(f"DATASETS/{env_name}/easy.h5")
    plot_states(pruned_dataset_easy.observations, xlim, ylim, goal, f"DATASETS_INFO/{env_name}/easy.jpg" )

    pruned_dataset_medium.dump(f"DATASETS/{env_name}/medium.h5")
    plot_states(pruned_dataset_medium.observations, xlim, ylim, goal, f"DATASETS_INFO/{env_name}/medium.jpg" )
    
    pruned_dataset_hard.dump(f"DATASETS/{env_name}/hard.h5")
    plot_states(pruned_dataset_hard.observations, xlim, ylim, goal, f"DATASETS_INFO/{env_name}/hard.jpg" )

def prune_pos_maze2d():
    #PRUNE MAZE2D LARGE
    prune_navigation("maze2d-large-v1", MAZE2D_LARGE_EASY, MAZE2D_LARGE_MEDIUM, MAZE2D_LARGE_HARD)

    #PRUNE MAZE2D MEDIUM
    prune_navigation("maze2d-medium-v1", MAZE2D_MEDIUM_EASY, MAZE2D_MEDIUM_MEDIUM, MAZE2D_MEDIUM_HARD)
    
    #PRUNE MAZE2D UMAZE
    prune_navigation("maze2d-umaze-v1", MAZE2D_UMAZE_EASY, MAZE2D_UMAZE_MEDIUM, MAZE2D_UMAZE_HARD)

def prune_pos_antmaze():
    #PRUNE ANTMAZE LARGE DIVERSE
    prune_navigation("antmaze-large-diverse-v0", ANTMAZE_LARGE_EASY, ANTMAZE_LARGE_MEDIUM, ANTMAZE_LARGE_HARD)

    #PRUNE ANTMAZE LARGE PLAY
    prune_navigation("antmaze-large-play-v0", ANTMAZE_LARGE_EASY, ANTMAZE_LARGE_MEDIUM, ANTMAZE_LARGE_HARD)

    #PRUNE ANTMAZE MEDIUM DIVERSE
    prune_navigation("antmaze-medium-diverse-v0", ANTMAZE_MEDIUM_EASY, ANTMAZE_MEDIUM_MEDIUM, ANTMAZE_MEDIUM_HARD)

    #PRUNE ANTMAZE MEDIUM PLAY
    prune_navigation("antmaze-medium-play-v0", ANTMAZE_MEDIUM_EASY, ANTMAZE_MEDIUM_MEDIUM, ANTMAZE_MEDIUM_HARD)

    #PRUNE ANTMAZE UMAZE
    prune_navigation("antmaze-umaze-v0", ANTMAZE_UMAZE_EASY, ANTMAZE_UMAZE_MEDIUM, ANTMAZE_UMAZE_HARD)

    #PRUNE ANTMAZE UMAZE DIVERSE
    prune_navigation("antmaze-umaze-diverse-v0", ANTMAZE_UMAZE_EASY, ANTMAZE_UMAZE_MEDIUM, ANTMAZE_UMAZE_HARD)

#------------------------------------------------[QUALITY BASED PRUNING]------------------------------------------------#
def prune_quality(env_name: str, percentage: int = 30):
    dataset, env = d3rlpy.datasets.get_d4rl(env_name)
    n_episodes = dataset.size()
    n_episodes_to_keep = int(n_episodes*percentage/100)

    #Compute the episodic returns
    episode_returns = [episode.compute_return() for episode in dataset.episodes]

    #Argsort the episodes
    sorted_idx = np.argsort(episode_returns)

    top_episode_index = sorted_idx[-n_episodes_to_keep:]
    bottom_episode_index = sorted_idx[:n_episodes_to_keep]
    random_episode_index = np.random.choice(sorted_idx, n_episodes_to_keep, replace=False)

    top_episodes = [dataset.episodes[i] for i in top_episode_index]
    bottom_episodes = [dataset.episodes[i] for i in bottom_episode_index]
    random_episodes = [dataset.episodes[i] for i in random_episode_index]

    top = dataset_from_episode_list(top_episodes)
    bottom = dataset_from_episode_list(bottom_episodes)
    random = dataset_from_episode_list(random_episodes)

    #Remove Version Name from Env Name
    env_name = env_name[:-3]

    #Setup Directories
    setup_directory(f"DATASETS/{env_name}/")

    top.dump(f"DATASETS/{env_name}/top{percentage}.h5")
    bottom.dump(f"DATASETS/{env_name}/bottom{percentage}.h5")
    random.dump(f"DATASETS/{env_name}/random{percentage}.h5")



if __name__ == "__main__":
    setup_directory("DATASETS/")
    setup_directory("DATASETS_INFO/")
    
    print("Pruning Position Based Datasets...")

    print("Pruning AntMaze...")
    prune_pos_antmaze()

    print("Pruning Maze2D...")
    prune_pos_maze2d()

    print("Pruning Quality Based Datasets...")

    print("Pruning AntMaze...")
    for envs in antmaze_envs:
        # prune_quality(envs, percentage = 50)
        prune_quality(envs, percentage = 30)
        prune_quality(envs, percentage = 20)
        prune_quality(envs, percentage = 10)

    print("Pruning Maze2D...")
    for envs in maze2d_envs:
        prune_quality(envs, percentage = 50)
        prune_quality(envs, percentage = 30)
        prune_quality(envs, percentage = 20)
        prune_quality(envs, percentage = 10)

    print("Pruning Locomotion...")
    for envs in locomotion_envs:
        prune_quality(envs, percentage = 50)
        prune_quality(envs, percentage = 30)
        prune_quality(envs, percentage = 20)
        prune_quality(envs, percentage = 10)



    

