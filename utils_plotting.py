from utils_common import *

# -------------------------------------------- [HELPER FUNCTIONS] -------------------------------------------- #
def get_active_rewards(experiment_root: str, metric = "rewards_raw"):
    '''
    Returns the training rewards array for the active experiment.
    Inputs:
        experiment_root: str containing the path to the experiment root
    Outputs:
        rewards: np.ndarray of shape (N,) containing the rewards
    '''
    epochs = glob.glob(f"{experiment_root}/Epoch_*/")
    epochs = sorted(epochs, key= lambda s: int(s.split('/')[-2].split('_')[-1]))

    rewards = []

    for epoch in epochs:
        epoch_rewards = pd.read_csv(f"{epoch}/{metric}.csv", header=None)[2].to_numpy().tolist()
        rewards.extend(epoch_rewards)

    rewards = np.array(rewards)
    return rewards
    
# -------------------------------------------- [PLOT FUNCTIONS] -------------------------------------------- #
def plot_collection(collected_dataset: d3rlpy.dataset.MDPDataset, xlim: tuple, ylim: tuple, goal: tuple, save_path: str = "test.jpg"):
    '''
    Plots the Active Datacollection process. Initial states are shown in red and the collected states in blue.

    Inputs:
        collected_states: d3rlpy.dataset.MDPDataset (containing the collected states)
        save_path: str containing the path to save the plot
    '''
    print("[PLOT UTIL] Plotting Collection...")

    #Setup the plot
    plt.figure(figsize=(10, 10))
    plt.title("Active Data Collection")
    plt.xlim(xlim)
    plt.ylim(ylim)

    for episode in collected_dataset.episodes:
        #Plot Initial States
        plt.scatter(episode.observations[0, 0], episode.observations[0, 1], color="red", s=10)

        #Plot Collected States
        plt.scatter(episode.observations[:, 0], episode.observations[:, 1], color="blue", s=0.1)

    #Plot Goal
    plt.scatter(goal[0], goal[1], color="lime", label="Goal", s=100, marker="^")

    plt.savefig(save_path)
    plt.close()

def plot_buffer_evolution(old_states: np.ndarray, new_states: np.ndarray, xlim: tuple, ylim: tuple, goal: tuple, save_path: str = "test.jpg"):
    '''
    Plots the evolution of the buffer. Old states are shown in red and the new states in blue.

    Inputs:
        old_states: np.ndarray of shape (N, >=2) containing the old states
        new_states: np.ndarray of shape (N, >=2) containing the new states
        save_path: str containing the path to save the plot
    '''
    print("[PLOT UTIL] Plotting Buffer Evolution...")
    #Setup the plot
    plt.figure(figsize=(10, 10))
    plt.title("Buffer Evolution")
    plt.xlim(xlim)
    plt.ylim(ylim)

    #Plot Old States
    plt.scatter(old_states[:, 0], old_states[:, 1], color="red", label="Old States", s=0.05)

    #Plot New States
    plt.scatter(new_states[:, 0], new_states[:, 1], color="blue", label="New States", s=0.05)

    #Plot Goal
    plt.scatter(goal[0], goal[1], color="lime", label="Goal", s=100, marker="^")

    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_states(states: np.ndarray, xlim: tuple, ylim: tuple, goal: tuple, save_path: str = "test.jpg"):
    '''
    Plots the states.

    Inputs:
        states: np.ndarray of shape (N, >=2) containing the states
        save_path: str containing the path to save the plot
    '''
    print("[PLOT UTIL] Plotting States...")
    #Setup the plot
    plt.figure(figsize=(10, 10))
    plt.title("States")
    plt.xlim(xlim)
    plt.ylim(ylim)

    #Plot States
    plt.scatter(states[:, 0], states[:, 1], color="blue", label="States", s=0.03, alpha=0.5)

    #Plot Goal
    plt.scatter(goal[0], goal[1], color="lime", label="Goal", s=100, marker="^")

    plt.savefig(save_path)
    plt.close()

def plot_rewards(rewards: np.ndarray, save_path: str = "test.jpg", ma_window = 10):
    '''
    Plots the rewards.

    Inputs:
        rewards: np.ndarray of shape (N,) containing the rewards
    '''
    print("[PLOT UTIL] Plotting Rewards...")

    #Calculate Moving Average
    rewards_ma = get_ma(rewards, ma_window)

    #Setup the plot
    plt.figure(figsize=(15, 10))
    plt.title("Rewards")
    plt.xlim(0, len(rewards))

    #Plot Rewards
    plt.plot(rewards_ma, color="blue", label=f"Rewards(MA{ma_window})", linewidth=3)
    plt.plot(rewards, color="blue", label="Rewards", linewidth=1, alpha=0.2, linestyle="dashed")
    plt.grid()
    plt.legend()

    plt.savefig(save_path)
    plt.close()

def plot_states_uncertainties(states: np.ndarray, uncertainties: np.ndarray, xlim: tuple, ylim: tuple, goal: tuple, save_path: str = "test.jpg"):
    '''
    Plots the states.

    Inputs:
        states: np.ndarray of shape (N, >=2) containing the states
        uncertainties: np.ndarray of shape (N,) containing the uncertainties
        save_path: str containing the path to save the plot
    '''
    print("[PLOT UTIL] Plotting States Uncertainties...")

    #Plot the size of the states
    selected_indices = np.random.permutation(states.shape[0])[:min(states.shape[0],400000)]
    states = states[selected_indices]
    uncertainties = uncertainties[selected_indices]

    #Setup the plot
    plt.figure(figsize=(10, 10))
    plt.title("States Uncertainties")
    plt.xlim(xlim)
    plt.ylim(ylim)

    #Plot States with Uncertainties as Color
    plt.scatter(states[:,0], states[:,1], c=uncertainties, cmap='viridis', s=0.5)
    plt.colorbar()
    
    #Plot Goal
    plt.scatter(goal[0], goal[1], color="lime", label="Goal", s=100, marker="^")

    plt.savefig(save_path)
    plt.close()