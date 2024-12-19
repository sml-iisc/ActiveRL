from utils_common import *
from utils_plotting import plot_states, plot_states_uncertainties, plot_collection, plot_buffer_evolution, plot_rewards
from utils_plotting import get_active_rewards
from utils_dataset import reduce_dataset
from utils_uncertainty import UncertaintyModel
from utils_uncertainty import Collector
from utils_dataset import dataset_from_episode_list

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--env', type=str, default="maze2d-large-v1")
	parser.add_argument("--data", type=str, default = "easy")

	#TD3BC Parameters
	parser.add_argument("--actor_lr", default=3e-4, type=float)
	parser.add_argument("--critic_lr", default=3e-4, type=float)
	parser.add_argument("--target_smoothing_sigma", default=0.2, type=float)
	parser.add_argument("--target_smoothing_clip", default=0.5, type=float)
	parser.add_argument("--alpha", default=2.5, type=float)
	parser.add_argument("--update_actor_interval", default=2, type=int)

	#Alpha Decay Parameters
	parser.add_argument("--alpha_decay_scale", default=5.0, type=float)

	
	#Uncertainty Model Parameters
	parser.add_argument("--hidden_dim", default=256, type=int)
	parser.add_argument("--uncertainty_lr", default=3e-4, type=float)
	parser.add_argument("--num_models", default=5, type=int)
	parser.add_argument("--width", default=2048, type=int)
	parser.add_argument("--depth", default=5, type=int)

	#Training Parameters
	parser.add_argument("--n_steps", default=25000, type=int)
	parser.add_argument("--n_steps_per_epoch", default=5000, type=int)
	parser.add_argument("--batch_size", default=256, type=int)

	#Active Parameters
	parser.add_argument("--active_epochs", default=20, type=int)
	parser.add_argument("--active_maxd", default=100000, type=int)

	parser.add_argument("--num_samples", default=5000, type=int)
	parser.add_argument("--candidate_size", default=20, type=int)
	parser.add_argument("--selection_size", default=2, type=int)
	parser.add_argument("--trajectory_size", default=200, type=int)

	parser.add_argument("--collection_method", default="active", type=str)
	parser.add_argument("--collection_policy", default="uncertain-dataset", type=str)
	parser.add_argument("--collection_epsilon", default=0.5, type=float)

	parser.add_argument("--es", default=False, action="store_true")
	parser.add_argument("--es_percentile", default=60, type=float)
	parser.add_argument("--up_percentile", default=90, type=float)
	parser.add_argument("--pn_noise", default=0.05, type=float)
	parser.add_argument("--active_epsilon", default=1.0, type=float)

	#Extra arguments
	parser.add_argument("--n_trial", type=int, default=15)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--device", type=int, default=0)
	parser.add_argument("--show_progress", action="store_true", default=False)

	args = parser.parse_args()
	return args


def main(args):
	#Directories
	logdir = f"TD3BC_FINAL/{args.env}/"
	logdir += f"native" if args.data == "" else f"{args.data}/"
	experiment_name = "" 

	if args.collection_method == 'active':
		experiment_name+= "a"

	#RANDOM: Pick the first states
	elif args.collection_method == 'random':
		experiment_name += "r"

	#INITIAL: Pick the initial states
	elif args.collection_method == 'initial':
		experiment_name += "i"
	else:
		raise ValueError


	if 'random' in args.collection_policy:
		experiment_name += "r"

	elif 'policy' in args.collection_policy:
		experiment_name += "p"

	elif 'uncertain' in args.collection_policy:
		if 'dataset' in args.collection_policy:
			experiment_name += f"ud{args.collection_epsilon}_d{args.depth}_w{args.width}"
		else:
			experiment_name += f"uu{args.collection_epsilon}_d{args.depth}_w{args.width}"
	elif 'up' in args.collection_policy:
		experiment_name += f"up({args.up_percentile},{args.es_percentile})_d{args.depth}_w{args.width}"
	elif 'pn' in args.collection_policy:
		experiment_name += f"pn{args.pn_noise}_eps{args.collection_epsilon}_d{args.depth}_w{args.width}"
	else:
		raise ValueError

	experiment_name += f"_t{args.trajectory_size}_cs{args.candidate_size}_ss{args.selection_size}"
	
	if args.es:
		experiment_name += f"_es"

	experiment_name += f"_{args.seed}"
	experiment_name += f"_{args.n_steps}_{args.num_samples}"

	experiment_root = f"{logdir}/{experiment_name}/"

	#Create Directory and constants
	setup_directory(experiment_root)
	xlim, ylim, goal = get_environment_constants(args.env)


	#Seed Everything
	seed_everything(args.seed)

	#Get Environment and Dataset
	dataset, env = d3rlpy.datasets.get_d4rl(args.env)
	active_states = torch.from_numpy(dataset.observations).float()
	env.seed(args.seed)

	#Load Pruned Dataset if specified
	if args.data != "":
		dataset_path = f"DATASETS/{args.env[:-3]}/{args.data}.h5"
		dataset = d3rlpy.dataset.MDPDataset.load(dataset_path)
		print(f"[TD3BC ACTIVE] Loaded Custom Dataset: {args.data}")

	#Reduce Dataset Size
	if dataset.observations.shape[0] > args.active_maxd:
		dataset = reduce_dataset(dataset, target_size=args.active_maxd)

	#Fix device
	device = torch.device(f"cuda:{args.device}")

	#Setup policy and metrics
	state_scaler = d3rlpy.preprocessing.StandardScaler(dataset)
	policy = d3rlpy.algos.TD3PlusBC(actor_learning_rate=args.actor_lr,
									critic_learning_rate=args.critic_lr,
									batch_size=args.batch_size,
									target_smoothing_sigma=args.target_smoothing_sigma,
									target_smoothing_clip=args.target_smoothing_clip,
									alpha=args.alpha,
									update_actor_interval=args.update_actor_interval,
									scaler=state_scaler,
									use_gpu=args.device)
	
	
	policy.build_with_dataset(dataset)
	metrics = {"rewards_raw": d3rlpy.metrics.scorer.evaluate_on_environment(env, n_trials=args.n_trial)}
	
	
	decay_rate = np.exp((1/args.n_steps*args.active_epochs)*np.log(args.alpha_decay_scale))
	def decay_alpha(algo, epoch, total_step):
		algo.impl._alpha = min(algo.impl._alpha*decay_rate, 12.5)


	#Setup Uncertainty Model
	uncertainty_model = UncertaintyModel(state_dim = env.observation_space.shape[0], 
										 action_dim = env.action_space.shape[0],
										 max_action = env.action_space.high[0],
										 hidden_dim = args.hidden_dim,
										 lr=args.uncertainty_lr,
										 num_models=args.num_models,
										 width=args.width,
										 depth=args.depth,
										 device=device)
	
	uncertainty_model.fit_dataset_action(actions=dataset.actions)
	
	#Setup Collector
	collector = Collector(env=env,
						  policy=policy,
						  unc_model=uncertainty_model,
						  config=args)
	

	#Load Policy and Uncertainty Model
	offline_experiment = "native" if args.data == "" else args.data
	policy.load_model(f"TD3BC_OFFLINE/{args.env}/{offline_experiment}/final.pt")
	uncertainty_model.load(f"TD3BC_OFFLINE/{args.env}/{offline_experiment}/uncertainty")

	#Plot Initial Uncertainty
	if args.env not in locomotion_envs:
		uncertainties = uncertainty_model.get_uncertainty_batched(active_states)
		plot_states_uncertainties(active_states.numpy(), uncertainties, xlim, ylim, goal, f"{logdir}/{experiment_name}/uncertainty_initial.jpg")

	#Plot Initial Dataset
	if args.env not in locomotion_envs:
		plot_states(dataset.observations, xlim, ylim, goal, f"{logdir}/{experiment_name}/dataset_initial.jpg")

	#Train Policy
	c_episodes = []
	print("======================================== TRAINING ACTIVE ========================================")
	
	for epoch in range(args.active_epochs):
		print(f"---------------------------------------- EPOCH {epoch} ----------------------------------------")
		
		if args.num_samples > 0:
			#Do Active Collection
			c_dataset = collector.collect(active_states)
			c_episodes.extend(c_dataset.episodes)

			#Plot Collection
			if args.env not in locomotion_envs:
				plot_collection(c_dataset, xlim, ylim, goal, f"{logdir}/{experiment_name}/collection_{epoch}.jpg")

			#Plot total collection
			if args.env not in locomotion_envs:
				plot_collection(dataset_from_episode_list(c_episodes), xlim, ylim, goal, f"{logdir}/{experiment_name}/total_collection_{epoch}.jpg")


			#Plot Buffer Evolution
			if args.env not in locomotion_envs:
				plot_buffer_evolution(dataset.observations, c_dataset.observations, xlim, ylim, goal, f"{logdir}/{experiment_name}/buffer_evolution_{epoch}.jpg")
			
			#Add to Dataset
			dataset.extend(c_dataset)

		#Train Policy
		policy.fit(dataset.episodes,
					n_steps=args.n_steps,
					n_steps_per_epoch=args.n_steps_per_epoch,
					eval_episodes=dataset.episodes,
					scorers=metrics,
					save_interval=10,
					logdir=f"{logdir}/{experiment_name}",
					experiment_name=f"Epoch_{epoch}",
					show_progress=args.show_progress,
					callback=decay_alpha,
					with_timestamp=False)
		
		#Save Policy
		policy.save_model(f"{logdir}/{experiment_name}/Epoch_{epoch}/final.pt")


		#Plot the rewards
		rewards = get_active_rewards(experiment_root)
		plot_rewards(rewards, f"{experiment_root}/rewards_{epoch}.jpg")


		#Train Uncertainty Model
		replay_buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=int(5e6),
															env=env,
															episodes=dataset.episodes)
		uncertainty_model.fit(replay_buffer,
								batch_size=args.batch_size,
								n_steps=args.n_steps)

		# Plot Uncertainty
		if args.env not in locomotion_envs:
			uncertainties = uncertainty_model.get_uncertainty_batched(active_states)
			plot_states_uncertainties(active_states.numpy(), uncertainties, xlim, ylim, goal, f"{logdir}/{experiment_name}/uncertainty_{epoch}.jpg")
		
		#Save Uncertainty Model
		uncertainty_model.save(f"{logdir}/{experiment_name}/Epoch_{epoch}/uncertainty")
		print("---------------------------------------------------------------------------------------------")
	
	
	#Save Policy
	policy.save_model(f"{logdir}/{experiment_name}/final.pt")
	
	#Save Uncertainty Model
	uncertainty_model.save(f"{logdir}/{experiment_name}/uncertainty")


if __name__ == "__main__":
	args = parse_args()
	main(args)


