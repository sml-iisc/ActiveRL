from utils_common import *
from utils_plotting import plot_states, plot_states_uncertainties, plot_collection, plot_buffer_evolution, plot_rewards
from utils_plotting import get_active_rewards
from utils_dataset import reduce_dataset
from utils_uncertainty import UncertaintyModel
from utils_uncertainty import Collector
from utils_dataset import dataset_from_episode_list

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--env', type=str, default="antmaze-medium-diverse-v0")
	parser.add_argument("--data", type=str, default = "random30")

	#IQL Parameters
	parser.add_argument("--actor_lr", default=3e-4, type=float)
	parser.add_argument("--critic_lr", default=3e-4, type=float)
	parser.add_argument("--expectile", default=0.9, type=float)
	parser.add_argument("--weight_temp", default=10.0, type=float)
	parser.add_argument("--max_weight", default=100.0, type=float)


	#Uncertainty Model Parameters
	parser.add_argument("--hidden_dim", default=512, type=int)
	parser.add_argument("--uncertainty_lr", default=3e-4, type=float)
	parser.add_argument("--num_models", default=5, type=int)
	parser.add_argument("--width", default=256, type=int)
	parser.add_argument("--depth", default=5, type=int)

	#Training Parameters
	parser.add_argument("--n_steps", default=50000, type=int)
	parser.add_argument("--n_steps_per_epoch", default=10000, type=int)
	parser.add_argument("--batch_size", default=256, type=int)

	#Active Parameters
	parser.add_argument("--active_epochs", default=10, type=int)
	parser.add_argument("--active_data_max_size", default=100000, type=int)

	parser.add_argument("--num_samples", default=5000, type=int)
	parser.add_argument("--candidate_size", default=20, type=int)
	parser.add_argument("--selection_size", default=2, type=int)
	parser.add_argument("--trajectory_size", default=200, type=int)

	parser.add_argument("--collection_method", default="active", type=str)
	parser.add_argument("--collection_policy", default="uncertain-dataset", type=str)
	parser.add_argument("--collection_epsilon", default=0.2, type=float)
	parser.add_argument("--active_epsilon", default=0.8, type=float)

	parser.add_argument("--es", default=False, action="store_true")
	parser.add_argument("--es_threshold", default=0.2, type=float)
	parser.add_argument("--es_percentile", default=60, type=float)
	parser.add_argument("--up_percentile", default=90, type=float)
	parser.add_argument("--pn_noise", default=0.05, type=float)

	#Extra arguments
	parser.add_argument("--n_trial", default=100, type=int)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--device", type=int, default=0)
	parser.add_argument("--show_progress", action="store_true", default=False)


	args = parser.parse_args()
	return args



def main(args):
	#Directories
	logdir = f"IQL_ACTIVE/{args.env}/"
	logdir += f"native" if args.data == "" else f"{args.data}/"
	experiment_name = "N_" 
	
	if args.collection_method == 'active':
		experiment_name+= f"a{args.active_epsilon}"

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
		experiment_name += f"u{args.collection_epsilon}_d{args.depth}_w{args.width}"
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

	experiment_root = f"{logdir}/{experiment_name}/"

	#Create Directory
	setup_directory(experiment_root)

	#Get Constants for plotting
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
		print(f"[IQL ACTIVE] Loaded Custom Dataset: {args.data}")

	#Reduce Dataset Size
	if dataset.observations.shape[0] > args.active_data_max_size:
		dataset = reduce_dataset(dataset, target_size=args.active_data_max_size)

	#Fix device
	device = torch.device(f"cuda:{args.device}")

	#Setup policy and metrics
	policy = d3rlpy.algos.IQL(actor_learning_rate=args.actor_lr,
									critic_learning_rate=args.critic_lr,
									batch_size=args.batch_size,
									weight_temp=args.weight_temp,
									max_weight=args.max_weight,
									expectile=args.expectile,
									reward_scaler=d3rlpy.preprocessing.ConstantShiftRewardScaler(shift=-1.0),
									use_gpu=args.device)
	
	policy.create_impl(dataset.get_observation_shape(), dataset.get_action_size())

	metrics = {"rewards_raw": d3rlpy.metrics.scorer.evaluate_on_environment(env, n_trials=args.n_trial)}
	
	
	#Learning Rate Scheduler
	lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(policy.impl._actor_optim, T_max=args.n_steps*args.active_epochs)

	def lr_scheduler_step(algo, epoch, total_step):
		lr_scheduler.step()

	


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
	
	#Save the dataset actions to sample from
	uncertainty_model.fit_dataset_action(actions=dataset.actions)
	
	#Setup Collector
	collector = Collector(env=env,
						  policy=policy,
						  unc_model=uncertainty_model,
						  config=args)
	

	#Load Policy and Uncertainty Model
	offline_experiment = "native" if args.data == "" else args.data
	policy.load_model(f"IQL_OFFLINE/{args.env}/{offline_experiment}/final.pt")
	uncertainty_model.load(f"IQL_OFFLINE/{args.env}/{offline_experiment}/uncertainty")

	#Reset Learning Rate
	for g in policy.impl._actor_optim.param_groups:
		g["lr"] = args.actor_lr


	#Plot Initial Uncertainty
	if args.env not in locomotion_envs:
		uncertainties = uncertainty_model.get_uncertainty_batched(active_states)
		plot_states_uncertainties(active_states.numpy(), uncertainties, xlim, ylim, goal, f"{logdir}/{experiment_name}/uncertainty_initial.jpg")

	#Plot Initial Dataset
	if args.env not in locomotion_envs:
		plot_states(dataset.observations, xlim, ylim, goal, f"{logdir}/{experiment_name}/dataset_initial.jpg")

	c_episodes = []


	#Train Policy
	print("======================================== TRAINING ACTIVE ========================================")

	for epoch in range(args.active_epochs):
		print(f"---------------------------------------- EPOCH {epoch} ----------------------------------------")
		#Do Active Collection
		if args.num_samples > 0:
			c_dataset = collector.collect(active_states, dataset)
			c_episodes.extend(c_dataset.episodes)

			#Plot Total Collection
			if args.env not in locomotion_envs:
				plot_collection(dataset_from_episode_list(c_episodes), xlim, ylim, goal, f"{logdir}/{experiment_name}/total_collection_{epoch}.jpg")

			#Plot Collection
			if args.env not in locomotion_envs:
				plot_collection(c_dataset, xlim, ylim, goal, f"{logdir}/{experiment_name}/collection_{epoch}.jpg")

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
					callback=lr_scheduler_step,
					with_timestamp=False)
		
		#Save Policy
		policy.save_model(f"{logdir}/{experiment_name}/Epoch_{epoch}/final.pt")


		#Plot the rewards
		rewards = get_active_rewards(experiment_root)
		plot_rewards(rewards, f"{experiment_root}/rewards_{epoch}.jpg", ma_window=5)


		#Train Uncertainty Model
		replay_buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=int(5e6),
															env=env,
															episodes=dataset.episodes)
		uncertainty_model.fit(replay_buffer,
								batch_size=args.batch_size,
								n_steps=args.n_steps)

		#Plot Uncertainty
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


