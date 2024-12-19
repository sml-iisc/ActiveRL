from utils_common import *

#Model that predicts dynamics epsilon
class DynamicsModel(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim = 256):
		super(DynamicsModel, self).__init__()
		self.fc1_state = nn.Linear(state_dim, hidden_dim)
		self.fc1_action = nn.Linear(action_dim, hidden_dim)
		self.fc2_state_action = nn.Linear(2 * hidden_dim, hidden_dim)
		self.fc1 = nn.Linear(state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, state_dim)
		
	def forward(self, state, action=None):
		if action is None:
			x = F.relu(self.fc1(state))
			x = F.relu(self.fc2(x))
		
		else:
			state_em = F.relu(self.fc1_state(state))
			action_em = F.relu(self.fc1_action(action))
			x = torch.cat([state_em, action_em], dim=1)
			x = F.relu(self.fc2_state_action(x))
		
		x = self.fc3(x)
		return x




class UncertaintyModel(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256, lr: float = 3e-4, num_models: int = 5,
	 width: int = 2048, depth: int = 5, device = torch.device('cpu')):
		super(UncertaintyModel, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action

		self.hidden_dim = hidden_dim
		self.lr = lr

		self.num_models = num_models
		self.width = width
		self.depth = depth

		self.device = device

		#Dynamics Model
		self.models = [DynamicsModel(state_dim, action_dim, hidden_dim).to(device) for _ in range(num_models)]
		self.optims = [torch.optim.Adam(model.parameters(), lr = lr) for model in self.models]

		#Initialize Dataset Actions
		self.dataset_actions = None

		
	def update(self, transition_minibatch: d3rlpy.dataset.TransitionMiniBatch):
		observations = torch.from_numpy(transition_minibatch.observations).float().to(self.device)
		actions = torch.from_numpy(transition_minibatch.actions).float().to(self.device)
		next_observations = torch.from_numpy(transition_minibatch.next_observations).float().to(self.device)

		model_index = np.random.randint(0, self.num_models)
		model = self.models[model_index]
		optim = self.optims[model_index]


		eps_actual = next_observations - observations
		eps_pred_1 = model(observations, actions)
		eps_pred_2 = model(observations)

		#Compute loss
		loss = F.mse_loss(eps_pred_1, eps_actual) + F.mse_loss(eps_pred_2, eps_actual)

		#Backprop
		optim.zero_grad()
		loss.backward()
		optim.step()

	def fit(self, replay_buffer: d3rlpy.online.buffers.ReplayBuffer, batch_size = 256, n_steps = int(1e5)):
		#Train the models
		for i in range(n_steps):
			#Sample a batch
			batch = replay_buffer.sample(batch_size)
			self.update(batch)

			# if (i+1)%5000 == 0:
			# 	print(f"Trained {(i+1)} Updates")

	def fit_dataset_action(self, actions: np.ndarray):
		self.dataset_actions = actions

	def sample_actions_dataset(self, n_actions):
		if self.dataset_actions is None:
			print("Dataset actions are not fitted yet")
			return None
		
		indices = np.random.choice(self.dataset_actions.shape[0], n_actions, replace=False)
		actions = self.dataset_actions[indices]

		return actions

	def get_uncertainty(self, state: torch.Tensor, action: torch.Tensor=None):
		predictions = []
		with torch.no_grad():
			for i in range(self.num_models):
				predictions.append(self.models[i](state, action))

		predictions = torch.stack(predictions).permute(1,0,2)
		unc_matrix = torch.zeros((state.shape[0], self.num_models, self.num_models)).to(self.device)

		for k1  in range(self.num_models):
			for k2 in range(self.num_models):
				unc_matrix[:, k1, k2] = torch.norm(predictions[:, k1, :] - predictions[:, k2, :], dim = 1)

		uncertainties = unc_matrix.reshape(state.shape[0], -1).max(dim = 1)[0]
		return uncertainties

	def get_uncertainty_batched(self, state: torch.Tensor, batch_size = 1024):
		n_data = state.shape[0]
		n_batches = np.ceil(n_data/batch_size).astype(int)

		uncertainties = np.zeros(n_data)
		for i in range(n_batches):
			batch_states = state[i*batch_size:(i+1)*batch_size].to(self.device)
			uncertainties[i*batch_size:(i+1)*batch_size] = self.get_uncertainty(batch_states).cpu().numpy()

		return uncertainties

	def get_uncertain_action(self, state: np.ndarray, policy = 'uncertain-uniform'):
		num_samples = self.width
		depth = self.depth
		#Select a model at random
		model_indices = np.random.randint(0, self.num_models, size = num_samples)

		#Repeat the observations for num_samples times
		state = np.repeat(state[None, :], num_samples, axis = 0)
		if 'uniform' in policy:
			actions = np.random.rand(num_samples,self.action_dim)*2*self.max_action - self.max_action
		elif 'dataset' in policy:
			actions = self.sample_actions_dataset(num_samples)
		else:
			raise NotImplementedError
		
		actions_orig = actions.copy()

		total_uncertainty = np.zeros(num_samples)
		for i in range(depth):
			state_t = torch.from_numpy(state).float().to(self.device)
			actions_t = torch.from_numpy(actions).float().to(self.device)

			next_state_ts = []
			with torch.no_grad():
				for i, model in enumerate(self.models):
					next_state_ts.append((model(state_t, actions_t)+state_t).detach().cpu())

			next_state_ts = torch.stack(next_state_ts)
			uncertainty_matrices = np.zeros((num_samples, self.num_models, self.num_models))

			for i in range(self.num_models):
				for j in range(self.num_models):
					sq_diff = (next_state_ts[i,:,:] - next_state_ts[j,:,:])**2
					uncertainty_matrices[:,i,j] = sq_diff.sum(axis = 1).numpy()
			
			uncertainty_matrices = np.sqrt(uncertainty_matrices)
			uncertainty = uncertainty_matrices.max(axis = 1).max(axis = 1)

			total_uncertainty += uncertainty

			state_t = next_state_ts[model_indices, np.arange(num_samples), :]
			state = state_t.numpy()

		action = actions_orig[np.argmax(total_uncertainty)]
		return action

	def save(self, filename):
		for i in range(self.num_models):
			torch.save(self.models[i].state_dict(), filename + f"_{i}")
			torch.save(self.optims[i].state_dict(), filename + f"_{i}_optim")

	def load(self, filename, device = torch.device('cpu')):
		for i in range(self.num_models):
			self.models[i].load_state_dict(torch.load(filename + f"_{i}", map_location = device))
			self.optims[i].load_state_dict(torch.load(filename + f"_{i}_optim", map_location = device))



class Collector:
	def __init__(self, env: gym.Env, policy: d3rlpy.algos.AlgoBase, unc_model: UncertaintyModel, config: argparse.Namespace):
		self.env = env
		self.policy = policy
		self.unc_model = unc_model

		#Collection Parameters
		self.num_samples = config.num_samples
		self.candidate_size = config.candidate_size
		self.selection_size = config.selection_size
		self.trajectory_size = config.trajectory_size
		self.config_trajectory_size = config.trajectory_size

		#Collection Strategy
		self.collection_method = config.collection_method
		self.collection_policy = config.collection_policy
		self.collection_epsilon = config.collection_epsilon

		#Early Stopping
		self.early_stop = config.es
		self.early_stop_thershold = 0.0

		#Uncertain-Policy Configuration
		self.up_threshold = float("inf")

		#Counters
		self.uncertain_actions = 0
		self.policy_actions = 0
		self.random_actions = 0
		self.n_train = 0

		#Extra Config
		self.up_percentile = config.up_percentile
		self.es_percentile = config.es_percentile
		self.pn_noise = config.pn_noise
		self.active_epsilon = config.active_epsilon

	def _collect_trajectory(self, state: np.ndarray, to_collect: int):
		n_collection = 0
		

		sim_state = get_sim_state(self.env, state)
		self.env.sim.set_state(sim_state)

		current_unc = float("inf")

		#Collect Trajectory
		done = False
		while not done:
			#RANDOM: Pick actions randomly (Uniformly)
			if 'random' in self.collection_policy:
				action = self.env.action_space.sample()
				self.random_actions += 1

			#POLICY: Pick actions from the policy
			elif 'policy' in self.collection_policy:
				action = self.policy.sample_action(state.reshape(1,-1)).reshape(-1)
				self.policy_actions += 1

			#UNCERTAIN: Pick actions from the uncertainty model based on epsilon greedy
			elif 'uncertain' in self.collection_policy:
				if np.random.rand() < self.collection_epsilon:
					action = self.unc_model.get_uncertain_action(state, policy = self.collection_policy)
					self.uncertain_actions += 1
				else:
					action = self.policy.sample_action(state.reshape(1,-1)).reshape(-1)
					self.policy_actions += 1
			#UP: Pick based on combination of uncertainty and policy depending on threshold
			elif 'up' in self.collection_policy:
				current_unc = self.unc_model.get_uncertainty(torch.from_numpy(state.reshape(1,-1)).float().to(self.unc_model.device)).item()
				if current_unc <= self.up_threshold:
					action = self.policy.sample_action(state.reshape(1,-1)).reshape(-1)
					self.policy_actions += 1
				else:
					action = self.unc_model.get_uncertain_action(state, policy = self.collection_policy)
					self.uncertain_actions += 1
			elif 'pn' in self.collection_policy:
				if np.random.rand() < self.collection_epsilon:
					state_r = state.reshape(1,-1)
					state_r = np.repeat(state_r, self.unc_model.width, axis = 0)
					noise = np.random.randn(self.unc_model.width, self.unc_model.action_dim)*self.pn_noise
					sampled_actions = torch.from_numpy(self.policy.sample_action(state_r) + noise).float().clip(-self.unc_model.max_action, self.unc_model.max_action).to(self.unc_model.device)
					state_r = torch.from_numpy(state_r).float().to(self.unc_model.device)
					unc = self.unc_model.get_uncertainty(state_r, sampled_actions).cpu()

					action = sampled_actions[np.argmax(unc)].cpu().numpy().reshape(-1)
					self.uncertain_actions += 1
				else:
					action = self.policy.sample_action(state.reshape(1,-1)).reshape(-1)
					self.policy_actions += 1

			else:
				raise ValueError
			
			next_state, reward, done, _ = self.env.step(action)
			episode_terminal = done

			to_collect -= 1
			n_collection += 1

			#Already collected enough -> STOP
			if to_collect <= 0:
				episode_terminal = True

			#Early Stopping -> CHECK Threshold -> STOP
			if self.early_stop:
				current_unc = self.unc_model.get_uncertainty(torch.from_numpy(state.reshape(1,-1)).float().to(self.unc_model.device)).item()
				if current_unc < self.early_stop_thershold:
					print("Early Stop!")
					episode_terminal = True  

			#Trajectory Limit reached -> STOP
			if n_collection >= self.trajectory_size:
				episode_terminal = True

			self.collection_buffer.append(state, action, reward, done, episode_terminal)
			state = next_state

			if self.current_buffer is not None and self.collection_buffer.size() >= 128:
				current_batch = self.current_buffer.sample(batch_size = 256 + 128)
				collection_batch = self.collection_buffer.sample(batch_size = 128)

				mixed_batch = d3rlpy.dataset.TransitionMiniBatch(collection_batch.transitions + current_batch.transitions)
				self.unc_model.update(mixed_batch)
				self.n_train += 1

			if episode_terminal:
				break

		return n_collection


	def collect(self, states: torch.Tensor, current_dataset: d3rlpy.dataset.MDPDataset = None):
		self.current_buffer = d3rlpy.online.buffers.ReplayBuffer(int(5e6), self.env, current_dataset.episodes) if current_dataset else None
		self.collection_buffer = d3rlpy.online.buffers.ReplayBuffer(self.num_samples, self.env)
		
		#Reset Counters
		self.uncertain_actions = 0
		self.policy_actions = 0
		self.random_actions = 0
		self.n_train = 0

		#REPORT UNCERTAINTY METRIC STATISTICS FOR states
		if current_dataset:
			uncertainties = self.unc_model.get_uncertainty_batched(torch.from_numpy(current_dataset.observations).float())
		uncertainties = self.unc_model.get_uncertainty_batched(states)

		#Set the early stop thresold to 99 percentile
		self.up_threshold = np.percentile(uncertainties, self.up_percentile)
		self.early_stop_thershold = np.percentile(uncertainties, self.es_percentile)

		print(f"UPThresh: {self.up_threshold}  ESThresh: {self.early_stop_thershold}")
		

		#Collection Counters
		to_collect = self.num_samples
		n_collection = 0

		#Collection Arrays

		while to_collect>0:
			#Sample Candidate States
			candidate_states = states[np.random.permutation(states.shape[0])[:self.candidate_size]].to(self.unc_model.device)

			#ACTIVE: Pick the most uncertain states
			if self.collection_method == 'active':
				#Do epsilon greedy
				if np.random.rand() < self.active_epsilon:
					uncertainty = self.unc_model.get_uncertainty(candidate_states)
					selected_indices = torch.argsort(uncertainty, descending=True)[:self.selection_size]
					selected_states = candidate_states[selected_indices]
					self.trajectory_size = self.config_trajectory_size
				else:
					if np.random.rand() < 0.5:
						selected_states = candidate_states[:self.selection_size]
						self.trajectory_size = self.config_trajectory_size
					else:
						selected_states = torch.from_numpy(np.stack([self.env.reset() for _ in range(self.selection_size)])).float()
						self.trajectory_size = 1000

			#RANDOM: Pick the first states
			elif self.collection_method == 'random':
				selected_states = candidate_states[:self.selection_size]
				self.trajectory_size = self.config_trajectory_size

			#INITIAL: Pick the initial states
			elif self.collection_method == 'initial':
				selected_states = torch.from_numpy(np.stack([self.env.reset() for _ in range(self.selection_size)])).float()
			else:
				raise ValueError
			
			for i in range(self.selection_size):
				state = selected_states[i].cpu().numpy()

				n_col = self._collect_trajectory(state, to_collect)
				to_collect -= n_col
				n_collection += n_col



				print(f"[ACTIVE UTIL] Collected: {n_collection} | RA: {self.random_actions} | PA: {self.policy_actions} | UA: {self.uncertain_actions} | Trained: {self.n_train}")

				if to_collect<=0:
					break
			

		print(f"[ACTIVE UTIL] Total Collection: {n_collection} | RA: {self.random_actions} | PA: {self.policy_actions} | UA: {self.uncertain_actions} | Trained: {self.n_train}")

		#Create d3rlpy dataset
		collected_dataset = self.collection_buffer.to_mdp_dataset()
		return collected_dataset




	

