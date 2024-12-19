from utils_common import *
from utils_uncertainty import UncertaintyModel

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default="antmaze-umaze-v0")
    parser.add_argument("--data", type=str, default = "")

    #TD3BC Parameters
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--target_smoothing_sigma", default=0.2, type=float)
    parser.add_argument("--target_smoothing_clip", default=0.5, type=float)
    parser.add_argument("--alpha", default=2.5, type=float)
    parser.add_argument("--update_actor_interval", default=2, type=int)
    
    #Uncertainty Model Parameters
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--uncertainty_lr", default=3e-4, type=float)
    parser.add_argument("--num_models", default=5, type=int)

    #Training Parameters
    parser.add_argument("--n_steps", default=int(1e6), type=int)
    parser.add_argument("--n_steps_per_epoch", default=50000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    #Extra arguments
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--show_progress", action="store_true", default=False)


    args = parser.parse_args()
    return args


def main(args):
    #Directories
    logdir = f"TD3BC_OFFLINE/{args.env}/"
    experiment_name = f"native" if args.data == "" else args.data

    #Seed Everything
    seed_everything(args.seed)

    #Get Environment and Dataset
    dataset, env = d3rlpy.datasets.get_d4rl(args.env)
    env.seed(args.seed)

    if args.data != "":
        dataset_path = f"DATASETS/{args.env[:-3]}/{args.data}.h5"
        dataset = d3rlpy.dataset.MDPDataset.load(dataset_path)
        print(f"[TD3BC OFFLINE] Loaded Custom Dataset: {args.data}")

    #Fix device
    device = torch.device(f"cuda:{args.device}")

    #Setup policy and metrics
    policy = d3rlpy.algos.TD3PlusBC(actor_learning_rate=args.actor_lr,
                                    critic_learning_rate=args.critic_lr,
                                    batch_size=args.batch_size,
                                    target_smoothing_sigma=args.target_smoothing_sigma,
                                    target_smoothing_clip=args.target_smoothing_clip,
                                    alpha=args.alpha,
                                    update_actor_interval=args.update_actor_interval,
                                    scaler="standard",
                                    use_gpu=args.device)
    
    policy.build_with_dataset(dataset)
    metrics = {"rewards_raw": d3rlpy.metrics.scorer.evaluate_on_environment(env, n_trials=10)}

    #Setup Uncertainty Model
    uncertainty_model = UncertaintyModel(state_dim = env.observation_space.shape[0], 
                                         action_dim = env.action_space.shape[0],
                                         max_action = env.action_space.high[0],
                                         hidden_dim = args.hidden_dim,
                                         lr=args.uncertainty_lr,
                                         num_models=args.num_models,
                                         device=device)
    
    #Train Policy
    print("======================================== TRAINING POLICY ========================================")
    policy.fit(dataset.episodes,
               n_steps=args.n_steps,
               n_steps_per_epoch=args.n_steps_per_epoch,
               eval_episodes=dataset.episodes,
               scorers=metrics,
               save_interval=10,
               logdir=logdir,
               experiment_name=experiment_name,
               show_progress=args.show_progress,
               with_timestamp=False)
    
    #Save Policy
    policy.save_model(f"{logdir}/{experiment_name}/final.pt")


    #Train Uncertainty Model
    print("======================================== TRAINING UNCERTAINTY MODEL ========================================")
    replay_buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=int(5e6),
                                                       env=env,
                                                       episodes=dataset.episodes)
    
    uncertainty_model.fit(replay_buffer,
                          batch_size=args.batch_size,
                          n_steps=args.n_steps)
    
    #Save Uncertainty Model
    uncertainty_model.save(f"{logdir}/{experiment_name}/uncertainty")


if __name__ == "__main__":
    args = parse_args()
    main(args)


