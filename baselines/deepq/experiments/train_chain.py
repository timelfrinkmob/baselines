import gym

from baselines import deepq
import argparse
from baselines.common.misc_util import (
    set_global_seeds
)
import baselines.deepq.experiments.gym_chain



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy", type=int, default=0, help="Noisy?")
    parser.add_argument("--greedy", type=int, default=0, help="Greedy?")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap?")
    parser.add_argument("--seed", type=int, default=0, help="seed?")
    parser.add_argument("--n", type=int, default=10, help="chain length?")
    parser.add_argument("--episodes", type=int, default=2000, help="nr episodes?")
    args = parser.parse_args()


    env = gym.make('Chainbla-v0')
    env.__init__(n=args.n)
    set_global_seeds(args.seed)
    env.seed(args.seed)

    nb_steps = (args.n + 9) * (args.episodes-1)

    # Enabling layer_norm here is import for parameter space noise!

    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=nb_steps,
        buffer_size=int(nb_steps/2),
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=1,
        noisy = args.noisy,
        greedy = args.greedy,
        bootstrap = args.bootstrap,
        seed = args.seed,
        env_name = 'Chain' + str(args.n)
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("mountaincar_model.pkl")


if __name__ == '__main__':
    main()
