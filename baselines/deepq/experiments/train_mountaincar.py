import gym

from baselines import deepq
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MountainCar-v0", help="name of the game")
    parser.add_argument("--noisy", type=int, default=0, help="Noisy?")
    parser.add_argument("--greedy", type=int, default=0, help="Greedy?")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap?")
    args = parser.parse_args()


    env = gym.make(args.env,bootstrap = args.bootstrap)

    # Enabling layer_norm here is import for parameter space noise!

    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=1,
        noisy = args.noisy,
        greedy = args.greedy,
        bootstrap = args.bootstrap
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("mountaincar_model.pkl")


if __name__ == '__main__':
    main()
