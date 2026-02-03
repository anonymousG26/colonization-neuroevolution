
import neat
import networkx as nx
import utils
#from flax import linen as nn
import multiprocessing
import gc
from reporter import CustomReporter
from parallel import ParallelEvaluator
import warnings
from env import MyEnv
import numpy as np
from nn import HebbianRecurrentNetwork
from genome import HebbianRecurrentGenome
import random
import visualize
from neat.graphs import creates_cycle
from reproduction import CustomReproduction
from stagnation import CustomStagnation

warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')

'''
V5: lr and edecay per connection. same as v4 apart from using genome_v16
'''

def log_complexity(genome, config, wandb_run, save=False, winner=False, gen=""):
        pruned = genome.get_pruned_copy(config.genome_config)

        G = utils.make_graph(pruned, config, 149)

        if save:
            run_name = wandb_run.id
            filename = run_name + "_gen" + str(gen) + "_Graph.adjlist"
            nx.write_adjlist(G, filename)
            wandb_run.save(filename)
            #filename = run_name + "_gen" + str(gen) + "_Graph.png"
            #utils.show_layered_graph(G, save=True, filename=filename)
            #wandb_run.save(filename)

        ns = utils.get_ns(G, 149)
        nc, mod, glob_eff, sp = utils.get_nc(G)
        num_nodes = G.number_of_nodes()
        num_conn = G.number_of_edges()

        self_loops, m22, m33, m38 = utils.motifs(G)

        #cn = utils.cn(G)

        return ns, nc, num_nodes, num_conn, mod, glob_eff, self_loops, m22, m33, m38, sp

def evaluate(genome, config, learn=True, n_trials=100, skip_evaluated=True, n_seasons=4,
                seed=1361, energy_costs=False, randcol=True, n_eval_trials=50, gen=1):

    #worst_fitness = 1000
    fitnesses = []
    nets = []
    net = None
    eval_seed = seed+5
    use_weights=True

    for i in range(10):
        fitness, net = evaluate_(genome, config, learn, n_trials, skip_evaluated, n_seasons,
                seed+i, energy_costs, randcol, n_eval_trials=50, gen=gen, use_weights=use_weights)
        fitnesses.append(fitness)
        nets.append(net)
    idx = np.argmin(fitnesses)
    fitness = np.mean(fitnesses)

    performances = []
    
    for i in range(1):
        performance, net = evaluate_(genome, config, learn, n_trials, skip_evaluated, n_seasons,
                    123456+i, energy_costs, randcol, n_eval_trials=50, gen=1, use_weights=True, net=None)
        performances.append(performance)

    return fitness, nets[idx], np.mean(performances)

def evaluate_(genome, config, learn=True, n_trials=100, skip_evaluated=True, n_seasons=4, 
                seed=1361, energy_costs=False, randcol=True, n_eval_trials=50, gen=1, use_weights=True, net=None):

    if net is None:
        net = HebbianRecurrentNetwork.create(genome, config, use_weights)
    rng = np.random.default_rng(seed)

    n_train_eps = n_trials - n_eval_trials
    seeds = rng.integers(1, 90000, size=n_trials)

    col_seed = 1361
    if randcol:
        col_seed = seed

    if energy_costs:
        pruned = genome.get_pruned_copy(config.genome_config)
        G = utils.make_graph(pruned, config, 149)
        ns = utils.get_ns(G, 149)
        energy_coef = 0.01/380
        env = MyEnv(n_seasons=n_seasons, col_dist=True, v=3, size=10, col_seed=seed, ns=ns, energy_coef=energy_coef, randcol=randcol)

    else:
        env = MyEnv(n_seasons=n_seasons, col_dist=True, v=3, size=10, col_seed=seed, randcol=randcol)
    
    all_episode_rewards = []
    reward = None
    mod_reward = None
    learning = False
    action = None

    for i in range(n_trials):  
        if learn and reward and not learning:
            net.avg_reward = np.mean(episode_rewards)
            learning = True
        episode_rewards = []
        done = False
        observation, info = env.reset(seed=int(seeds[i]))
        net.reset()
        while not done:
            if learning:
                output = net.update_activate(observation, mod_reward, last_action=action)
            else:
                output = net.activate(observation)
            if i >= n_train_eps:
                action = np.argmax(output)
            else:
                probs = np.exp(output) / np.sum(np.exp(output))
                action = rng.choice(len(probs), p=probs)
            observation, reward, done, truncated, info = env.step(action)
            mod_reward = reward
            episode_rewards.append(reward)
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards[-n_eval_trials:])
    env.close()
    return mean_episode_reward, net

def run(config_file='neat_config', parallel=True, wandb_run=None, restore=False, checkpoint_file="neat-checkpoint-0", n_seasons=4, 
        seed=1361, n_gens=1, energy_costs=True, n_procs=3, n_trials=100, learn=True, randcol=False, unpredictable=False, transition=None):

    print("run transition: ", transition)
    if transition is not None:
        n_seasons = 1
    if unpredictable and transition is None:
        randcol = True
    else:
        randcol = False

    if restore:
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)

    else:

        # Load configuration.
        config = neat.Config(HebbianRecurrentGenome, CustomReproduction,
                            neat.DefaultSpeciesSet, CustomStagnation,
                            config_file)
    
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config,seed=seed)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = CustomReporter()
    p.add_reporter(stats)

    if wandb_run:
        filename_prefix = 'checkpoint-' + wandb_run.name + '-'
    else:
        filename_prefix = 'neat-checkpoint-'
    checkpointer = neat.Checkpointer(generation_interval=50, time_interval_seconds=None, filename_prefix=filename_prefix)
    p.add_reporter(checkpointer)

    if parallel:
        pe = ParallelEvaluator(n_procs, evaluate, timeout=600, n_seasons=n_seasons, seed=seed, energy_costs=energy_costs, n_trials=n_trials, learn=learn, randcol=randcol)
        gen_start = p.generation
        print("gen start: ", gen_start)

    best_fitness = 0
    for generation in range(gen_start, n_gens):

        if parallel:
            try:
                if generation == 200 and transition == "sudden":
                    pe.n_seasons = 4
                    if unpredictable:
                        pe.randcol=True
                elif generation % 100 == 0 and generation > 0 and transition == "gradual" and pe.n_seasons < 4:
                    pe.n_seasons += 1
                    if unpredictable and not pe.randcol:
                        pe.randcol=True
                print("n_seasons: ", pe.n_seasons)
                print("randcol: ", pe.randcol)
                if generation%50==0:
                    pe.gen = generation
                    pe.seed = seed+generation
                gen_best = p.run(pe.evaluate, 1)
                w_min, w_max, w_mean, w_stdev = gen_best.net.get_weight_stats()
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if isinstance(e, multiprocessing.context.TimeoutError):
                    print("Timeout occurred during evaluation.")
                raise
        else: # non parallel version incomplete: Doesn't parse eval genomes arguments
            try:
                print("function not implemented")
                #gen_best = p.run(eval_genomes, 1)
            except Exception as e:
                print(f"Error during evaluation: {e}")      
        if wandb_run:
            gen_mean = stats.get_fitness_mean()
            ns, nc, num_nodes, num_conn, mod, glob_eff, self_loops, m22, m33, m38, sp = log_complexity(gen_best, p.config, wandb_run)
            if energy_costs:
                gen_best_task_perf = gen_best.fitness + (ns/380)
                unseen_perf = gen_best.mean_perf + (ns/380)
            else: 
                gen_best_task_perf = gen_best.fitness + 1
                unseen_perf = gen_best.mean_perf + 1
            non_input_nodes = ns - num_conn
            wandb_run.log({"gen": p.generation-1, "gen_best_fitness": gen_best.fitness, "gen_mean_fitness": gen_mean, 
                "gen_best_ns": ns, "gen_best_nc": nc, "gen_best_num_nodes": num_nodes, "gen_best_num_conn": num_conn,
                "gen_best_task_perf": gen_best_task_perf, "modularity": mod, "global efficiency": glob_eff,
                "non_input_nodes": non_input_nodes, "self_loops": self_loops, "m22": m22, "m33": m33, "m38": m38, "sp":sp,
                "min weight": w_min, "max weight": w_max, "mean weight": w_mean, "stdev weight":w_stdev, "unseen perf": unseen_perf})
        gc.collect()
    winner = gen_best

    if wandb_run:
        ns, nc, num_nodes, num_conn, mod, glob_eff, self_loops, m22, m33, m38, sp = log_complexity(winner, p.config, wandb_run, save=False, winner=True, gen = p.generation-1)
        wandb_run.log({"winner_ns_v2": ns, "winner_nc_v2": nc})
        visualize.draw_net(p.config, winner, filename=wandb_run.id+"_winnerNN", fmt='png', prune_unused=True)
        wandb_run.save(wandb_run.id+"_winnerNN.png")
    
    #visualize.draw_net(config, winner, view=True)
    #visualize.draw_net(config, winner, filename="winnerNN", fmt='png', prune_unused=True)

    return winner.fitness

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')

    # seed argument
    parser.add_argument('--seed', type=int, default=1361,
                        help='seed value')

    # n_seasons argument
    #parser.add_argument('--n_seasons', type=int, default=4,
    #                    help='Number of seasons in environment')

    # n_gens argument
    parser.add_argument('--n_gens', type=int, default=1,
                        help='Number of Generations')


    # unpredictable argument
    parser.add_argument('--unpredictable', action='store_true', help="Make seasonal changes unpredictable (default: predictable / False)")

    # transition argument
    parser.add_argument('--transition', type=str, default=None,
                        choices=['sudden', 'gradual'],
                        help='Transition type: "sudden" or "gradual"; default is None (no transition)')


    # EC argument
    #parser.add_argument('--no_EC', action='store_false', help="remove energy costs that scale with ANN size")

    # n_procs argument
    parser.add_argument('--n_procs', type=int, default=3, 
                        help='Number of parallel processes for evaluating population fitness (default: 3)')


    args = parser.parse_args()
    print("Argument values:")
    print("seed: ", args.seed)
    print("Generations: ", args.n_gens)
    print("Unpredictable changes: ", args.unpredictable)
    print("Transition: ", args.transition)
    #print("Energy Costs: ", args.no_EC)
                        
    run(seed=args.seed, n_gens=args.n_gens, n_procs=args.n_procs, unpredictable=args.unpredictable, transition=args.transition)
