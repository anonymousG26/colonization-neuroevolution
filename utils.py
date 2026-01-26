import networkx as nx
import matplotlib.pyplot as plt
#from forward.models import ForWard as fwd
import numpy as np
import copy
from typing import Callable

def mod_eff(G):
    Gu = G.to_undirected()
    try:
        mods = []
        seeds = [120,121,122,123,124,125,126,127,128,129]
        for seed in seeds:
            mod = nx.community.modularity(Gu, nx.community.louvain_communities(Gu, seed=seed, weight=None), weight=None)
            mods.append(mod)
        mod = np.mean(mods)
    except:
        mod = 1
    glob_eff = nx.global_efficiency(Gu)
    return mod, glob_eff

def get_sp(Graph):
    G = Graph.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    # Strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    s = len(sccs)

    # Count cycles in SCCs and nodes in SCCs
    c = 0
    n_in_scc = 0
    for scc in sccs:
        if len(scc) < 2:
            continue
        subG = G.subgraph(scc)
        cycles_in_scc = list(nx.simple_cycles(subG))
        c += len(cycles_in_scc)
        n_in_scc += subG.number_of_nodes()

    # Add feedforward triads (030T)
    triads = nx.triadic_census(G)
    t030T = triads.get('030T', 0)

    c = c + (0.5 * t030T)

    t = G.number_of_nodes()
    sp = (c / s) * (n_in_scc / t) if s > 0 else 0
    return sp

def get_nc(G):
    # Iterate over the nodes and remove those with no incoming or outgoing edges
    nodes_to_remove = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    # Remove the nodes from the graph
    G.remove_nodes_from(nodes_to_remove)
    mod, glob_eff = mod_eff(G)
    #sp = get_sp(G)
    nc = min(mod,glob_eff)/max(mod,glob_eff)
    #nc2 = min(sp,glob_eff)/max(sp,glob_eff)
    return nc, mod, glob_eff, -1

def get_smallword(genome, config, mod, num_inputs=29):
    pruned = genome.get_pruned_copy(config.genome_config)
    G = make_graph(pruned, config, num_inputs)
    # Iterate over the nodes and remove those with no incoming or outgoing edges
    nodes_to_remove = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    # Remove the nodes from the graph
    G.remove_nodes_from(nodes_to_remove)
    Gu = G.to_undirected()
    try:
        #if nx.is_connected(Gu) and mod > 0:
        if mod > 0:
            omega = nx.omega(Gu, niter=20, nrand=30, seed=1361)
            sigma = nx.sigma(Gu, niter=20, nrand=30, seed=1361)
        else:
            omega = float('nan')
            sigma = float('nan')
    except:
        omega = float('nan')
        sigma = float('nan')
    return omega, sigma

def get_ns(G, input_len=243):
    return G.number_of_nodes()-input_len+G.number_of_edges()

def make_graph(genome, config, input_size):
    G = nx.DiGraph()

    for node in genome.nodes.keys():
        bias = genome.nodes[node].bias
        activation = genome.nodes[node].activation
        G.add_node(node, bias=bias, activation=activation)
    for k in genome.connections.keys():
        enabled = genome.connections[k].enabled
        if enabled:
            weight = genome.connections[k].weight
            G.add_edge(k[0], k[1], weight=weight)

    # ensure correct input size
    for n in range(-1*input_size,0):
        if n not in (G.nodes):
            G.add_node(n)

    return G

def cn_mu(G):
    weights = [d['weight'] for (_, _, d) in G.edges(data=True) if 'weight' in d]
    mu = np.mean(weights)
    mu2 = np.mean(np.square(weights))
    return mu, mu2

def cn(Graph):
    # Copy and remove self-loops
    G = Graph.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    m1 = G.number_of_edges()
    m22 = nx.reciprocity(G) * m1 / 2
    n = G.number_of_nodes()
    #mu, mu2 = cn_mu(G)
    mu = 1
    mu2 = 1
    cn1 = ((n+1)/48) * ((mu2*m1) + (2*(mu**2)*m22))

    triads = nx.triadic_census(G)
    m33 = triads['030T']
    m38 = triads['030C']

    cn2 = ((n+1)/32)*(mu**3)*(m33+m38)

    return (cn1 + cn2)/((n+1)*m1)

def motifs(Graph):
    self_loops = nx.number_of_selfloops(Graph)
    G = Graph.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    m1 = G.number_of_edges()
    m22 = nx.reciprocity(G) * m1 / 2
    triads = nx.triadic_census(G)
    m33 = triads['030T']
    m38 = triads['030C']
    return self_loops, m22, m33, m38

def generate_neat_config(
    filename="neat_config",
    fitness_criterion = "max",
    fitness_threshold = 100,
    pop_size = 120,
    reset_on_extinction = "false",
    no_fitness_termination = "true",

    # Node activation options
    activation_default = "sigmoid",
    activation_mutate_rate = 0.0,
    activation_options = "sigmoid clamped tanh",

    # Node aggregation options
    aggregation_default = "sum",
    aggregation_mutate_rate = 0.0,
    aggregation_options = "sum",

    # Node bias options
    bias_init_mean = 0.0,
    bias_init_stdev = 0.0,
    bias_max_value = 10.0,
    bias_min_value = -10.0,
    bias_mutate_power = 0.1,
    bias_mutate_rate = 0.1,
    bias_replace_rate = 0,

    # Genome compatibility options
    compatibility_disjoint_coefficient = 1,
    compatibility_weight_coefficient = 1,

    # Connection add/remove rates
    conn_add_prob = 0.5,
    conn_delete_prob = 0.5,

    # Connection enable options
    enabled_default = "true",
    enabled_mutate_rate = 0,

    feed_forward = "false",
    initial_connection = "partial_nodirect 0.5",

    # Node add/remove rates
    node_add_prob = 0.5,
    node_delete_prob = 0.5,

    # Network parameters
    num_hidden = 0,
    num_inputs = 149,
    num_outputs = 5,

    # Node response options
    response_init_mean = 1.0,
    response_init_stdev = 0.0,
    response_max_value = 30.0,
    response_min_value = -30.0,
    response_mutate_power = 0.0,
    response_mutate_rate = 0.0,
    response_replace_rate = 0.0,

    # Connection weight options
    weight_init_type = "uniform",
    weight_init_mean = 0.0,
    weight_init_stdev = 1.0,
    weight_max_value = 1.0,
    weight_min_value = -1.0,
    weight_mutate_power = 0.8,
    weight_mutate_rate = 0.0,
    weight_replace_rate = 0.0,

    # Learning rate (lr) config for each connection
    lr_init_mean = 0.01,
    lr_init_stdev = 0.005,
    lr_max_value = 10.0,
    lr_min_value = 0.0001,
    lr_mutate_power = 0.01,
    lr_mutate_rate = 0.2,
    lr_replace_rate = 0.0,

    edecay_init_mean = 0.5,
    edecay_init_stdev = 0.12,
    edecay_max_value = 1.0,
    edecay_min_value = 0.01,
    edecay_mutate_power = 0.05,
    edecay_mutate_rate = 0.2,
    edecay_replace_rate = 0.0,

    decay_init_mean = 0.0,
    decay_init_stdev = 0.00001,
    decay_max_value = 0.0001,
    decay_min_value = 0.0,
    decay_mutate_power = 0.00001,
    decay_mutate_rate = 0.0,
    decay_replace_rate = 0.0,


    structural_mutation_surer = "true",
    single_structural_mutation = "true",

    compatibility_threshold = 6,

    species_fitness_func = "max",
    max_stagnation = 20,
    species_elitism = 1,

    elitism = 2,
    survival_threshold = 0.5

):
    with open(filename, "w") as config_file:
        config_file.write(f"[NEAT]\n")
        config_file.write(f"fitness_criterion = {fitness_criterion}\n")
        config_file.write(f"fitness_threshold = {fitness_threshold}\n")
        config_file.write(f"pop_size = {pop_size}\n")
        config_file.write(f"reset_on_extinction = {reset_on_extinction}\n")
        config_file.write(f"no_fitness_termination = {no_fitness_termination}\n\n")

        config_file.write(f"[HebbianRecurrentGenome]\n")
        config_file.write(f"# Node activation options\n")
        config_file.write(f"activation_default = {activation_default}\n")
        config_file.write(f"activation_mutate_rate = {activation_mutate_rate}\n")
        config_file.write(f"activation_options = {activation_options}\n\n")

        config_file.write(f"# Node aggregation options\n")
        config_file.write(f"aggregation_default = {aggregation_default}\n")
        config_file.write(f"aggregation_mutate_rate = {aggregation_mutate_rate}\n")
        config_file.write(f"aggregation_options = {aggregation_default}\n\n")

        config_file.write(f"# Node bias options\n")
        config_file.write(f"bias_init_mean = {bias_init_mean}\n")
        config_file.write(f"bias_init_stdev = {bias_init_stdev}\n")
        config_file.write(f"bias_max_value = {bias_max_value}\n")
        config_file.write(f"bias_min_value = {bias_min_value}\n")
        config_file.write(f"bias_mutate_power = {bias_mutate_power}\n")
        config_file.write(f"bias_mutate_rate = {bias_mutate_rate}\n")
        config_file.write(f"bias_replace_rate = {bias_replace_rate}\n\n")

        config_file.write(f"# Genome compatibility options\n")
        config_file.write(f"compatibility_disjoint_coefficient = {compatibility_disjoint_coefficient}\n")
        config_file.write(f"compatibility_weight_coefficient = {compatibility_weight_coefficient}\n\n")

        config_file.write(f"# Connection add/remove rates\n")
        config_file.write(f"conn_add_prob = {conn_add_prob}\n")
        config_file.write(f"conn_delete_prob = {conn_delete_prob}\n\n")

        config_file.write(f"# Connection enable options\n")
        config_file.write(f"enabled_default = {enabled_default}\n")
        config_file.write(f"enabled_mutate_rate = {enabled_mutate_rate}\n\n")

        config_file.write(f"feed_forward = {feed_forward}\n")
        config_file.write(f"initial_connection = {initial_connection}\n\n")

        config_file.write(f"# Node add/remove rates\n")
        config_file.write(f"node_add_prob = {node_add_prob}\n")
        config_file.write(f"node_delete_prob = {node_delete_prob}\n\n")

        config_file.write(f"# Network parameters\n")
        config_file.write(f"num_hidden = {num_hidden}\n")
        config_file.write(f"num_inputs = {num_inputs}\n")
        config_file.write(f"num_outputs = {num_outputs}\n\n")

        config_file.write(f"# Node response options\n")
        config_file.write(f"response_init_mean = {response_init_mean}\n")
        config_file.write(f"response_init_stdev = {response_init_stdev}\n")
        config_file.write(f"response_max_value = {response_max_value}\n")
        config_file.write(f"response_min_value = {response_min_value}\n")
        config_file.write(f"response_mutate_power = {response_mutate_power}\n")
        config_file.write(f"response_mutate_rate = {response_mutate_rate}\n")
        config_file.write(f"response_replace_rate = {response_replace_rate}\n\n")

        config_file.write(f"# Connection weight options\n")
        config_file.write(f"weight_init_type = {weight_init_type}\n")
        config_file.write(f"weight_init_mean = {weight_init_mean}\n")
        config_file.write(f"weight_init_stdev = {weight_init_stdev}\n")
        config_file.write(f"weight_max_value = {weight_max_value}\n")
        config_file.write(f"weight_min_value = {weight_min_value}\n")
        config_file.write(f"weight_mutate_power = {weight_mutate_power}\n")
        config_file.write(f"weight_mutate_rate = {weight_mutate_rate}\n")
        config_file.write(f"weight_replace_rate = {weight_replace_rate}\n\n")

        config_file.write(f"# Learning rate (lr) config for each connection\n")
        config_file.write(f"lr_init_mean = {lr_init_mean}\n")
        config_file.write(f"lr_init_stdev = {lr_init_stdev}\n")
        config_file.write(f"lr_max_value = {lr_max_value}\n")
        config_file.write(f"lr_min_value = {lr_min_value}\n")
        config_file.write(f"lr_mutate_power = {lr_mutate_power}\n")
        config_file.write(f"lr_mutate_rate = {lr_mutate_rate}\n")
        config_file.write(f"lr_replace_rate = {lr_replace_rate}\n")

        config_file.write(f"edecay_init_mean = {edecay_init_mean}\n")
        config_file.write(f"edecay_init_stdev = {edecay_init_stdev}\n")
        config_file.write(f"edecay_max_value = {edecay_max_value}\n")
        config_file.write(f"edecay_min_value = {edecay_min_value}\n")
        config_file.write(f"edecay_mutate_power = {edecay_mutate_power}\n")
        config_file.write(f"edecay_mutate_rate = {edecay_mutate_rate}\n")
        config_file.write(f"edecay_replace_rate = {edecay_replace_rate}\n")

        config_file.write(f"decay_init_mean = {decay_init_mean}\n")
        config_file.write(f"decay_init_stdev = {decay_init_stdev}\n")
        config_file.write(f"decay_max_value = {decay_max_value}\n")
        config_file.write(f"decay_min_value = {decay_min_value}\n")
        config_file.write(f"decay_mutate_power = {decay_mutate_power}\n")
        config_file.write(f"decay_mutate_rate = {decay_mutate_rate}\n")
        config_file.write(f"decay_replace_rate = {decay_replace_rate}\n")

        config_file.write(f"structural_mutation_surer = {structural_mutation_surer}\n")
        config_file.write(f"single_structural_mutation = {single_structural_mutation}\n\n")

        config_file.write(f"[DefaultSpeciesSet]\n")
        config_file.write(f"compatibility_threshold = {compatibility_threshold}\n\n")

        config_file.write(f"[DefaultStagnation]\n")
        config_file.write(f"species_fitness_func = {species_fitness_func}\n")
        config_file.write(f"max_stagnation = {max_stagnation}\n")
        config_file.write(f"species_elitism = {species_elitism}\n\n")

        config_file.write(f"[CustomReproduction]\n")
        config_file.write(f"elitism = {elitism}\n")
        config_file.write(f"survival_threshold = {survival_threshold}\n")

    print(f"Configuration file '{filename}' has been generated.")
    return filename
