from neat.graphs import required_for_output
import copy
import numpy as np
import random
'''
V3: reset eligibility traces, lr and edecay per connection, tau=0.01
'''

class HebbianRecurrentNetwork(object):
    def __init__(self, inputs, outputs, node_evals, connections, connections_lr, connections_edecay, connections_decay):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.connections = copy.deepcopy(connections)
        self.connections_lr = connections_lr
        self.connections_edecay = connections_edecay
        self.connections_decay = connections_decay
        self.avg_reward = 0.0
        self.tau = 0.01
        self.hvalues = None

        self.values = [{}, {}]
        for v in self.values:
            for k in [*inputs, *outputs]:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0
        self.active = 0

        # Initialize eligibility traces for each connection
        self.eligibilities = {}
        for post_node, _, _, _, _, links in self.node_evals:
            for pre_node, _ in links:
                self.eligibilities[(pre_node, post_node)] = 0.0

    def get_weight_stats(self):
        weights = np.array(list(self.connections.values()))
        return weights.min(), weights.max(), weights.mean(), weights.std()

    def get_activation_stats(self):
        return self.hvalues.min(), self.hvalues.max(), self.hvalues.mean(), self.hvalues.std()

    def update_reward_baseline(self, reward):
        self.avg_reward = (1 - self.tau) * self.avg_reward + self.tau * reward

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0
        # reset eligibility traces
        for post_node, _, _, _, _, links in self.node_evals:
            for pre_node, _ in links:
                self.eligibilities[(pre_node, post_node)] = 0.0
        # reset hvalues
        self.hvalues = None

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        #conns = self.connections
        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = [ivalues[i] * self.connections[(i, node)] for i, _ in links]
            s = aggregation(node_inputs)
            ovalues[node] = activation(bias + response * s)

        self.hvalues = ovalues.copy()

        return [ovalues[i] for i in self.output_nodes]

    def hebbian_update(self, reward, learning_rate=0.01, dt=1.0, tau_e=10.0):
        # Update reward baseline
        self.update_reward_baseline(reward)
        modulation = reward - self.avg_reward
        # Use the values from the previous forward pass
        ovalues = self.values[1 - self.active]  # these are the outputs just computed

        for post_node, _, _, _, _, links in self.node_evals:
            a_j = ovalues[post_node]  # activation of the post-synaptic node
            for pre_node, _ in links:
                a_i = ovalues[pre_node]  # activation of the pre-synaptic node
                key = (pre_node, post_node)
                hebb = a_i * a_j
                # Update eligibility trace (discretized decay)
                self.eligibilities[key] = (1 - (dt / tau_e)) * self.eligibilities[key] + (dt * hebb)
                # Update weight using eligibility and modulation
                delta_w = learning_rate * modulation * self.eligibilities[key]
                #delta_w = modulation * self.eligibilities[key]
                self.connections[(pre_node,post_node)] += delta_w


    def update_activate(self, inputs, reward, last_action=None):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        if self.hvalues is None:
            return self.activate(inputs)

        conns = self.connections
        lrs = self.connections_lr
        edecays = self.connections_edecay
        eligs = self.eligibilities
        hvals = self.hvalues

        self.update_reward_baseline(reward)
        modulation = reward - self.avg_reward

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active
        #hvalues = ivalues.copy()

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        if last_action:
            for i in self.output_nodes:
                hvals[i] = 0
            hvals[last_action] = 1
            

        for node, activation, aggregation, bias, response, links in self.node_evals:
            #node_inputs = [ivalues[i] * w for i, w in links]
            post_node = node
            #if self.hvalues:
            a_j = hvals[post_node]  # activation of the post-synaptic node
            node_inputs = []
            for pre_node, _ in links:
                #if self.hvalues:
                a_i = hvals[pre_node]  # activation of the pre-synaptic node
                key = (pre_node, post_node)
                hebb = a_i * a_j
                # Update eligibility trace (discretized decay)
                decay_rate = edecays[key]
                eligs[key] = ((1 - decay_rate) * eligs[key]) + hebb
                # Update weight using eligibility and modulation
                learning_rate = lrs[key]
                delta_w = learning_rate * modulation * eligs[key]
                #delta_w = modulation * self.eligibilities[key]
                conns[(pre_node,post_node)] += delta_w
                node_inputs.append(ivalues[pre_node] * conns[(pre_node, post_node)])
            s = aggregation(node_inputs)
            ovalues[node] = activation(bias + response * s)

        self.hvalues = ovalues.copy()

        return [ovalues[i] for i in self.output_nodes]

    def create(genome, config, use_weights=True):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Initialize connections dict
        connections = {}
        connections_lr = {}
        connections_edecay = {}
        connections_decay = {}

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            # Store weight in connections dict using (i, o) key
            if use_weights or cg.lr == 0.0:
                connections[(i, o)] = cg.weight
            else:
                connections[(i, o)] = random.uniform(-0.05,0.05)
            connections_lr[(i,o)] = cg.lr
            connections_edecay[(i,o)] = cg.edecay
            connections_decay[(i,o)] = cg.decay


            # Store (i, o) key in inputs instead of weight
            if o not in node_inputs:
                node_inputs[o] = [(i, o)]  # Store the input node and the lookup key
            else:
                node_inputs[o].append((i, o))

        # Build node_evals with references to connection keys
        node_evals = []
        for node_key, input_keys in node_inputs.items():
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)

            # Store the keys instead of the weights
            node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.response, input_keys))

        # Return network with connections dict included
        return HebbianRecurrentNetwork(genome_config.input_keys, genome_config.output_keys, node_evals, connections, connections_lr, connections_edecay, connections_decay)
