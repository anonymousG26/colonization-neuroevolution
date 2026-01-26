import neat
import random
from neat.genes import BaseGene, DefaultNodeGene, DefaultConnectionGene
from neat.attributes import FloatAttribute, BoolAttribute
from neat.genome import DefaultGenomeConfig

TAU_E_VALUES = [1,2,3,4,5,10,20,50,100]
LR_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]


'''
v14: lr and edecay per connection. output node to hidden or loop
'''

class HebbianConnectionGene(DefaultConnectionGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled'),
                        FloatAttribute('lr'),
                        FloatAttribute('edecay'),
                        FloatAttribute('decay')]


    def distance(self, other, config):
        # start with standard weight difference
        d = abs(self.weight - other.weight)
        # include hebbian params in distance (scale as you see fit)
        d += 2*abs(self.lr - getattr(other, 'lr', 0.0))
        d += 2*abs(self.edecay - getattr(other, 'edecay', 0.0))
        #d += abs(self.decay - getattr(other, 'decay', 0.0))
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient


    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # For connection genes, verify innovation numbers match
        # (they should represent the same historical mutation)
        if hasattr(self, 'innovation'):
            assert hasattr(gene2, 'innovation'), "Both genes must have innovation numbers"
            assert self.innovation == gene2.innovation, (
                f"Genes with same key must have same innovation number: "
                f"{self.innovation} vs {gene2.innovation}"
            )

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        if hasattr(self, 'innovation'):
            new_gene = self.__class__(self.key, innovation=self.innovation)
        else:
            new_gene = self.__class__(self.key)

        for a in self._gene_attributes:
            if random.random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))


        if hasattr(new_gene, 'enabled'):
            new_gene.enabled = self.enabled

        return new_gene

class HebbianRecurrentGenome(neat.DefaultGenome):

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = HebbianConnectionGene
        return DefaultGenomeConfig(param_dict, cls.__name__)

    def __init__(self, key):
        super().__init__(key)
        self.net = None
        self.mean_perf = None

    def configure_new(self, config):
        super().configure_new(config)
        self.net = None
        self.mean_perf = None


    def mutate(self, config):
        r = random.random()
        if r < 0.3:
            self.mutate_structure(config)
        elif r < 0.6:
            self.mutate_weights(config)
        else:
            self.mutate_structure(config)
            self.mutate_weights(config)

    def mutate_weights(self, config):
        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_structure(self, config):
        """ Mutates this genome. """

        available_nodes = [k for k in self.nodes if k not in config.output_keys]

        hidden_nodes = False

        n_hidden = len(available_nodes)

        #if no hidden nodes, remove delete node mutation
        if n_hidden == 0:
            node_delete_prob = 0
        elif n_hidden == 1:
            node_delete_prob = 0.1
            hidden_nodes = True
        else:
            node_delete_prob = config.node_delete_prob * (1 - (1/n_hidden))
            hidden_nodes = True

        if config.single_structural_mutation:
            div = max(1, (config.node_add_prob + node_delete_prob +
                          config.conn_add_prob + config.conn_delete_prob
                          ))
            r = random.random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob) / div):
                self.mutate_add_del_connection(config, hidden_nodes)
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_del_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

    def mutate_add_del_connection(self, config, hidden_nodes=False, in_node=None, out_node=None):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """

        possible_outputs = list(self.nodes)


        if in_node is None:
            if hidden_nodes and random.random() < 0.1:
                possible_inputs = possible_outputs
            else:
                possible_inputs = possible_outputs + config.input_keys
            in_node = random.choice(possible_inputs)
        
        if out_node is None:
            if hidden_nodes and in_node in config.output_keys:
                possible_outputs = [k for k in self.nodes if k not in config.output_keys]
                possible_outputs.append(in_node)
            out_node = random.choice(possible_outputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                if self.connections[key].enabled: #possible infinite loop
                    #return self.mutate_add_connection(config)
                    self.connections[key].enabled = False
                    #del self.connections[key]
                else:
                    self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node != out_node and in_node in config.output_keys and out_node in config.output_keys:
            return self.mutate_add_del_connection(config, hidden_nodes)

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(self.connections), key):
            return

        # Get innovation number for this connection
        # Same connection added by multiple genomes in same generation gets same number
        innovation = config.innovation_tracker.get_innovation_number(
            in_node, out_node, 'add_connection'
        )

        cg = self.create_connection(config, in_node, out_node, innovation)
        self.connections[cg.key] = cg

    def add_connection_v1(self, config, input_key, output_key, weight, enabled, lr, edecay):
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_add_node_v1(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        assert config.innovation_tracker is not None, (
            "Innovation tracker must be set before genome mutations. "
            "This should be set by the reproduction module."
        )

        # Choose a random connection to split
        conn_to_split = random.choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)

        if hasattr(ng, "bias"):
            ng.bias = 0.0

        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key

        
        # Get innovation numbers for the two new connections
        # These are keyed by the connection being split, so multiple genomes splitting
        # the same connection get matching innovation numbers
        in_innovation = config.innovation_tracker.get_innovation_number(
            i, new_node_id, 'add_node_in'
        )
        out_innovation = config.innovation_tracker.get_innovation_number(
            new_node_id, o, 'add_node_out'
        )
        

        # Add the two new connections with their innovation numbers
        self.add_connection(config, i, new_node_id, 1.0, True, innovation=in_innovation)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True, innovation=out_innovation)
