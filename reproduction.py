"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""

import math
import random
from itertools import count

#from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean

from neat.reproduction import DefaultReproduction

# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.


class CustomReproduction(DefaultReproduction):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """


    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        super().__init__(config, reporters, stagnation)
        self.mean_distance = 0

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.

        config.genome_config.innovation_tracker = self.innovation_tracker
        self.innovation_tracker.reset_generation()

        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        survival_threshold = self.reproduction_config.survival_threshold

        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        mutations = 0
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)
            elitism = self.reproduction_config.elitism

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0 and spawn > 5:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue


            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(survival_threshold *
                                         len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]
            
            # Assign weights based on rank (higher rank = higher weight).
            # For example, use linear decay: weight = rank (1-based) reversed.
            ranks = list(range(len(old_members), 0, -1))  # e.g., [10, 9, ..., 1] for 10 genomes
            total = sum(ranks)
            weights = [r / total for r in ranks]  # Normalized weights
            print("best and worst surviving genome parent prob:", weights[0], weights[-1])

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                #parent1_id, parent1 = random.choice(old_members)
                parent1_id, parent1 = random.choices(old_members, weights=weights, k=1)[0]
                #parent2_id, parent2 = random.choice(old_members)
                parent2_id, parent2 = random.choices(old_members, weights=weights, k=1)[0]

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                '''
                if random.random() < 0.5:
                    child.mutate_structure(config.genome_config)
                    mutations += 1
                else:
                    child.mutate_weights(config.genome_config)
                '''
                child.mutate(config.genome_config)
                # TODO: if config.genome_config.feed_forward, no cycles should exist
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)
        print(mutations, "structural mutations")
        return new_population

