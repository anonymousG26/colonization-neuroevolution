"""
Gathers (via the reporting interface) and provides (to callers and/or a file)
the most-fit genomes and information on genome/species fitness and species sizes.
"""
import copy
import csv

from neat.math_util import mean, stdev, median2
from neat.reporting import BaseReporter

class CustomReporter(BaseReporter):
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genome/species fitness and species sizes.
    """

    def __init__(self):
        BaseReporter.__init__(self)
        self.stats = None

    def post_evaluate(self, config, population, species, best_genome):
        # Store the fitnesses of the members of each currently active species.
        species_stats = {}
        for sid, s in species.species.items():
            species_stats[sid] = dict((k, v.fitness) for k, v in s.members.items())
        self.stats = species_stats

    def get_fitness_stat(self, f):
        scores = []
        for species_stats in self.stats.values():
            scores.extend(species_stats.values())
        return (f(scores))

    def get_fitness_mean(self):
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat(mean)

    def get_fitness_stdev(self):
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(stdev)

    def get_fitness_median(self):
        """Get the per-generation median fitness."""
        return self.get_fitness_stat(median2)
