from multiprocessing import Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None, n_seasons=4, seed=1361, energy_costs=False, n_trials=100, learn=True, randcol=True):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        self.eval_function = eval_function
        self.timeout = timeout
        self.num_workers = num_workers
        self.maxtasksperchild = maxtasksperchild
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)
        self.n_seasons = n_seasons
        self.seed = seed
        self.energy_costs = energy_costs
        self.n_trials = n_trials
        self.learn = learn
        self.randcol = randcol
        self.gen = 1

    def __del__(self):
        """Clean up the current pool."""
        if self.pool:
            print("Terminating pool...")
            self.pool.terminate()
            try:
                print("Joining pool...")
                self.pool.join()
            except Exception as e:
                print(f"Error during pool join: {e}")
        self.pool = None

    def reinitialize(self):
        """Clean up the current pool."""
        if self.pool:
            print("Terminating pool...")
            self.pool.terminate()
            try:
                print("Joining pool...")
                self.pool.join()
            except Exception as e:
                print(f"Error during pool join: {e}")
        """Reinitialize the pool."""
        self.pool = Pool(processes=self.num_workers, maxtasksperchild=self.maxtasksperchild)

    def evaluate_(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, self.learn, self.n_trials, True, self.n_seasons, self.seed, self.energy_costs, self.randcol,50,self.gen)))

        timeout_occurred = False
        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            #try:
            if True:
                if timeout_occurred:
                    genome.fitness, genome.net, genome.mean_perf = job.get(timeout=3)
                else:
                    genome.fitness, genome.net, genome.mean_perf = job.get(timeout=self.timeout)


    def evaluate(self, genomes, config):
        self.evaluate_(genomes, config)
        retry = False
        for genome_id, genome in genomes:
            if genome.fitness is None:
                retry = True
                break
        
        if retry:
            print("retrying")
            self.reinitialize()  # Clean up and recreate the pool
            print("reinitialized")
            self.evaluate_(genomes, config)


        
