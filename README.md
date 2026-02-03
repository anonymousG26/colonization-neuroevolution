# colonization-neuroevolution

## Setup

1. Copy the repository:
```bash
git clone https://github.com/anonymousG26/colonization-neuroevolution.git
cd colonization-neuroevolution
```

2. (Optional) Set up a Python virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiment Configuration
All NEAT evolution settings, including mutations for hebbian learning parameters, are specified in the configuration file neat_config. Adjust this file to modify population size, learning parameters, or mutation rates.

## Running Experiments
Note: Depending on your configuration, evolutionary runs can take significant time.
For example, the default experiment evolves 150 agents using Hebbian learning over 1,000 timesteps per agent for 600 generations.

To start a standard evolutionary run (predictable 4-season environment with no transition):
```bash
python evolve.py
```

For a full list of options and command-line arguments:
```bash
python evolve.py -h
```

## Example: Custom Run
Run an experiment with modified parameters, including unpredictable seasonal changes and a gradual transition (from 1-season to 4-season environment):
```bash
python evolve.py --n_gens 400 --seed 123 --n_procs 2 --unpredictable --transition gradual
```
