import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from dataclasses import dataclass
from functools import partial
import random

EDIBLE: int = 1
POISON: int = 2

UP = [0,1]
DOWN = [0,-1]
RIGHT = [1,0] 
LEFT = [-1,0]
STAY = [0,0]

EAT: int = 4

MOVES = np.array([UP, RIGHT, DOWN, LEFT, STAY])

@dataclass
class State:
    scene: np.ndarray
    red_scene: np.ndarray
    green_scene: np.ndarray
    blue_scene: np.ndarray
    agent_location: np.ndarray
    timestep: int
    season: int
    edible_rgb: np.ndarray
    poison_rgb: np.ndarray
    reward: float

class MyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 10, v=4, n_edibles=10, n_poisons=10, lifetime=100, render_mode="human", n_seasons=1, col_dist=False, ns=100, 
            col_seed=None, col_var=0.2, energy_coef=0.0001, randcol=True) -> None:
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.v = v
        self.n_edibles = n_edibles
        self.n_poisons = n_poisons
        self.lifetime = lifetime
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=-1.01, high=1.1, shape=((3*((2*v)+1)*((2*v)+1))+2,), dtype=np.float64)

        # We have 5 actions, corresponding to "right", "up", "left", "down", "eat"
        self.action_space = spaces.Discrete(5)

        self.window = None
        self.clock = None

        self.state = None
        self.change_season = int(lifetime/n_seasons)
        self.n_seasons = n_seasons
        #self.season = 1
        self.col_var = col_var
        self.energy_coef = energy_coef
        self.randcol = randcol

        green = [0.1,0.8,0.1]
        red = [0.8,0.1,0.1]
        yellow = [0.8,0.8,0.1]
        blue= [0.1,0.1,0.8]
        orange = [245/255, 130/255, 48/255]
        purple = [0.4, 0.1, 0.4]
        brown = [0.4, 0.2, 0.1]
        pink = [0.8, 0.2, 0.7]

        self.random_colour = random.Random()

        #self.colours = [green, yellow, blue, red]#, orange, brown, purple, pink]

        if col_seed:
            self.random_colour.seed(col_seed)

        
        #edible, poison = self.generate_two_distinct()
        #self.colours = [edible, poison]

        #if not randcol:
        #self.random_colour.shuffle(self.all_colours)
        #self.colours = self.colours[:n_seasons]
        
        #self.edibles = colours[:4]
        #self.poisons = colours[4:]

        self.col_dist = col_dist
        self.text = "Reward: "

        self.ns = ns

        self.edibles = None
        self.poisons = None

        if not randcol:
            self.gen_season_colours_rand()

        #self.gen_season_colours()

    def gen_season_colours_rand(self):
        self.edibles = []
        self.poisons = []

        for _ in range(self.n_seasons):
            edible, poison = self.generate_two_distinct()
            self.edibles.append(edible)
            self.poisons.append(poison)

    def gen_season_colours_fluc(self):
        self.edibles = []
        self.poisons = []
        edible = None
        poison = None
        for _ in range(self.n_seasons):

            if edible is not None:
                possible_edible = [c for c in self.colours if c != edible]
                edible = self.random_colour.choice(possible_edible)
            else:
                edible = self.random_colour.choice(self.colours)
            
            # choose poison but different from edible
            if poison is None:
                poison = edible
            possible_poison = [c for c in self.colours if (c != edible) and (c != poison)]
            poison = self.random_colour.choice(possible_poison)

            self.edibles.append(edible)
            self.poisons.append(poison)

    def random_rgb(self):
        return [self.random_colour.random(), self.random_colour.random(), self.random_colour.random()]

    def generate_two_distinct(self, min_dist=0.7):
        while True:
            edible = self.random_rgb()
            poison = self.random_rgb()
            if np.linalg.norm(np.array(edible) - np.array(poison)) >= min_dist:
                return edible, poison

    def _get_obs(self):
        state = self.state
        x, y = state.agent_location
        v = self.v

        full_red = np.zeros([self.size+(2*v), self.size+(2*v)])
        full_red[v:v + state.red_scene.shape[0], v:v + state.red_scene.shape[1]] = state.red_scene
        full_green = np.zeros([self.size+(2*v), self.size+(2*v)])
        full_green[v:v + state.green_scene.shape[0], v:v + state.green_scene.shape[1]] = state.green_scene
        full_blue = np.zeros([self.size+(2*v), self.size+(2*v)])
        full_blue[v:v + state.blue_scene.shape[0], v:v + state.blue_scene.shape[1]] = state.blue_scene
    
        
        obs_red = full_red[x:x + (2 * v) + 1, y:y + (2 * v) + 1]
        obs_green = full_green[x:x + (2 * v) + 1, y:y + (2 * v) + 1]
        obs_blue = full_blue[x:x + (2 * v) + 1, y:y + (2 * v) + 1]

        loc = ((self.size*x) + y)/(self.size*self.size)
        
        reward_loc = [state.reward, loc]

        #return np.array([obs_red, obs_green, obs_blue]).flatten()
        return np.append(np.array([obs_red, obs_green, obs_blue]).flatten(), reward_loc)
    
    def _get_info(self):
        return {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randcol:
            self.random_colour.seed(seed)
            #self.random_colour.shuffle(self.colours)

            #self.edibles = self.colours[:4]
            #self.poisons = self.colours[4:]

            #self.edibles = []
            #self.poisons = []

            self.gen_season_colours_rand()
            '''
            edible = None
            for _ in range(self.n_seasons):

                if edible is not None:
                    possible_edible = [c for c in self.colours if c != edible]
                    edible = self.random_colour.choice(possible_edible)

                else:
                    edible = self.random_colour.choice(self.colours)

                # choose poison but different from edible
                possible_poison = [c for c in self.colours if c != edible]
                poison = self.random_colour.choice(possible_poison)

                self.edibles.append(edible)
                self.poisons.append(poison)
            '''

        edible_rgb, poison_rgb = self.edibles[0], self.poisons[0]
        scene, red_scene, green_scene, blue_scene = self._generate_scenes(edible_rgb, poison_rgb)
        # Random agent location within the grid
        agent_location = self.np_random.integers(low=0, high=self.size, size=(2,))
        new_state = State(
                agent_location=agent_location,
                scene=scene,
                red_scene=red_scene,
                green_scene=green_scene,
                blue_scene=blue_scene,
                timestep= 1,
                season = 0,
                edible_rgb=edible_rgb,
                poison_rgb=poison_rgb,
                reward = 0
            )
        self.state = new_state

        return self._get_obs(), self._get_info()
    
    #@partial(jit, static_argnums=(0,))
    def step(self, action):
        state = self.state

        ate_food = np.logical_and(state.scene[state.agent_location[0], state.agent_location[1]] != 0, action == EAT)

        reward = 0.0

        if ate_food:  # Check if food was eaten
            updated_scene, updated_red_scene, updated_green_scene, updated_blue_scene, reward = self.update_scenes()
        else:
            updated_scene = state.scene
            updated_red_scene = state.red_scene
            updated_green_scene = state.green_scene
            updated_blue_scene = state.blue_scene

        reward = reward - (self.ns * self.energy_coef)

        agent_location = np.clip(
            state.agent_location + MOVES[action], 0, self.size-1
        )

        done = bool(np.equal(state.timestep, self.lifetime))

        edible_rgb, poison_rgb = state.edible_rgb, state.poison_rgb
        season = state.season
        
        if not done and state.timestep % self.change_season == 0 and state.timestep > 0 and season+1 < self.n_seasons:
            season += 1
            edible_rgb = self.edibles[season]
            poison_rgb = self.poisons[season]
            updated_red_scene, updated_green_scene, updated_blue_scene = self.rgb_scene(updated_scene, edible_rgb, poison_rgb)

        # Build the state.
        new_state = State(
            agent_location=agent_location,
            scene=updated_scene,
            red_scene=updated_red_scene,
            green_scene=updated_green_scene,
            blue_scene=updated_blue_scene,
            timestep=state.timestep + 1,
            season = season,
            edible_rgb=edible_rgb,
            poison_rgb=poison_rgb,
            reward = reward
        )
        self.state = new_state

        self.text = "Reward: " + str(reward)
      
        return self._get_obs(), reward, done, False, self._get_info()
    
    def rgb_scene(self,scene,edible_rgb,poison_rgb):
        edible_scene = np.where(scene == EDIBLE, 1, 0)
        poison_scene = np.where(scene == POISON, 1, 0)
        food_scene = np.where(scene != 0, 1, 0)

        if self.col_dist:
            cvar = self.col_var
            red_scene = np.clip((edible_rgb[0]*edible_scene) + (poison_rgb[0]*poison_scene) + (self.np_random.uniform(-cvar, cvar, size=(self.size, self.size))*food_scene),0,1)
            green_scene = np.clip((edible_rgb[1]*edible_scene) + (poison_rgb[1]*poison_scene) + (self.np_random.uniform(-cvar, cvar, size=(self.size, self.size))*food_scene),0,1)
            blue_scene = np.clip((edible_rgb[2]*edible_scene) + (poison_rgb[2]*poison_scene) + (self.np_random.uniform(-cvar, cvar, size=(self.size, self.size))*food_scene),0,1)
        else:
            red_scene = (edible_rgb[0]*edible_scene) + (poison_rgb[0]*poison_scene) 
            green_scene = (edible_rgb[1]*edible_scene) + (poison_rgb[1]*poison_scene) 
            blue_scene = (edible_rgb[2]*edible_scene) + (poison_rgb[2]*poison_scene)

        return red_scene, green_scene, blue_scene

    
    def _generate_scenes(self, edible_rgb, poison_rgb):
        scene = np.zeros([self.size, self.size], dtype=int)
        # Generate random indices for placing the foods
        indices = self.np_random.choice(self.size * self.size, self.n_edibles + self.n_poisons, replace=False)

        # Get the 2D coordinates from the 1D indices
        coords = np.unravel_index(indices, (self.size, self.size))

        # Place edible items in the scene (value of 1)
        scene[coords[0][:self.n_edibles], coords[1][:self.n_edibles]] = EDIBLE

        # Place poisonous items in the scene (value of 2)
        scene[coords[0][self.n_edibles:], coords[1][self.n_edibles:]] = POISON

        red_scene, green_scene, blue_scene = self.rgb_scene(scene, edible_rgb, poison_rgb)

        return scene, red_scene, green_scene, blue_scene
    

    def update_scenes(self):
        state = self.state
        x, y = state.agent_location[0], state.agent_location[1]
        val = state.scene[x,y]
        
        if val == EDIBLE:
            updated_reward = 1.0
            rgb = state.edible_rgb
        else:
            updated_reward = -1.0
            rgb = state.poison_rgb

        state.scene[x,y] = 0
        state.red_scene[x,y] = 0
        state.green_scene[x,y] = 0
        state.blue_scene[x,y] = 0

        # place food in new location
        x_coords, y_coords = np.where(state.scene == 0)
        coords = list(zip(x_coords, y_coords))
        random_coord = coords[self.np_random.choice(len(coords))]
        x,y = random_coord

        state.scene[x,y] = val
        if self.col_dist:
            cvar = self.col_var
            state.red_scene[x,y] = np.clip(rgb[0] + self.np_random.uniform(-cvar, cvar), 0, 1)
            state.green_scene[x,y] = np.clip(rgb[1] + self.np_random.uniform(-cvar, cvar), 0, 1)
            state.blue_scene[x,y] = np.clip(rgb[2] + self.np_random.uniform(-cvar, cvar), 0, 1)
        else:
            state.red_scene[x,y] = rgb[0] 
            state.green_scene[x,y] = rgb[1] 
            state.blue_scene[x,y] = rgb[2]

        return (state.scene, state.red_scene, state.green_scene, state.blue_scene, updated_reward)
    
    def render(self):
        return self._render_frame()

    def _render_frame(self):
        state = self.state
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pygame.display.set_caption(self.text)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        for i in range(self.size):
            for j in range(self.size):
                pygame.draw.rect(canvas,(255*state.red_scene[i,j], 255*state.green_scene[i,j], 255*state.blue_scene[i,j]),
                                     pygame.Rect(pix_square_size * np.array([i,j]),(pix_square_size, pix_square_size),),)
                
              
        i,j = state.agent_location
        pygame.draw.rect(canvas,(0.5*255*state.red_scene[i,j], 0.5*255*state.green_scene[i,j], 0.5*255*state.blue_scene[i,j]),
                                     pygame.Rect(pix_square_size * np.array([i,j]),(pix_square_size, pix_square_size),),)    
           
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (255, 255, 255),
            (state.agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 3):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    

def testEnv():
    env = MyEnv(n_seasons=4, col_dist=True, col_seed=10953)
    observation, info = env.reset(1361)
    all_episode_rewards = []
    #print("action space: ", env.action_space)
    print("edibles")
    print(env.edibles)
    print("poisons")
    print(env.poisons)
    for i in range(1):
        print(i)
        episode_rewards = []
        done = False
        observation, info = env.reset()
        step = 0
        while not done:
            #print(i)
            print(step)
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, done, truncated, info = env.step(action)
            #print(observation)
            #print()
            env.render()
            episode_rewards.append(reward)
            step += 1
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", 1000)
    env.close()
    
'''
def test_agent():

    from sbx import PPO
    #from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy

    env = MyEnv(n_seasons=4, col_dist=True)
    check_env(env, warn=True, skip_render_check=True)
    
    
    #policy_kwargs = dict(net_arch=dict(pi=[64,64], vf=[64,64]))
    model = PPO("MlpPolicy", env, verbose=0, seed=2059)#, policy_kwargs=policy_kwargs)
    
    model.learn(total_timesteps=100_000, progress_bar=True)
    env = model.get_env()
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False, render=False)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    #evaluate_policy(model, env, n_eval_episodes=1, deterministic=False, render=True)
    return mean_reward,  std_reward
'''
    
if __name__ == '__main__':
    testEnv()
    #test_agent()
