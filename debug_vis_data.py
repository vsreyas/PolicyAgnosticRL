import pandas as pd
import numpy as np
import pickle

data = pickle.load(open('outputs/vlm_actions_replay_ep10_1_v2.pkl', 'rb'))
df = pd.DataFrame(data['episode_ids'])
df.columns = ['episode_id']
df['episode_timesteps'] = data['episode_timesteps']
df['terminals'] = data['terminals']
df['truncates'] = data['truncates']
df['rewards'] = data['rewards']
df['masks'] = data['masks']
df['mc_returns'] = data['mc_returns']

df['dones'] = df['terminals'] | df['truncates']

out = df[df['dones'] == True]
breakpoint()