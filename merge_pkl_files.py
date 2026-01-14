import pandas as pd
import numpy as np
import pickle
from typing import List

data_files = [
    'clean_skip_v3_server_scale5_actnorm.pkl',
    'clean_skip_v4_server_scale5_actnorm.pkl',
]
_data = []
episodes_so_far = 0
for file in data_files:
    cur_data = pickle.load(open(file, 'rb'))
    # add episode_ids to cur_data
    episodes_in_cur_data = len(np.unique(cur_data['episode_ids']))
    cur_data['episode_ids'] += episodes_so_far
    episodes_so_far += episodes_in_cur_data
    _data.append(cur_data)


def recursive_concat(data: List[dict]) -> dict:
    out_dict = {}
    for key, value in data[0].items():
        if key == 'metadata':
            out_dict[key] = {}
        elif isinstance(value, dict):
            out_dict[key] = recursive_concat([d[key] for d in data])
        elif value.ndim == 1:
            out_dict[key] = np.concatenate([d[key].reshape(-1, 1) for d in data], axis=0).reshape(-1)
        else:
            # orig_shape = value.shape
            out_dict[key] = np.concatenate([d[key] for d in data], axis=0)
    return out_dict

_data = recursive_concat(_data)
# breakpoint()

# Do some sanity checks before saving
data = _data
df = pd.DataFrame(data['episode_ids'])
df.columns = ['episode_id']
df['episode_timesteps'] = data['episode_timesteps']
df['terminals'] = data['terminals']
df['truncates'] = data['truncates']
df['rewards'] = data['rewards']
df['masks'] = data['masks']
df['mc_returns'] = data['mc_returns']
# df['dones'] = df['terminals'] | df['truncates']
# out = df[df['dones'] == True]
# breakpoint()

pickle.dump(data, open('pkl_files/clean_skip_merged_v3_v4_server_scale5_actnorm.pkl', 'wb'))
