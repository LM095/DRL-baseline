import seaborn as sns
import pandas as pd
import matplotlib.ticker as tkr
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


sns.set_style("whitegrid")
sns.set(font_scale=1.2)
path_data_reinforce = 'Policy-gradient methods/REINFORCE/REINFORCE_logs/LunarLander-v2/'
file_name_reinforce = 'REINFORCE_LunarLander-v2_seed_3'
reinforce = pd.read_csv(f'{path_data_reinforce}{file_name_reinforce}.csv')

#path_data_DQN = 'Policy-gradient methods/REINFORCE/REINFORCE_logs/LunarLander-v2/'
#file_name_DQN = 'REINFORCE_LunarLander-v2_seed_3'
#dqn = pd.read_csv(f'{path_data_DQN}{file_name_DQN}.csv')

#ax = sns.lineplot(x="episode", y="mean_reward", hue="method", labeldata=reinforce, ci= 'sd')
ax = sns.lineplot(x="episode", y="mean_reward",hue = 'method', data=reinforce, ci= 'sd')

plt.title('LunarLander-v2')
plt.xlabel('Number of episodes')
plt.ylabel('Mean reward')
plt.legend(loc='lower right', fontsize=8)

plt.show(ax)

