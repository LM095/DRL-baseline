import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class logger:
    def __init__(self, env_name, seed, method):
        
        ###################### create log file ######################
        log_dir = f"{method}_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### create new log file for each run
        log_f_name = log_dir + f'/{method}_' + env_name + "_seed_" + str(seed) + ".csv"

         # open file to write
        self.log_f = open(log_f_name,"w+")
        self.log_f.write('episode,episode_reward,mean_reward,method\n')

        ################### checkpointing ###################
       
        directory = f"{method}_models"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)


        self.checkpoint_path = directory + "{}_{}_{}".format(method,env_name, seed)

       
    def write(self, episode, episode_reward, mean_reward):
        method = 'DDQN'
        self.log_f.write(f'{episode},{episode_reward},{mean_reward},{method}\n')
        self.log_f.flush()

          
    def save_model(self, model, episode, mean_reward):
        model.save(self.checkpoint_path + \
            '_episode' + str(episode) + \
            '_meanRew' + str(mean_reward) + \
            '.h5')