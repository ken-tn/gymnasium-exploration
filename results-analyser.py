import pandas as pd
import matplotlib.pyplot as plt
import pickle

experimentName = "priority_smartreward_DDQN_double_convrelu_dense512relu_huber_64batch"
with open("results/{}.pkl".format(experimentName), 'rb') as resultsFile:
    results = pickle.load(resultsFile)
    
    timestamps = [item['timestamp'] for item in results]
    episodes = [item['episode'] for item in results]
    rewards = [item['total_reward'] / 1000 for item in results]
    epsilon = [item['epsilon'] for item in results]
    pieces = [item['drawn_pieces'] for item in results]
    linesCleared = [item['total_lines_cleared'] for item in results]
    score = [item['score'] for item in results]

    df_days_calories = pd.DataFrame( 
        {'Games' : episodes ,  
        'Reward': rewards ,  
        'Pieces': pieces,
        'Lines Cleared': linesCleared,
        'Score': score,
        'Epsilon * 100': epsilon}) 
    
    ax = plt.gca()
    
    #use plot() method on the dataframe 
    df_days_calories.plot( x = 'Games' , y = 'Reward', ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Pieces' , ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Lines Cleared' , ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Epsilon * 100' , ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Score' , ax = ax ) 
    plt.show()