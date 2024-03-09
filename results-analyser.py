import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open("data/results_conv_test.pkl", 'rb') as resultsFile:
    results = pickle.load(resultsFile)
    
    timestamps = [item['timestamp'] for item in results]
    episodes = [item['episode'] for item in results]
    rewards = [item['total_reward'] for item in results]
    epsilon = [item['epsilon']*100 for item in results]
    pieces = [item['drawn_pieces'] for item in results]
    linesCleared = [item['total_lines_cleared'] for item in results]
    tetrisClears = [item['total_tetris'] for item in results]
    print(len(results))
    

    df_days_calories = pd.DataFrame( 
        {'Games' : episodes ,  
        'Reward': rewards ,  
        'Pieces': pieces,
        'Lines Cleared': linesCleared,
        'Epsilon * 100': epsilon}) 
    
    ax = plt.gca()
    
    #use plot() method on the dataframe 
    df_days_calories.plot( x = 'Games' , y = 'Reward', ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Pieces' , ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Lines Cleared' , ax = ax ) 
    df_days_calories.plot( x = 'Games' , y = 'Epsilon * 100' , ax = ax ) 
    plt.show()