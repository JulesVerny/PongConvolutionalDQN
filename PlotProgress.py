#
#  MyPong DQN Reinforcement Learning Experiment

# Plot the progree of the Game 
# ==========================================================================================
import numpy as np 
import pickle, os
import numpy as np  
import matplotlib.pyplot as plt

#  Now Unpickle the Game Dat File
GFile = open('TrainHistory.dat','rb')
GameHistory = pickle.load(GFile)
GFile.close()	

# Plot the Score vs Game Time profile
x_val = [x[0] for x in GameHistory]
y_val = [x[1] for x in GameHistory]

plt.plot(x_val,y_val)
plt.xlabel("Game Time")
plt.ylabel("Score")
plt.show()
