# We play the game of 5

# Two players, p1 and p2
# each can draw 1,2,3 with equal probability
# Target is to get 5 points

# Payoff:
#  Win: 1
#  Lose: 0
#  Draw: 1/2


import numpy as np
import nashpy as nash


N = 9
T = 7
U = 2

TARGET = 5

# Value function from player 1's perspective
V = np.empty((N,N,U,U,T))*np.nan

# Terminal condition for the value function
for i in range(N):
    for j in range(N):
        if (i<j and j <= TARGET) or (i>TARGET and j<=TARGET):
            V[i,j,:,:,-1] = 0
        elif (j<i and i <=TARGET) or (j>TARGET and i<=TARGET):
            V[i,j,:,:,-1] = 1
        elif (i==j and i<=TARGET) or (i>TARGET and j>TARGET):
            V[i,j,:,:,-1] = 0.5 #draw: can be set to 0.5


def is_between_0_and_1(arr):
    if 0<arr[0]<1 and 0<arr[1]<1:
        return True
    return False
    
def is_mixed(res):
    for element in res:
        arr1,arr2 = element[0],element[1]
        if is_between_0_and_1(arr1) and is_between_0_and_1(arr2):
            return True
    return False


# If player 2 has 6 or higher player 1 wins hence:
V[:TARGET+1,TARGET+1:,:,:,:] = 1.0
# If player 1 and player 2 has 6 or higher on their hand then we have a draw:
V[TARGET+1:,TARGET+1:,:,:,:] = 0.5 #draw: can be set to 0.5
# If player 1 has 6 or higher player 1 losses hence
V[TARGET+1:,:TARGET+1,:,:,:] = 0.0

for t in range(T-2,-1,-1):
    for i in range(TARGET+1):
        for j in range(TARGET+1):
            for k1 in range(U-1,-1,-1):
                for k2 in range(U-1,-1,-1):
                    # træk værdier for hver matrice

                    if k1 == 1 and k2 == 1: # both players has stopped   
                        v22 = V[i,j,k1,k2,t+1]
                        V[i,j,k1,k2,t] = v22       
                    elif k1 == 1 and k2 == 0: # player 1 has stopped, player 2 can stop or take a card
                        v12 = np.mean(V[i,j+1:(j+3)+1,k1,k2,t+1])
                        V[i,j,k1,k2,t] = min(V[i,j,k1,k2,t+1],v12)
                    elif k1 == 0 and k2 == 1: # player 2 has stopped, player 1 can stop or take a card
                        v21 = np.mean(V[i+1:(i+3)+1,j,k1,k2,t+1])
                        V[i,j,k1,k2,t] = max(V[i,j,k1,k2,t+1],v21)
                    elif k1 == 0 and k2 == 0: # both players can stop or take a card
                        v11 = np.mean(V[i+1:(i+3)+1,j+1:(j+3)+1,k1,k2,t+1])

                        if i == 1 and j==0 and t == 0:                            
                            print("her")

                        payoff_p1 = np.array([[v11,v21],[v12,v22]]) 
                        payoff_p2 = 1-payoff_p1
                        strategy = nash.Game(payoff_p1, payoff_p2)
                        eqs = strategy.support_enumeration()
                        res = list(eqs)
                        V[i,j,k1,k2,t] = res[0][0] @ payoff_p1 @ res[0][1]
                            



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax= plt.subplots(2, 2)
im_00 = ax[0,0].imshow(V[:,:,0,0,0])
ax[0,0].set_title('k1=0, k2=0')

im_01 = ax[0,1].imshow(V[:,:,0,1,0])
ax[0,1].set_title('k1=0, k2=1')

im_10 = ax[1,0].imshow(V[:,:,1,0,0])
ax[1,0].set_title('k1=1, k2=0')
im_11 = ax[1,1].imshow(V[:,:,1,1,0])
ax[1,1].set_title('k1=1, k2=1')
plt.show()


# t = T-3

# A = V[:,:,1,1,t]
# B = V[:,:,1,1,t+1]

# A = V[:,:,0,1,t]
# B = V[:,:,0,1,t+1]

# A = V[:,:,1,0,t]
# B = V[:,:,1,0,t+1]

# A = V[:,:,0,0,t]
# B = V[:,:,0,0,t+1]


# V[3,4,0,1,t]
