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


N = 8
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
            V[i,j,:,:,-1] = 1/2


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


for t in range(T-2,T-4,-1):
    print(t)
    for i in range(TARGET+1):
        for j in range(TARGET+1):
            for k1 in range(U-1,-1,-1):
                for k2 in range(U-1,-1,-1):
                    # træk værdier for hver matrice

                    if k1 == 1 and k2 == 1: # both players stop
                        v22 = V[i,j,k1,k2,t+1]
                        V[i,j,k1,k2,t] = v22
                    elif k1 == 1 and k2 == 0: # player 1 stops, player 2 go
                        v12 = np.mean(V[i,j+1:(j+1)+3,k1,k2,t+1])
                        V[i,j,k1,k2,t]=max(V[i,j,k1,k2,t+1],v12)
                    elif k1==0 and k2==1: # player 2 stops, player 1 go
                        v21 = np.mean(V[i+1:(i+1)+3,j,k1,k2,t+1])
                        V[i,j,k1,k2,t]=max(V[i,j,k1,k2,t+1],v21)
                    elif k1 == 0 and k2 == 0: # both players go
                        v11 = np.mean(V[i+1:(i+1)+3,j+1:(j+1)+3,k1,k2,t+1])

                        payoff_p1 = np.array([[v11,v21],[v12,v22]]) 
                        payoff_p2 = 1-payoff_p1
                        strategy = nash.Game(payoff_p1, payoff_p2)
                        eqs = strategy.support_enumeration()
                        res = list(eqs)
                        if is_mixed(res):
                            print(f"{i,j} Find out what to do")
                        else:
                            if len(res) == 1:
                                row,col = np.where(res[0][0]==1)[0][0],np.where(res[0][1]==1)[0][0]
                                V[i,j,k1,k2,t] = payoff_p1[row,col]
        
                            else:
                                row,col = np.where(res[0][0]==1)[0][0],np.where(res[0][1]==1)[0][0]
                                V[i,j,k1,k2,t] = payoff_p1[row,col]
                                #print("Degenerative solution")



        # Now update
        
        # Payoff matrix

        # Calculate nash


# t = T-2

# V[0,0,:,:,t]

# V[0:i,0:j,:,:,T-1]



# #  W[i,j,k,1,t] <- min(V[i,j,k,2,t+1],
# #                                     mean(V[i,j+(1:10),k,1,t+1]))

