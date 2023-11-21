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
#from tqdm import tqdm


TARGET = 5
N = TARGET+1
G = 2
B = 2

def fmap(X,U,w):
    # Unpack variables
    n1,g1,b1,n2,g2,b2 = X
    u1,u2 = U
    # Update player 1 
    n1 = n1+w
    g1 = min( w*(g1+u1) , G-1 )
    b1 = min( b1-u1+3*w+(1-w) , B-1 )
    # Update player 1 
    n2 = n2+(1-w)
    g2 = min( (1-w)*(g2+u2) , G-1 )
    b2 = min( b2-u2+3*(1-w)+w , B-1 )
    # return next state X
    X = [n1,g1,b1,n2,g2,b2]
    return X

def get_U_values(b,g):
    if ((G-1)-g) >= b:
        U = b
    else:
        U = ((G-1)-g)
    return U

# 2x2 example
P1_win = np.array([[0.5,0.3],[0.7,0.5]])
# 3x3 example
#P1_win = np.array([[0.5,0.3,0.1],[0.7,0.5,0.3],[0.9,0.7,0.5]])

assert P1_win.shape[0] == G, "Check dimensions of P1_win"
P2_win = 1 - P1_win


####################################################
####################################################


# Value function from player 1's perspective
# V = np.empty((N,G,B,N,G,B,TARGET**2))*np.nan

# Terminal condition for the value function
# for t in range(0,(TARGET**2)):
#     for n1 in range(N):
#         for g1 in range(G):
#             for b1 in range(B):
#                 for n2 in range(N):
#                     for g2 in range(G):
#                         for b2 in range(B):
#                             if t == (TARGET**2 - 1):
#                                 V[n1,g1,b1,n2,g2,b2,t] = 0
                            
#                             if n1==TARGET and n2 < TARGET:
#                                 V[n1,g1,b1,n2,g2,b2,t] = 1
#                             elif n1 < TARGET and n2==TARGET:
#                                 V[n1,g1,b1,n2,g2,b2,t] = 0


# Terminal condition for the value function
V = np.empty((N,G,B,N,G,B,TARGET**2))*np.nan
V[(N-1),:,:,:(N-1),:,:,:] = 1               # Case where player 1 wins
V[:(N-1),:,:,(N-1),:,:,:] = 0               # Case where player 1 losses
V[:(N-1),:,:,:(N-1),:,:,TARGET**2-1] = 0    # Case where neither wins, and therefore 0 for player 1



for t in range((TARGET**2)-2,-1,-1):
    print(f"t={t}")
    for n1 in range(N-2,-1,-1):
        for n2 in range(N-2,-1,-1):
            for g1 in range(G):
                for b1 in range(B):
                    for g2 in range(G):
                        for b2 in range(B):
                            X = [n1,g1,b1,n2,g2,b2]
                            U1,U2 = get_U_values(b1,g1),get_U_values(b2,g2)

                            payoff_p1 = np.zeros((U1+1,U2+1))
                            for u1 in range(U1+1):
                                for u2 in range(U2+1):
                                    X_win = fmap(X,[u1,u2],1)
                                    n1_w,g1_w,b1_w,n2_w,g2_w,b2_w = X_win
                                    X_loss = fmap(X,[u1,u2],0)
                                    n1_l,g1_l,b1_l,n2_l,g2_l,b2_l = X_loss

                                    # Calculate from value function
                                    p = P1_win[g1+u1,g2+u2]
                                    term1 = p * V[n1_w,g1_w,b1_w,n2_w,g2_w,b2_w,t+1]
                                    term2 = (1-p) * V[n1_l,g1_l,b1_l,n2_l,g2_l,b2_l,t+1]

                                    total = term1 + term2
                                    payoff_p1[u1,u2] = total
                                    
                            strategy = nash.Game(payoff_p1, 1-payoff_p1)
                            eqs = strategy.support_enumeration() 
                            res = list(eqs)
                            V[n1,g1,b1,n2,g2,b2,t] = res[0][0] @ payoff_p1 @ res[0][1]



#%%

print(f"Tmax = {(TARGET**2 - 1)}")
T = 0

# All 16 combinations of g1,b1,g2 and b2 for 2x2 matrix
#V[n1,g1,b1,n2,g2,b2,T] 
V[:,1,1,:,1,1,T]
V[:,1,1,:,1,0,T]
V[:,1,1,:,0,1,T]
V[:,1,0,:,1,1,T]
V[:,0,1,:,1,1,T]
V[:,1,1,:,0,0,T]
V[:,1,0,:,1,0,T]
V[:,0,1,:,1,0,T]
V[:,1,0,:,0,1,T]
V[:,0,1,:,0,1,T]
V[:,0,0,:,1,1,T]
V[:,1,0,:,0,0,T]
V[:,0,1,:,0,0,T]
V[:,0,0,:,1,0,T]
V[:,0,0,:,0,1,T]
A = V[:,0,0,:,0,0,T]




                            