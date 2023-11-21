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
from tqdm import tqdm


TARGET = 3
N = TARGET+1
G = 2
B = 2

def fmap(X,U,w):
    n1,g1,b1,n2,g2,b2 = X
    u1,u2 = U
    # 
    n1 = n1+w
    g1 = min(w*g1 + u1,G-1)
    b1 = min(b1-u1+3*w+(1-w),B-1)

    n2 = n2+(1-w)
    g2 = min((1-w)*g2 + u2,G-1)
    b2 = min(b2-u2+3*(1-w)+w,B-1)
    X = [n1,g1,b1,n2,g2,b2]
    return X



P1_win = np.array([[0.5,0.3],[0.7,0.5]])
assert P1_win.shape[0] == G, "Check dimensions of P1_win"
P2_win = 1- P1_win


####################################################
####################################################


# Value function from player 1's perspective
V = np.empty((N,G,B,N,G,B,TARGET**2))*np.nan

# Terminal condition for the value function
for t in range(0,(TARGET**2)):
    for n1 in range(N):
        for g1 in range(G):
            for b1 in range(B):
                for n2 in range(N):
                    for g2 in range(G):
                        for b2 in range(B):
                            if n1==TARGET and n2 < TARGET:
                                V[n1,g1,b1,n2,g2,b2,t] = 1
                            elif n1< TARGET and n2==TARGET:
                                V[n1,g1,b1,n2,g2,b2,t] = 0
        



for t in range((TARGET**2)-2,-1,-1):
    for n1 in range(N-2,-1,-1):
        for n2 in range(N-2,-1,-1):
            for g1 in range(G):
                for b1 in range(B):
                    for g2 in range(G):
                        for b2 in range(B):
                            X = [n1,g1,b1,n2,g2,b2]
                            U1,U2 = max(b1-g1,G-g1),max(b2-g2,G-g2)

                            payoff_p1 = np.zeros((U1,U2))
                            for u1 in range(U1):
                                for u2 in range(U2):
                                    X_win = fmap(X,[u1,u2],1)
                                    n1_w,g1_w,b1_w,n2_w,g2_w,b2_w = X_win
                                    X_loss = fmap(X,[u1,u2],0)
                                    n1_l,g1_l,b1_l,n2_l,g2_l,b2_l = X_loss

                                    # Calculate from value function
                                    term1 = P1_win[g1_w,g2_w] * V[n1_w,g1_w,b1_w,n2_w,g2_w,b2_w,t+1]
                                    term2 = (1-P1_win[g1_w,g2_w]) * V[n1_l,g1_l,b1_l,n2_l,g2_l,b2_l,t+1]

                                    total = term1 + term2
                                    payoff_p1[u1,u2] = total
                            if n1==2 and g1==0 and b1==0 and n2==1 and g2==0 and b2==0:
                                print("hej")
                            strategy = nash.Game(payoff_p1, 1-payoff_p1)
                            eqs = strategy.support_enumeration() 
                            res = list(eqs)
                            V[n1,g1,b1,n2,g2,b2,t] = res[0][0] @ payoff_p1 @ res[0][1]








                                    


                                    # payoff1 = P1_win[g1:max(b1,G),g2:max(b2,G)]
                                    # strategy = nash.Game(payoff1, 1-payoff1)
                                    # eqs = strategy.support_enumeration() 
                                    # res = list(eqs)
                                    # tmp_value =  res[0][0] @ payoff1 @ res[0][1]
                                    # value_dict[f"v[{u1}][{u2}]"] = tmp_value

            
                            






            # for u1 in range(G1-1,-1,-1):
            #     for u2 in range(G2-1,-1,-1):
            #         # ensure that if b_i i in {1,2} is 0 then not possible for investment








            #         if u1 == 1 and u2 == 1: # both players has stopped   
            #             v22 = V[i,j,u1,u2,t+1]
            #             V[i,j,u1,u2,t] = v22       
            #         elif u1 == 1 and u2 == 0: # player 1 has stopped, player 2 can stop or take a card
            #             v12 = np.mean(V[i,j+1:(j+N_CARDS)+1,u1,u2,t+1])
            #             V[i,j,u1,u2,t] = min(V[i,j,u1,u2,t+1],v12)
            #         elif u1 == 0 and u2 == 1: # player 2 has stopped, player 1 can stop or take a card
            #             v21 = np.mean(V[i+1:(i+N_CARDS)+1,j,u1,u2,t+1])
            #             V[i,j,u1,u2,t] = max(V[i,j,u1,u2,t+1],v21)
            #         elif u1 == 0 and u2 == 0: # both players can stop or take a card
            #             v11 = np.mean(V[i+1:(i+N_CARDS)+1,j+1:(j+N_CARDS)+1,u1,u2,t+1])

            #             payoff_p1 = np.array([[v11,v21],[v12,v22]]) 
            #             payoff_p2 = 1-payoff_p1
            #             strategy = nash.Game(payoff_p1, payoff_p2)
            #             eqs = strategy.support_enumeration()
            #             res = list(eqs)
            #             V[i,j,u1,u2,t] = res[0][0] @ payoff_p1 @ res[0][1]
                            