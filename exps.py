import numpy as np
import nashpy as nash
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

TARGET = 5
N = TARGET+1
G = 5 #5
B = 6
SIM_GAMES = 500

ORDER_TYPE = 0.5 #1,2,4 # 'Poly', 'Exp'

def fmap(X,U,w):
    # Unpack variables
    n1,g1,b1,n2,g2,b2 = X
    u1,u2 = U
    # Update player 1 
    n1 = n1+w
    g1 = min( w*(g1+u1) , G-1 )
    b1 = min( b1-u1+3*w+(1-w) , B-1 )
    # Update player 2 
    n2 = n2+(1-w)
    g2 = min( (1-w)*(g2+u2) , G-1 )
    b2 = min( b2-u2+3*(1-w)+w , B-1 )
    # Return next state vector X
    X = [n1,g1,b1,n2,g2,b2]
    return X

def get_admissible_controls(x):
    # Unpack state vector
    n1,g1,b1,n2,g2,b2 = x
    # Player 1
    if ((G-1)-g1) >= b1:
        U1 = b1
    else:
        U1 = ((G-1)-g1)
    # Player 2
    if ((G-1)-g2) >= b2:
        U2 = b2
    else:
        U2 = ((G-1)-g2)
    # Return admissible set of controls
    return U1, U2

# 2x2 example
#P1_win = np.array([[0.5,0.3],[0.7,0.5]])
# 3x3 example
#P1_win = np.array([[0.5,0.2,0.1],[0.8,0.5,0.2],[0.9,0.8,0.5]])
# 4x4 example
# =============================================================================
# p0 = 0.5
# p1 = 0.6
# p2 = 0.8
# p3 = 0.9
# P1_win = np.array([[p0, 1-p1, 1-p2, 1-p3],
#                     [p1, p0, 1-p1, 1-p2],
#                     [p2, p1, p0, 1-p1],
#                     [p3, p2, p1, p0]
#                     ])
# =============================================================================


# 5x5 example
# =============================================================================

def get_probs(order,n):
    x = np.linspace(0,n,n+1)
    b = 0.5
    a = (0.99-0.5)/(n)**order
    return a*x**order + b
p_array = get_probs(ORDER_TYPE,G-1)
q_array = 1-p_array

P1_win = np.zeros((G,G))
P1_win[0:,0]=p_array[:]
for i in range(1,len(p_array)):
    P1_win[i:,i]=p_array[:-i]
    P1_win[:i,i] = q_array[1:i+1][::-1]

# =============================================================================

assert P1_win.shape[0] == G, "Check dimensions of P1_win"
P2_win = 1 - P1_win


####################################################
####################################################


# Terminal condition for the value function
V = np.empty((N,G,B,N,G,B))*np.nan
V[(N-1),:,:,:(N-1),:,:] = 1               # Case where player 1 wins
V[:(N-1),:,:,(N-1),:,:] = 0               # Case where player 1 losses
V[:(N-1),:,:,:(N-1),:,:] = 0    # Case where neither wins, and therefore 0 for player 1
#V[-1,:,:,-1,:,:] = 0                     # This state is not possible to hit. Could be assigned np.nan.

counter = 0
counter_global = 0
for n1 in range(N-2,-1,-1):
    for n2 in range(N-2,-1,-1):
        for g1 in range(G):
            for b1 in range(B):
                for g2 in range(G):
                    for b2 in range(B):
                        X = [n1,g1,b1,n2,g2,b2]
                        U1,U2 = get_admissible_controls(X)

                        payoff_p1 = np.zeros((U1+1,U2+1))
                        for u1 in range(U1+1):
                            for u2 in range(U2+1):
                                X_win = fmap(X,[u1,u2],1)
                                n1_w,g1_w,b1_w,n2_w,g2_w,b2_w = X_win
                                X_loss = fmap(X,[u1,u2],0)
                                n1_l,g1_l,b1_l,n2_l,g2_l,b2_l = X_loss

                                # Calculate from value function
                                p = P1_win[g1+u1,g2+u2]
                                term1 = p * V[n1_w,g1_w,b1_w,n2_w,g2_w,b2_w]
                                term2 = (1-p) * V[n1_l,g1_l,b1_l,n2_l,g2_l,b2_l]

                                total = term1 + term2
                                payoff_p1[u1,u2] = total

                        with warnings.catch_warnings(record=True) as w:
                            # Trigger a filter to catch runtime warnings
                            warnings.simplefilter("always", RuntimeWarning)

                            # Call the function that might generate a warning
                            strategy = nash.Game(payoff_p1, 1-payoff_p1)
                            eqs = strategy.support_enumeration()
                            res = list(eqs)
                            counter_global+=1
                            # Check if any runtime warnings were caught
                            if w:
                                counter+=1
                                print(f"Counter: {counter} payoff_p1: {payoff_p1} X: {X}")

                        V[n1,g1,b1,n2,g2,b2] = res[0][0] @ payoff_p1 @ res[0][1]





# print(V[:,0,0,:,0,0])

print(counter/counter_global,"her")
# #%%
def get_optimal_policy(x): # This function depends on the value array V
    
    # Unpack state vector
    n1,g1,b1,n2,g2,b2 = x
    
    # Given the state x we find the admissible controls
    U1,U2 = get_admissible_controls(x)
    #print(U1, U2)
    
    # Initialize payoff matrix
    payoff_p1 = np.zeros((U1+1,U2+1))
    
    # Loop through all possible combinations of controls
    for u1 in range(U1+1):
        for u2 in range(U2+1):
            #print(f"u1 = {u1}, u2 = {u2}")
            # Advance state in case p1 wins
            X_win = fmap(x,[u1,u2],1)
            n1_w,g1_w,b1_w,n2_w,g2_w,b2_w = X_win
            
            # Advance state in case p1 losses
            X_loss = fmap(x,[u1,u2],0)
            n1_l,g1_l,b1_l,n2_l,g2_l,b2_l = X_loss
            
            # Calculate from value function the expected_payoff
            p = P1_win[g1+u1,g2+u2]
            term1 = p * V[n1_w,g1_w,b1_w,n2_w,g2_w,b2_w]
            term2 = (1-p) * V[n1_l,g1_l,b1_l,n2_l,g2_l,b2_l]
            expected_payoff = term1 + term2
            
            # add expected_payoff to the payoff matrix
            payoff_p1[u1,u2] = expected_payoff
    
    # Solve nash game for the given payoff matrix
    payoff_p2 = 1-payoff_p1
    strategy = nash.Game(payoff_p1, payoff_p2)
    eqs = strategy.support_enumeration() 
    optimal_policy = list(eqs)
    return optimal_policy, U1, U2, payoff_p1

# # Given a state x
# x = [0,2,2,0,0,2]
# # We can find the optimal policy/desicion using the 
# policy,U1,U2,payoff_p1 = get_optimal_policy(x)
# print(policy)

# #%% Simulate full dynamic game:

print("Simulation of a Game")
print("------------------")

# # Initial state
game_df = pd.DataFrame({'Game':[],'t':[],'n1':[],'g1':[],'b1':[],'n2':[],'g2':[],'b2':[],'u1':[],'u2':[]})

plt.figure()
for game_realizations in tqdm(range(SIM_GAMES)):
    X = [0,0,1,0,0,1]
    # # Array to visialize game
    game = X
    game_on = True
    t = 0
    wins_1 = 0
    wins_2 = 0
    #game_df = pd.DataFrame({'round':[],'t':[],'n1':[],'g1':[],'b1':[],'n2':[],'g2':[],'b2':[],'u1':[],'u2':[]})


    while game_on:
        #print(f"X = {X}")
        #print(f"Score: n1={X[0]} and n2={X[3]}")
        
        # Take optimal desicion
        policy,U1,U2,payoff_p1 = get_optimal_policy(X)
        #print(policy)
        u1 = np.random.choice(list(range(U1+1)),p=policy[-1][0])
        u1_vec = np.zeros(G)
        u1_vec[u1] = 1
        #print(f"Player 1: Gear={X[1]}, Bank={X[2]} | decision: u1 = {u1}")
        
        u2 = np.random.choice(list(range(U2+1)),p=policy[-1][1])
        u2_vec = np.zeros(G)
        u2_vec[u2] = 1
        #print(f"Player 2: Gear={X[4]}, Bank={X[5]} | decision: u2 = {u2}")
        
        U = np.array([u1, u2])
        game_df.loc[len(game_df)] = [int(game_realizations+1),int(t),int(wins_1),int(X[1]),int(X[2]),int(wins_2),int(X[4]),int(X[5]),int(u1),int(u2)]

        # Simulate outcome
        p = u1_vec @ P1_win @ u2_vec
        w = np.random.choice([1, 0], p = [p, 1-p])
        #print(f"Did player 1 win: w = {w}")

        wins_1+=w
        wins_2 += (1-w)
        # Advance the system to next state
        X = fmap(X,U,w)
        game = np.row_stack((game,X))
        t+=1
        if X[0] == TARGET or X[3] == TARGET:
            game_on = False
            game_df.loc[len(game_df)] = [int(game_realizations+1),int(t),int(wins_1),int(X[1]),int(X[2]),int(wins_2),int(X[4]),int(X[5]),np.nan,np.nan]
    
    plt.plot(game_df['n1']-game_df['n2'])
    #print("------------------")
    #print("Game over")
    #print(f"X = {X}")

    #print(f"Simulated Game outcome using {ORDER_TYPE} initialization")
    #print(game_df.to_latex(index=False))

plt.savefig(f"Lineplot {ORDER_TYPE} G={G}.pdf",dpi=500)

plt.figure()
game_df['S']=game_df['n1']-game_df['n2']
filtered_game_df = game_df[game_df['u1'].isna()]

counts,bins,_ = plt.hist(filtered_game_df['S'],bins=np.arange(-G,G+2,1),density=True)

np.var(game_df['S'])

np.sum(counts*(bins[:-1])**2)

len(counts)
len(bins)
np.var(counts*bins[:-1]) # to match dimension



np.var(counts,ddof=0)


print(counter/counter_global,"her")
print(f"Skewness: {stats.skew(counts)}")
plt.savefig(f"Histograms {ORDER_TYPE} G={G}.pdf",dpi=500)




# # # obs = np.array([])
# # # time = np.array([])
# # # plt.figure()
# # # for game_real in range(int(game_df['Game'].max())):
# # #     tmp_res = game_df[game_df['Game']==game_real+1]['S']
# # #     obs = np.append(obs,tmp_res)
# # #     time = np.append(time,game_df)
# # # plt.hist2d(time,obs,density=True)

# # # plt.savefig(f"{ORDER_TYPE}_2dhist.pdf",dpi=400)


obs = np.array([])
time = np.array([])
plt.figure()
for game_real in range(int(game_df['Game'].max())):
    tmp_res = game_df[game_df['Game']==game_real+1]['S']
    if tmp_res.shape[0] <10:
        tmp_res = np.append(tmp_res,np.zeros(10-tmp_res.shape[0]))
    obs = np.append(obs,tmp_res)
    time = np.append(time,np.arange(0,10,1))
bins,count_x,count_y,_ = plt.hist2d(time,obs,density=True,bins=[np.unique(time),np.arange(-G,G+2,1)])
plt.imshow(bins)
plt.savefig(f"2d Histograms {ORDER_TYPE} G={G}.pdf",dpi=500)


# plt.savefig(f"{ORDER_TYPE}_2dhist.pdf",dpi=400)
# NEW = pd.DataFrame({'t':game_df['t'],'S':(game_df['n1']-game_df['n2']).values})




# NEW.columns
# import matplotlib.pyplot as plt
# game_df['g1']
# plt.hist(game_df['b2'])
# plt.show()

# plt.hist2d(game_df['b1'], game_df['b2'])
# plt.show()


# #%%

# # All 16 combinations of g1,b1,g2 and b2 for 2x2 matrix
# #V[n1,g1,b1,n2,g2,b2] 
# V[:,1,1,:,1,1]
# V[:,1,1,:,1,0]
# V[:,1,1,:,0,1]
# V[:,1,0,:,1,1]
# V[:,0,1,:,1,1]
# V[:,1,1,:,0,0]
# V[:,1,0,:,1,0]
# V[:,0,1,:,1,0]
# V[:,1,0,:,0,1]
# V[:,0,1,:,0,1]
# V[:,0,0,:,1,1]
# V[:,1,0,:,0,0]
# V[:,0,1,:,0,0]
# V[:,0,0,:,1,0]
# V[:,0,0,:,0,1]
# A = V[:,0,0,:,0,0]
                        


