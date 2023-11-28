import numpy as np
import nashpy as nash

TARGET = 5
N = TARGET+1
G = 4
B = 6

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
P1_win = np.array([[0.5,0.2,0.1],[0.8,0.5,0.2],[0.9,0.8,0.5]])
# 4x4 example
# =============================================================================
p0 = 0.5
p1 = 0.6
p2 = 0.8
p3 = 0.9
P1_win = np.array([[p0, 1-p1, 1-p2, 1-p3],
                    [p1, p0, 1-p1, 1-p2],
                    [p2, p1, p0, 1-p1],
                    [p3, p2, p1, p0]
                    ])
# =============================================================================


# 5x5 example
# =============================================================================
# P1_win = np.array([[0.50, 0.55, 0.70, 0.75, 0.90],
#                    [0.55, 0.50, 0.55, 0.70, 0.75],
#                    [0.70, 0.55, 0.50, 0.55, 0.70],
#                    [0.75, 0.70, 0.55, 0.50, 0.55],
#                    [0.90, 0.75, 0.70, 0.55, 0.50]
#                    ])
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
                        strategy = nash.Game(payoff_p1, 1-payoff_p1)
                        eqs = strategy.support_enumeration() 
                        res = list(eqs)
                        V[n1,g1,b1,n2,g2,b2] = res[0][0] @ payoff_p1 @ res[0][1]


#%%
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

# Given a state x
x = [0,2,2,0,0,2]
# We can find the optimal policy/desicion using the 
policy,U1,U2,payoff_p1 = get_optimal_policy(x)
print(policy)

#%% Simulate full dynamic game:

print("Simulation of a Game")
print("------------------")

# Initial state
X = [0,0,1,0,0,1]

# Array to visialize game
game = X

game_on = True
while game_on:
    print(f"X = {X}")
    print(f"Score: n1={X[0]} and n2={X[3]}")
    
    # Take optimal desicion
    policy,U1,U2,payoff_p1 = get_optimal_policy(X)
    
    u1 = np.random.choice(list(range(U1+1)),p=policy[0][0])
    u1_vec = np.zeros(G)
    u1_vec[u1] = 1
    print(f"Player 1: Gear={X[1]}, Bank={X[2]} | decision: u1 = {u1}")
    
    u2 = np.random.choice(list(range(U2+1)),p=policy[0][1])
    u2_vec = np.zeros(G)
    u2_vec[u2] = 1
    print(f"Player 2: Gear={X[4]}, Bank={X[5]} | decision: u2 = {u2}")
    
    U = np.array([u1, u2])
    
    # Simulate outcome
    p = u1_vec @ P1_win @ u2_vec
    w = np.random.choice([1, 0], p = [p, 1-p])
    print(f"Did player 1 win: w = {w}")
    
    # Advance the system to next state
    X = fmap(X,U,w)
    
    game = np.row_stack((game,X))
    
    if X[0] == TARGET or X[3] == TARGET:
        game_on = False
print("------------------")
print("Game over")
print(f"X = {X}")

#%%

# All 16 combinations of g1,b1,g2 and b2 for 2x2 matrix
#V[n1,g1,b1,n2,g2,b2] 
V[:,1,1,:,1,1]
V[:,1,1,:,1,0]
V[:,1,1,:,0,1]
V[:,1,0,:,1,1]
V[:,0,1,:,1,1]
V[:,1,1,:,0,0]
V[:,1,0,:,1,0]
V[:,0,1,:,1,0]
V[:,1,0,:,0,1]
V[:,0,1,:,0,1]
V[:,0,0,:,1,1]
V[:,1,0,:,0,0]
V[:,0,1,:,0,0]
V[:,0,0,:,1,0]
V[:,0,0,:,0,1]
A = V[:,0,0,:,0,0]
                        


