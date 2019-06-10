def value_iteration(mdp, gamma, nIt):
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        Vprev = Vs[-1] # V^{(it)}
        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS)
        for state in mdp.P: # for all the states in the finite MDP
            maxv = 0 # track the max value across all the actions in the current state
            for action in mdp.P[state]: # for all the actions in current state
                v = 0
                for probability, nextstate, reward in mdp.P[state][action]:
                    v += probability * (reward + gamma * Vprev[nextstate]) # update the value
                if v > maxv: # if value is largest across all the actions, set the policy to that action
                    maxv = v
                    pi[state] = action
            V[state] = maxv # set the value function to the max value across all the actions
        Vs.append(V)
        pis.append(pi)
    return Vs, pis