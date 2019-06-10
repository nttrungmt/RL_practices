def eps_greedy(q_vals, eps, state): # random action with probability of eps; argmax Q(s, .) with probability of (1-eps)
    import random
    if random.random() < eps:
        action = np.random.choice(len(q_vals[state])) # randomly select action from state
    else:
        action = np.argmax(q_vals[state]) # greedily select action from state
    return action

def q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward): # implement one step of Q-learning
    target = reward + gamma * np.max(q_vals[next_state]) # use the next state's maximum Q value
    q_vals[cur_state][action] = (1 - alpha) * q_vals[cur_state][action] + alpha * target