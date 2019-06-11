
bellman.update <- function(action, state, values, gamma=1) {
    state.transition.prob <- transition[[action]]
    q <- rep(0, length(state.transition.prob))
    for(i in 1:length(state.transition.prob)) {        
        new.state <- act(names(state.transition.prob)[i], state) 
        q[i] <- (state.transition.prob[i] * (rewards[state["y"], state["x"]] + (gamma * values[new.state["y"], new.state["x"]])))
    }
    sum(q)
}

value.iteration <- function(states, actions, rewards, values, gamma, niter) {
    for (j in 1:niter) {
        for (i in 1:nrow(states)) {
            state <- unlist(states[i,])
            if(i %in% c(4, 8)) next # terminal states
            q.values <- as.numeric(lapply(actions, bellman.update, state=state, values=values, gamma=gamma))
            values[state["y"], state["x"]] <- max(q.values)
        }
    }
    return(values)
}

final.values <- value.iteration(states=states, actions=actions, rewards=rewards, values=values, gamma=0.99, niter=100)

> final.values
          [,1]      [,2]      [,3]     [,4]
[1,] 0.9516605 0.9651596 0.9773460  1.00000
[2,] 0.9397944        NA 0.8948359 -1.00000
[3,] 0.9266500 0.9150957 0.9027132  0.81989
