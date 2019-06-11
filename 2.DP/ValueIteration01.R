actions <- c("N", "S", "E", "W")

x <- 1:4
y <- 1:3

rewards <- matrix(rep(0, 12), nrow=3)
rewards[2, 2] <- NA
rewards[1, 4] <- 1
rewards[2, 4] <- -1

values <- rewards # initial values

states <- expand.grid(x=x, y=y)

# Transition probability
transition <- list("N" = c("N" = 0.8, "S" = 0, "E" = 0.1, "W" = 0.1), 
        "S"= c("S" = 0.8, "N" = 0, "E" = 0.1, "W" = 0.1),
        "E"= c("E" = 0.8, "W" = 0, "S" = 0.1, "N" = 0.1),
        "W"= c("W" = 0.8, "E" = 0, "S" = 0.1, "N" = 0.1))

# The value of an action (e.g. move north means y + 1)
action.values <- list("N" = c("x" = 0, "y" = 1), 
        "S" = c("x" = 0, "y" = -1),
        "E" = c("x" = -1, "y" = 0),
        "W" = c("x" = 1, "y" = 0))

# act() function serves to move the robot through states based on an action
act <- function(action, state) {
    action.value <- action.values[[action]]
    new.state <- state
    #
    if(state["x"] == 4 && state["y"] == 1 || (state["x"] == 4 && state["y"] == 2))
        return(state)
    #
    new.x = state["x"] + action.value["x"]
    new.y = state["y"] + action.value["y"]
    # Constrained by edge of grid
    new.state["x"] <- min(x[length(x)], max(x[1], new.x))
    new.state["y"] <- min(y[length(y)], max(y[1], new.y))
    #
    if(is.na(rewards[new.state["y"], new.state["x"]]))
        new.state <- state
    #
    return(new.state)
}


> rewards
     [,1] [,2] [,3] [,4]
[1,]    0    0    0    1
[2,]    0   NA    0   -1
[3,]    0    0    0    0