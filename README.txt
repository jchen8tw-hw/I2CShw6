I got the endgame state data set from this website
https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
And what I did is get all the state that x wins and o wins
for the data x wins, I take one of the x out and put it in input. This is for attacking.
for the data o wins, I take one of the y out and one of the x out, put x in the y position, this is for defending.
All of the input x has been check if it is valid(no lines on board)