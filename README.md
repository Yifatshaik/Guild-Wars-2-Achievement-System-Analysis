# Guild-Wars-2-Achievement-System-Analysis
Cupstone project in Brainstation Data Science course

Guild Wars 2 is a Massively multiplayer online role-playing game created by ArenaNet

This project analyzes the game achievement systems in order to determine which achievement players 
should pursue according to their play style and time commitment to the game.
This as done by looking at the rewards of each achievements and determining their importance based on their 
monetary value, the strength and the rarity of the items received.

Using supervised learning and clustering methods on the rewards given for
completing achievements I predict the financial worth of each achievement in game.
Specifically, I used the Linear Regression and Decision Trees Regression to create a model which will predict 
the importance of an achievement based on the vendor value of the reward players received from completing
the achievement. I also used Forest Trees to classify those achievements according to categories, so it will be possible to predict
the type of achievements which the players should pursue.

The Data:
To get the information I used the Guild Wars 2 API (found here: https://wiki.guildwars2.
com/wiki/API:Main)

For the analysis I used the following data:
Achievement Category: which gave me the information about the type of achievement it is and was used to cluster the data

Achievement Points: the amount of points players receive from each achievement, collecting those point allow the players to
collect better rewards in the future

Reward Type: the type of reward the player gets for completing the achievement

Item Type: the type of item the player gets as a reward

Rarity: the rarity of the item the player gets

Level of Reward: the level of the reward (80 is the max level in the game)

Vendor Value: the sell value (in coins) of the reward item
