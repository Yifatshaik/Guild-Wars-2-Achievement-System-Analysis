# Guild-Wars-2-Achievement-System-Analysis
### Cupstone project in Brainstation Data Science course

This project analyzes the video game Guild Wars 2 achievement systems in order to determine which achievement players should pursue according to their play style and time commitment to the game. This is be done by looking at the rewards of each achievements and determining their importance based on their monetary value. 

Guild Wars 2 is a Massively multiplayer online role-playing game created by ArenaNet, unlike most games of that genre (World of Warcraft for example), the game tends to cater to a more casual players and focuses more on cooperative play with some single player campaign. Due to that fact, some of the game systems are design to dissuade competitive play, specifically (in this case) the achievement system is built in a way that the achievements are perceived as less important (not being a statue symbol), often do not give the players clear instructions and the rewards received are chosen randomly (from a pool of rewards). 

I hypothesize that this results in the players not utilizing the achievements system as effectively as they should, and by doing so they miss a crucial element which can improve their gameplay and enjoyment of the game.

Using supervised learning and clustering methods on the rewards given for completing achievements, I predict the financial worth of each achievement in game.

Specifically, I used the Linear Regression and Decision Trees Regression to create a model which will predict the importance of an achievement based on the vendor value of the reward players received from completing the achievement. 
I also used Forest Trees to classify those achievements according to categories, so it will be possible to predict the type of achievements which the players should pursue.


## The Data:
To get the information I used the Guild Wars 2 API (found here: [https://wiki.guildwars2.com/wiki/API:Main)]

For the analysis I used the following data:

Achievement Category: which gave me the information about the type of achievement it is 
and was used to cluster the data

Achievement Points: the amount of points players receive from each achievement, collecting 
those point allow the players to collect better rewards in the future

Reward Type: the type of reward the player gets for completing the achievement

Item Type: the type of item the player gets as a reward

Rarity: the rarity of the item the player gets

Level of Reward: the level of the reward (80 is the max level in the game)

Vendor Value: the sell value (in coins) of the reward item
