# Guild-Wars-2-Achievement-System-Analysis
### Cupstone project in Brainstation Data Science course

This project analyzes the video game Guild Wars 2 achievement systems in order to determine which achievement players should pursue according to their play style and time commitment to the game. This is be done by looking at the rewards of each achievements and determining their importance based on their monetary value. 

Guild Wars 2 is a Massively multiplayer online role-playing game created by ArenaNet, unlike most games of that genre (World of Warcraft for example), the game tends to cater to a more casual players and focuses more on cooperative play with some single player campaign. Due to that fact, some of the game systems are design to dissuade competitive play, specifically (in this case) the achievement system is built in a way that the achievements are perceived as less important (not being a statue symbol), often do not give the players clear instructions and the rewards received are chosen randomly (from a pool of rewards). 

I hypothesize that this results in the players not utilizing the achievements system as effectively as they should, and by doing so they miss a crucial element which can improve their gameplay and enjoyment of the game.

Using supervised learning and clustering methods on the rewards given for completing achievements, I predict the financial worth of each achievement in game.

Specifically, I used the Linear Regression and Decision Trees Regression to create a model which will predict the importance of an achievement based on the vendor value of the reward players received from completing the achievement. 
I also used Forest Trees to classify those achievements according to categories, so it will be possible to predict the type of achievements which the players should pursue.


### The Data:
To get the information I used the Guild Wars 2 API (found here: https://wiki.guildwars2.com/wiki/API:Main)

For the analysis I used the following data:

**Achievement Category**: which gave me the information about the type of achievement it is 
and was used to cluster the data

**Achievement Points**: the amount of points players receive from each achievement, collecting 
those point allow the players to collect better rewards in the future

**Reward Type**: the type of reward the player gets for completing the achievement

**Item Type**: the type of item the player gets as a reward

**Rarity**: the rarity of the item the player gets

**Level of Reward**: the level of the reward (80 is the max level in the game)

**Vendor Value**: the sell value (in coins) of the reward item

### Cleaning the data

For this project I needed to combine the achievement and item data into one dataframe. This was done by combining the achievement data with the achievement/group data, and then combining achievements with rewards and the item lists. This gave me a small dataset, but it was enough for this exercise. 
The final dataset looked like this:   
![datafarme](http://yifatshaik.com/img/table.png)
As a note, all data cleaning was done in Python with the exception of a some data cleaning in Open Refine 

### Linear Regression

The first machine learning model I ran on the information was a linear regression. The independent variables were: Rarity, Level of Reward, Reward Type, Item Type and Achievement Points while the dependent value was the Vendor Value. Due to the way the data was distributed the overall prediction score was too low to get any real value of the prediction

**Linear Regression of the rarity of the reward items**
![LinearRegression1](http://yifatshaik.com/img/reg0.png)

**Linear Regression of the level of the reward item**
![LinearRegression2](http://yifatshaik.com/img/reg1.png)

**Linear Regression of the type of rewards**
![LinearRegression3](http://yifatshaik.com/img/reg2.png)

### Final Words 



**Linear Regression of the item types**
![LinearRegression4](http://yifatshaik.com/img/reg3.png)

**Linear Regression of the achievement points**
![LinearRegression5](http://yifatshaik.com/img/reg4.png)


### Decision Trees Regressor

In order to get a better predication then the linear regression I used decision trees regressor on the same data. This ended up being the right idea as the decision trees score was 97% and allowed me to get some more accurate predictions. 

**Decision Trees Progression with 3 depth**
![DecisionTree](http://yifatshaik.com/img/tree.png)

**Feature importance table**
![FeatureImportance](http://yifatshaik.com/img/importanceDecision.png)

### Decision Forest Classification 

The last model I ran was a decision forest classification, organizing the data I have versus the achievement categories. This was made to give me an idea about the length and difficulty of the achievement as the categories give me some inclination on the type of achievement it is and what will it take to complete it. So, for example, a daily achievement will be fairly-short and easy to get. 
This was used as a mean to let the player know if an achievement is worth perusing- so if an achievement takes a long time but the reward isn’t great it might not be worth trying to accomplish it and vice versa. 

**Decision Forest Classification Heatmap**
![DecisionTreeCalssification](http://yifatshaik.com/img/heatMap.png)

**Feature importance table**
![FeatureImportance](http://yifatshaik.com/img/importanceForest.png)

### Final Words

This is far from a perfect model as the vendor value of items is not the best indicator for the worth of an achievement in Guild Wars 2. While this prediction can be used, it should be a smaller part of a much larger projects that extensively uses the Achievement Points (as collecting them rewards the players with bigger and more powerful rewards), reward items stats and statue markers like titles and special skins.
