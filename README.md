# Predicting Voting Propensity and Political Party Affiliation in Primary Elections
## Rodrigo Pimentel<sup>a,b</sup> and Zachary deWardener<sup>c</sup>
<sup>a</sup>Department of Psychology, University of Rhode Island; <sup>b</sup>Department of Computer Science and Statistics, University of Rhode Island; <sup>c</sup>Department of Mechanical, Industrial and Systems Engineering, University of Rhode Island

![Title Page for Research Paper found in the report folder](https://user-images.githubusercontent.com/4823699/208801683-dbb47a69-757c-41ba-9c03-842c6f6bda80.png)


### Problem Definition

This project aims to provide a supervised machine learning model that can take historical voter data as an input and output the probability that an individual will vote in the upcoming election. Along with this, the model should be able to predict the most probable party designation of unaffiliated voters prior to polling. These metrics will, in theory, help campaign managers to more efficiently direct their efforts towards likely voters. 

### Data 

The dataset used is publicly available statewide voter registration data for the State of Rhode Island from the Rhode Island Secretary of State over the course of eight elections from 2018 to 2022 consisting of statewide primary/general, presidential preference primary, statewide referenda, and presidential elections. Along with specific elections, voter ID, party affiliation, year of birth, and locational information is also provided. A total of 816,279 registered voters are in the Statewide dataset. 

### Experiments


#### Experiment 1
For this experiment, the focus was on utilizing scikit-learn classifiers, including logistic regression, decision trees, random forests, and bagging estimators. The best random forest model performed better than all of our baseline classifiers in regards to the average_precision_score(). In addition, the model also outperformed all of the other models tested in experiment 1 with an AUC-PR of 0.6635.

#### Experiment 2
For this experiment, the focus was on ANN-based binary classification models for propensity and likely-party affiliations. The datasets used are the same as previous experiments with the exception of MinMaxScaler being applied to propensity models and strictly RandomUnderSampling to party affiliation models. The party affiliation models achieved an average precision of 0.96 and 0.97 for the South Kingstown and Statewide datasets respectively. The party affiliation models perform very well whereas the remaining propensity models performed roughly on-par with the models used in Experiment 1. 

### Deployed Voter Propensity Model: https://huggingface.co/spaces/rodrigopimentel/predicting-voting-propensity 

