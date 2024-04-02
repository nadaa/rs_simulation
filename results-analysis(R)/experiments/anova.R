# install.packages("AlgDesign")
library(AlgDesign)

setwd('c:\\Phd\\code-last\\results\\doe-tables')

data = read.csv('responses\\responses-s1.csv')

data$noise = factor(data$noise)
data$prob_sm = factor(data$prob_sm)

# aov1.out = aov(trust ~ ., data=data)
# summary(aov1.out)

aov2.out = aov(trust+consum+profit+profit_cum ~ .^2, data=data)
summary(aov2.out)
# 
# # This provides all of the means commonly calculated in ANOVA analysis
# model.tables(aov2.out, type="means", se=T)