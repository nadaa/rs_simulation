# install.packages("AlgDesign")
library(AlgDesign)

setwd('C:\\Phd\\extension-current\\results-analysis(R)\\experiments\\factorial design\\iter1_analysis_variance')

data = read.csv('response-s6.csv')

data$noise = factor(data$noise)
data$p_soc = factor(data$p_soc)
data$volume = factor(data$volume)


# aov1.out = aov(trust ~ ., data=data)
# summary(aov1.out)

#aov2.out = aov(trust+consum+profit_step+profit_cum ~ .^2, data=data)

aov2.out = aov(cbind(trust,consum,profit_step,profit_cum)  ~ .^2, data=data)
summary(aov2.out)
# 
# # This provides all of the means commonly calculated in ANOVA analysis
# model.tables(aov2.out, type="means", se=T)