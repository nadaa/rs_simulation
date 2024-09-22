library(hash)

# pre-experiments analysis
#compute means and variance coeffcients

#read the csv files for all strategies
setwd('C:\\Phd\\code-last\\results\\preexperiments\\find-runs')


runs= c(3,5,7)
strategies = c('consumer-centric','profit-centric','consumer-biased','balanced', 'popular-uncorrelated-profit',
               'popular-correlated-profit')

h <- hash() 

#loop for time steps
#loop over the strategies
#group by step and run, and compute mean and variance coeffcient


for (s in runs){
  i = 1
  for (strategy in strategies ){
    agent_file = paste(as.character(s), "agents-data-scenario", sep = "/")
    agent_file = paste(agent_file,as.character(i),'.csv',sep='') 
    agent_data = read.csv(agent_file)
    agent_data = subset(agent_data,step==1000)
    
    #model data
    model_file = paste(as.character(s), "model-data-scenario", sep = "/")
    model_file = paste(model_file,as.character(i),'.csv',sep='') 
    model_data = read.csv(model_file)    
    model_data = subset(model_data,step==s)
    
    #....
    
    avg_trust = mean(agent_data$trust) 
    cv_trust = round(sd(agent_data$trust)/avg_trust,3)
   


    
    avg_consumption_prob =mean(agent_data$consumption_probability)
    cv_consumption_prob = round(sd(agent_data$consumption_probability)/avg_consumption_prob,3)
    
    
    
    
    avg_total_profit= mean(model_data$total_profit)
    cv_total_profit = round(sd(model_data$total_profit)/avg_total_profit,3)
    

    # store the result of each step in a dictionary  
    h[[strategy]]=list(c(avg_trust,cv_trust),c(avg_consumption_prob,cv_consumption_prob),c(avg_total_profit,cv_total_profit))  
      
    i= i+1
      
  }
  
  #print the results of the current step
  print(paste('Results of step= ',as.character(s)))
  print('======================================')
  print(h)
}

