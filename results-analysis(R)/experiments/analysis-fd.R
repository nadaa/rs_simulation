library(hash)

# pre-experiments analysis
#compute means and variance coeffcients

#read the csv files for all strategies
setwd('C:\\Phd\\extension-current\\results\\sa\\iter2')


dps= c('1','2') #'010','011','100','101','110','111')

strategies = c('consumer-centric','profit-centric','consumer-biased','balanced', 'popular-uncorrelated-profit',
               'popular-correlated-profit')

h <- hash() 

#loop for time steps
#loop over the strategies
#group by step and run, and compute mean and variance coeffcient


for (dp in dps){
  i = 1
  for (strategy in strategies ){
    agent_file = paste(as.character(dp), "agents-data-scenario", sep = "/")
    agent_file = paste(agent_file,as.character(i),'.csv',sep='') 
    agent_data = read.csv(agent_file)
    agent_data = subset(agent_data,step==1000)
    
    #model data
    model_file = paste(as.character(dp), "model-data-scenario", sep = "/")
    model_file = paste(model_file,as.character(i),'.csv',sep='') 
    model_data = read.csv(model_file)  
    
    #compute the mean of total profit, group by step
    # compute the cummulative profit
    
    
    model_data = aggregate(total_profit~step,model_data,mean)
    
    model_data$cumsum_profit = cumsum(model_data$total_profit)

    model_data = subset(model_data,step==1000)
    
    
    #....
    
    avg_trust = mean(agent_data$trust) 
  
   
    avg_consumption_prob =mean(agent_data$consumption_probability)

    
    avg_total_profit= model_data$total_profit
    avg_cum_profit = model_data$cumsum_profit


    # store the result of each step in a dictionary  
    h[[strategy]]=list(avg_trust, avg_consumption_prob,avg_total_profit, avg_cum_profit)  
      
    i= i+1
      
  }
  
  #print the results of the current step
  print(paste('Results of design point= ',as.character(dp)))
  print('======================================')
  print(h)
}

