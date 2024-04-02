library(ggplot2)
#library(gridExtra)

#colors = c("consumer-centric"="cornflowerblue","profit-centric"= "black","consumer-biased"="mediumorchid1","balanced"="red","popular-uncorrelated-profit"="mediumseagreen","popular-correlated-profit"="darkorange2")



plot_drop = function(df,colors){
  ggplot(df,aes(x=step))+
    geom_line(aes(y = number_of_dropout,color=strategy),size=1)+
    labs(x='Time',y='# consumer dropout')+
    scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) +
    scale_y_continuous(limits = c(0, 600), breaks = seq(0, 600, by = 100))+
    
    geom_ribbon(aes(ymin=number_of_dropout-error,ymax=number_of_dropout+error,fill=strategy),alpha=0.1)+
    #theme_classic()+
    ylim(0,600)+
    theme(legend.position = "top",
          legend.title = element_text(size=24),
          legend.text = element_text(size=20),
          axis.title = element_text(size=24),
          axis.text = element_text(size=20,color="black"),
          #axis.text.x = element_text(angle = 90),
          panel.grid.major.y = element_line(colour = "gray50", linetype="dashed"),
          
          panel.border = element_rect(colour = "black", fill=NA),
          panel.background = element_blank(), 
          
          axis.line = element_line(colour = "black"),
    )+
    scale_color_manual(values = colors)+
    scale_fill_manual(values = colors)+
    theme(legend.position="none")
  
}


compute_CI_error = function(data){
  sd_data = aggregate(number_of_dropout~step+strategy,data,sd)
  error = qt(0.975, df=nrow(data)-1)*sd_data$number_of_dropout/sqrt(nrow(sd_data))
  return(error)
}


aggregated_data = function(d,socr,exp, rec_s){
  avg_drop_data = aggregate(number_of_dropout~step+strategy,d,mean)

  
  error = compute_CI_error(d)
  avg_drop_data$error = error
  avg_drop_data$reliance = socr
  avg_drop_data$expectation = exp
  avg_drop_data$strategy = rec_s
  return(avg_drop_data)
}



process_consumerdropout = function(d,socr,exp,colors){
  #aggregate all data seperately
  d1 = aggregated_data(data.frame(d[1]),socr,exp,"consumer-centric")
  d2 = aggregated_data(data.frame(d[2]),socr,exp,"Profit-centric")
  d3 = aggregated_data(data.frame(d[3]),socr,exp,"consumer-biased")
  d4 = aggregated_data(data.frame(d[4]),socr,exp,"balanced")
  d5 = aggregated_data(data.frame(d[5]),socr,exp,"popular-uncorrelated-profit")
  d6 = aggregated_data(data.frame(d[6]),socr,exp,"popular-correlated-profit")
  
  df = do.call("rbind",list(d1,d2,d3,d4,d5,d6))
 return(df)
  
  
}






