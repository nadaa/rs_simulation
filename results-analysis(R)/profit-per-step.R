
plot_profit_per_step = function(df,colors){
  
    p = ggplot(df,aes(x=step,y = total_profit,col=strategy,shape=strategy),size=1)+
      geom_line()+
      geom_smooth(aes(group=strategy, color = factor(strategy)))+
      labs(x='Time steps',y='Profit per time step')+
      geom_ribbon(aes(ymin=total_profit-error,ymax=total_profit+error,fill=strategy),alpha=0.1)+
      scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) +
      scale_y_continuous(limits = c(100, 2100), breaks = seq(100, 2100, by = 200))+
      
      theme(legend.position = "top",
        legend.title = element_text(size=24),
        legend.text = element_text(size=20),
        axis.title = element_text(size=24),
        axis.text = element_text(size=20,color="black"),
        
        panel.grid.major.y = element_line(colour = "gray50", linetype="dashed"),
        
        panel.border = element_rect(colour = "black", fill=NA),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black")
      )+
      scale_color_manual(values = colors)+
      scale_fill_manual(values = colors)+
      theme(legend.position="none")

      #guides(color=guide_legend(nrow=3, byrow=TRUE))
     
      
      
    return (p)
    
  }
  



compute_CI_error = function(data){
  sd_data = aggregate(total_profit~step+strategy,data,sd)
  error = qt(0.975, df=nrow(data)-1)*sd_data$total_profit/sqrt(nrow(sd_data))
  return(error)
}


aggregated_data = function(d, socr,exp,rec_s){
  avg_profit_data = aggregate(total_profit~step+strategy,d,mean)
  # compute the margin error of the 95% CI
  error = compute_CI_error(d)
  avg_profit_data$strategy = rec_s
  avg_profit_data$reliance = socr
  avg_profit_data$expectation = exp
  avg_profit_data$error = error
  return(avg_profit_data)
}



process_profit_per_step= function(d,socr,exp,colors){
  #aggregate all data seperately
  d1 = aggregated_data(data.frame(d[1]),socr,exp,"consumer-centric")
  d2 = aggregated_data(data.frame(d[2]),socr,exp,"profit-centric")
  d3 = aggregated_data(data.frame(d[3]),socr,exp,"consumer-biased")
  d4 = aggregated_data(data.frame(d[4]),socr,exp,"balanced")
  d5 = aggregated_data(data.frame(d[5]),socr,exp,"popular-uncorrelated-profit")
  d6 = aggregated_data(data.frame(d[6]),socr,exp,"popular-correlated-profit")

  df = do.call("rbind",list(d1,d2,d3,d4,d5,d6))
  
  #df = do.call("rbind",list(d1,d2,d3,d4,d5))
  #p = plot(df,colors)
  return(df)
  
  # save plot
  # png(filename= file.path(newdir,"time-totalprofitplot.png"))
  # print(plot(df))
  # dev.off()
  
}


