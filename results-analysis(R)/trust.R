
windowsFonts(A=windowsFont("Times New Roman"))

plot_trust = function(df,colors){
  p = ggplot(df,aes(step,trust,col=strategy),size=1)+
    geom_line(size=1)+
    geom_ribbon(aes(ymin=trust-error,ymax=trust+error,fill=strategy),alpha=0.1)+
    scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) +
    scale_y_continuous(limits = c(0.2,1), breaks = seq(0.2,1, by=0.1)) +
    labs(x='Time steps',y='Trust')+
    
    theme(legend.position = "none",
          text=element_text(size=16,family="A"),
          legend.title = element_text(size=24),
          legend.text = element_text(size=20),
          axis.title = element_text(size=24),
          axis.text = element_text(size=20,color="black"),
          #axis.text.x = element_text(angle = 90),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_line(colour = "gray50", linetype="dashed"),
          
          panel.background = element_blank(), 
          panel.border = element_rect(colour = "black", fill=NA),
          
          axis.line = element_line(colour = "black"),
          
          #legend.box="vertical",
          #legend.margin=margin()
    )+
    scale_color_manual(values = colors)+
    scale_fill_manual(values = colors)+
    guides(color=guide_legend(nrow=3, byrow=TRUE))
    return(p)
  }
  

plot_trust_expectation = function(df,colors){
  
  p =  ggplot(df,aes(step,trust,col=strategy),size=1)+
    geom_line(size=1)+
    labs(x='Timesteps',y='Trust')+
    geom_ribbon(aes(ymin=trust-error,ymax=trust+error,fill=strategy),alpha=0.3)+
    
    scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    
    scale_fill_manual(values = colors)+
    scale_color_manual(values = colors)+
    
    theme(legend.position = "bottom",
          legend.title = element_text(size=20),
          legend.text = element_text(size=18),
          axis.title = element_text(size=18),
          axis.text = element_text(size=15,color="black"),
          #axis.text.x = element_text(angle = 90),
          
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_line(colour = "gray50", linetype="dashed"),
          
          panel.background = element_blank(), 
          
          axis.line = element_line(colour = "black"),
          strip.text.x = element_text(size=17)
    )+
    facet_grid(~expectation)
  
  return(p)
  
}

compute_CI_error = function(d,t){
  sd_data = aggregate(trust~step+strategy,d,sd)
  error = qt(0.975, df=nrow( sd_data)-1)*sd_data$trust/sqrt(nrow(sd_data))
  return(error)
}


aggregated_data = function(d,exp,rec_s){
  avg_trust_data = aggregate(trust~step+strategy,d,mean)
  # compute the margin error of the 95% CI
  error = compute_CI_error(d)
  avg_trust_data$error = error
  avg_trust_data$expectation = exp
  avg_trust_data$strategy = rec_s
  return(avg_trust_data)
}


process_trust = function(d,exp,colors){
  #aggregate all data seperately
 d1 = aggregated_data(data.frame(d[1]),exp,"consumer-centric")
  d2 = aggregated_data(data.frame(d[2]),exp,"Profit-centric")
  d3 = aggregated_data(data.frame(d[3]),exp,"consumer-biased")
  d4 = aggregated_data(data.frame(d[4]),exp,"balanced")
  d5 = aggregated_data(data.frame(d[5]),exp,"popular-uncorrelated-profit")
  d6 = aggregated_data(data.frame(d[6]),exp,"popular-correlated-profit")

  df = do.call("rbind",list(d1,d2,d3,d4,d5,d6))
  
  #p = plot(df,colors)
  
  return(df)
  # save plot
  # png(filename= file.path(newdir,"time-trustplot.png"))
  # print(plot(df))
  # dev.off()
  
}