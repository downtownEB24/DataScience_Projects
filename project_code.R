library(plotly)
library(readxl)
passengerData<- read_excel("C:/Users/esbro/Desktop/STAT 484/Project/2020passengerData1.xlsx", 
                                    col_types = c("numeric", "text", "text", 
                                                  "text", "text", "text", "text", "text", 
                                                   "numeric", "numeric", "numeric", 
                                                   "skip"))
str(passengerData)
head(passengerData)

#airportPassengers=list(head(sort(passengerData$CY20_Enplanements,decreasing=TRUE,na.last=TRUE),
#                       n=10))
#list1<-vector(mode="list",length=0)

topAirports<-head(passengerData[order(passengerData$CY20_Enplanements,
                                      na.last = TRUE, decreasing = TRUE),c(6,9)],20)
text1<- c("Large Airport Size","Large Airport Size","Large Airport Size","Large Airport Size",
         "Large Airport Size","Large Airport Size","Large Airport Size","Large Airport Size",
         "Large Airport Size","Large Airport Size","Large Airport Size","Large Airport Size",
         "Large Airport Size","Large Airport Size","Large Airport Size","Large Airport Size",
         "Large Airport Size","Large Airport Size","Large Airport Size","Large Airport Size")
fig1<- plot_ly(
  x= topAirports$Airport_Name,
  y= topAirports$CY20_Enplanements,
  name= "Airport Passenger Data",
  type= "bar",
  text=text1,
  marker=list(color=c("red","blue","blue",
                      "blue","blue","blue",
                      "blue","blue","blue",
                      "blue")))%>% 
    layout(title = "Top 10 Business Airports in the US",
           xaxis = list(title = "Airport Name",
                        zeroline = FALSE),
           yaxis = list(title = "Number Passengers",
                        zeroline = FALSE))
fig1

topAirportsGrouped<-head(passengerData[order(passengerData$CY20_Enplanements,
                                      na.last = TRUE, decreasing = TRUE),c(6,9,10)],10)
fig2<- plot_ly(
  x= topAirportsGrouped$Airport_Name,
  y= topAirportsGrouped$CY20_Enplanements,
  name= "Airport Passenger Data 2020",
  type= "bar",
  marker=list(color=c("purple","purple","purple",
                      "purple","purple","purple",
                      "purple","purple","purple",
                      "purple")))%>% 
  add_trace(y=topAirportsGrouped$CY19_Enplanements,type="bar",
            name="Airport Passenger Data 2019",
            marker=list(color=c("orange","orange","orange",
                               "orange","orange","orange",
                               "orange","orange","orange",
                               "orange"),
                        line = list(color = 'rgb(8,48,107)', width = 1.5)))%>%
  
  layout(title = "Top 10 Business Airports in the US Grouped by Travel Year",
         xaxis = list(title = "Airport Name",
                      zeroline = FALSE),
         yaxis = list(title = "Number Passengers",
                      zeroline = FALSE),
         barmode="group")
fig2


newark_passengerData1 <- read_excel(
  "C:/Users/esbro/Desktop/STAT 484/Project/newark_passengerData1.xlsx")

newarkData<-newark_passengerData1[with(newark_passengerData1, 
                        order(newark_passengerData1$Year, 
                              newark_passengerData1$Month)),]
library(plotly)
fig3 <- plot_ly(data = newarkData, x = newarkData$Year, y = newarkData$TOTAL,
                color=newark_passengerData1$Month,
                type = 'scatter',
                mode="markers")%>%
  
  layout(title = "Newark Intl. Airport Total Passengers from 2002-2021",
         xaxis = list(title = "Year",
                      zeroline = FALSE),
         yaxis = list(title = "Number Passengers",
                      zeroline = FALSE)
        )

fig3
fig4 <- plot_ly(data = newarkData, x = newarkData$Year, y = newarkData$INTERNATIONAL,
                color=newark_passengerData1$Month,colors="Set2",
                type = 'scatter',
                mode="markers")%>%
  
  layout(title = "Newark Intl. Airport Total Passengers With
         International Flights from 2002-2021",
         xaxis = list(title = "Year",
                      zeroline = FALSE),
         yaxis = list(title = "Number Passengers",
                      zeroline = FALSE))
fig4
fig5 <- plot_ly(data = newarkData, x = newarkData$Year, y = newarkData$DOMESTIC,
                color=newark_passengerData1$Month,colors="Set1",
                type = 'scatter',
                mode="markers")%>%
  
  layout(title = "Newark Intl. Airport Total Passengers With
         Domestic Flights from 2002-2021",
         xaxis = list(title = "Year",
                      zeroline = FALSE),
         yaxis = list(title = "Number Passengers",
                      zeroline = FALSE))
fig5
#do not use code below
p7 <- plot_ly(x = names(table(cities)),
              y = as.numeric(table(cities)),
              name = "Cities",
              type = "bar",
              marker = list(color = c("rgba(150, 150, 150, 0.7)",
                                      "rgba(150, 150, 150, 0.7",
                                      "rgba(255, 20, 0, 0.7)",
                                      "rgba(150, 150, 150, 0.7"))) %>% 
  layout(title = "Number of offices per city",
         xaxis = list(title = "Cities",
                      zeroline = FALSE),
         yaxis = list(title = "Number",
                      zeroline = FALSE))
fig <- plot_ly(data, x = ~x, y = ~y, type = 'bar',
               marker = list(color = c('rgba(204,204,204,1)', 'rgba(222,45,38,0.8)',
                                       'rgba(204,204,204,1)', 'rgba(204,204,204,1)',
                                       'rgba(204,204,204,1)')))
fig <- fig %>% layout(title = "Least Used Features",
                      xaxis = list(title = ""),
                      yaxis = list(title = ""))
#end of do not use code
airportDelay<-read.csv("../Project/newarkAirportDelay_2019_2020.csv",
                       comm = "#", stringsAsFactors = TRUE)
library(plotly)
fig<-plot_ly(data= airportDelay, y= airportDelay$weather_delay, type = "box")
fig

airportDelayAll<-read.csv("../Project/allAirportDelay_2019_2020.csv",
                          comm = "#", stringsAsFactors = TRUE)
fig2<-plot_ly(data= airportDelay, y= airportDelayAll$weather_delay, type = "box",
             color=airportDelayAll$airport_name)%>% 
  
  layout(title = "Boxplot of Weather Delays at Top 20 Busiest Airport in US",
         xaxis = list(title = "Airport Name",
                      zeroline = FALSE),
         yaxis = list(title = "Number Weather Delays",
                      zeroline = FALSE))
fig2

airlinesDes<-read.csv("../Project/2021_MARKET_ALL_CARRIER.csv",
                    comm = "#", stringsAsFactors = FALSE)
unitedDes<-airlinesDes[airlinesDes$UNIQUE_CARRIER_NAME=="United Air Lines Inc."
                       & airlinesDes$MONTH==8,]
unitedDes<-unitedDes[order(-unitedDes$PASSENGERS),]
topUnitedDes<-head(unitedDes,100)

#library(plotly)
fig9 <- plot_ly(
  x=topUnitedDes$ORIGIN, y=topUnitedDes$DEST, 
  z=topUnitedDes$PASSENGERS,
    type = "heatmap"
)
fig9
