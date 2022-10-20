
#reading in dataset from excel workout
library(readxl)
#excel workbook has multiple sheets in it that contain each of the four datasets used for this assignment
#each worksheet has a different name
sheet_names <- excel_sheets("C:/Users/esbro/Desktop/IE 575/Week 5/Anscombe.xlsx")  
#generic function to get each dataset from each sheet in the workbook excel file
anscomData <- lapply(sheet_names, function(x) {          # Read all sheets to list
  as.data.frame(read_excel("C:/Users/esbro/Desktop/IE 575/Week 5/Anscombe.xlsx", sheet = x)) } )
anscomData
names(anscomData)<-sheet_names #changing the default names for each dataset back to their orignial sheet name

#Dataset 1 - calculations
anscomData$Data1
#mean calculations
Dat1x_mn<-mean(anscomData$Data1$x)
Dat1x_mn
Dat1y_mn<-mean(anscomData$Data1$y)
Dat1y_mn
#variance calculations
Dat1x_var<-var(anscomData$Data1$x)
Dat1x_var
Dat1y_var<-var(anscomData$Data1$y)
Dat1y_var
#correlation calculations
Dat1_corr<-cor(anscomData$Data1$x,anscomData$Data1$y)
Dat1_corr
#creating linear regression model for dataset 1
Dat1_lreg<-lm(Data1$y~Data1$x,anscomData)
summary(Dat1_lreg)
#visualizations for dataset 1
plot(anscomData$Data1$x,anscomData$Data1$y,main="Data1 - Fitted Linear Regression")
abline(Dat1_lreg,col=10)
par(mfrow=c(2,2))
#residual analysis for dataset 1 - fitted regression model
plot(Dat1_lreg, main="Residual Analysis of DataSet1")
par(mfrow=c(1,1))

#Dataset 2 - calculations
anscomData$Data2
#mean calculations
Dat2x_mn<-mean(anscomData$Data2$x)
Dat2x_mn
Dat2y_mn<-mean(anscomData$Data2$y)
Dat2y_mn
#variance calculations
Dat2x_var<-var(anscomData$Data2$x)
Dat2x_var
Dat2y_var<-var(anscomData$Data2$y)
Dat2y_var
#correlation calculations
Dat2_corr<-cor(anscomData$Data2$x,anscomData$Data2$y)
Dat2_corr
#creating linear regression model for dataset 2
Dat2_lreg<-lm(Data2$y~Data2$x,anscomData)
summary(Dat2_lreg)
#visualizations for dataset 2
plot(anscomData$Data2$x,anscomData$Data2$y,main="Data2 - Fitted Linear Regression")
abline(Dat2_lreg,col=10)
par(mfrow=c(2,2))
#residual analysis for dataset 2 - fitted regression model
plot(Dat2_lreg,main="Residual Analysis of DataSet2")
par(mfrow=c(1,1))

#Dataset 3 - calculations
anscomData$Data3
##mean calculations
Dat3x_mn<-mean(anscomData$Data3$x)
Dat3x_mn
Dat3y_mn<-mean(anscomData$Data3$y)
Dat3y_mn
#variance calculations
Dat3x_var<-var(anscomData$Data3$x)
Dat3x_var
Dat3y_var<-var(anscomData$Data3$y)
Dat3y_var
#correlation calculations
Dat3_corr<-cor(anscomData$Data3$x,anscomData$Data3$y)
Dat3_corr
#creating linear regression model for dataset 3
Dat3_lreg<-lm(Data3$y~Data3$x,anscomData)
summary(Dat3_lreg)
#visualizations for dataset 3
plot(anscomData$Data3$x,anscomData$Data3$y,main="Data3 - Fitted Linear Regression")
abline(Dat3_lreg,col=10)
par(mfrow=c(2,2))
#residual analysis for dataset 3 - fitted regression model
plot(Dat3_lreg,main="Residual Analysis of DataSet3")
par(mfrow=c(1,1))

#Dataset 4 - calculations
anscomData$Data4
##mean calculations
Dat4x_mn<-mean(anscomData$Data4$x)
Dat4x_mn
Dat4y_mn<-mean(anscomData$Data4$y)
Dat4y_mn
#variance calculations
Dat4x_var<-var(anscomData$Data4$x)
Dat4x_var
Dat4y_var<-var(anscomData$Data4$y)
Dat4y_var
#correlation calculations
Dat4_corr<-cor(anscomData$Data4$x,anscomData$Data4$y)
Dat4_corr
#creating linear regression model for dataset 4
Dat4_lreg<-lm(Data4$y~Data4$x,anscomData)
summary(Dat4_lreg)
#visualizations for dataset 4
plot(anscomData$Data4$x,anscomData$Data4$y,main="Data4 - Fitted Linear Regression")
abline(Dat4_lreg,col=10)
par(mfrow=c(2,2))
#residual analysis for dataset 4 - fitted regression model
plot(Dat4_lreg,main="Residual Analysis of DataSet 4")
par(mfrow=c(1,1))
