#need this library to use read_excel import function
library(readxl)
#importing dataset from excel workbook
finalProject_data <- read_excel("C:/Users/esbro/Desktop/STAT 485/Project/FinalProject_data.xlsx", 
                               col_types = c("numeric", "numeric", "numeric", 
                                             "numeric", "numeric", "numeric", 
                                             "numeric", "numeric", "text", "text"))
#need to factor both the Level of Depression & Covid Year
#since they values no do not represent count values 
finalProject_data$Level_ofDepression1_3<-as.factor(finalProject_data$Level_ofDepression1_3)#scale variable 1-3
finalProject_data$Covid_year<-as.factor(finalProject_data$Covid_year) #binary variable 0 or 1
#getting summary of data values in the project dataset 
#checking to make sure data values are consistent and not skewed
summary(finalProject_data)
#checking the structure of the dataset, including variable types
str(finalProject_data)
#making sure no data values are missing before creating our ANCOVA model
sum(is.na(finalProject_data))

#install.packages("rstatix", repos = "https://cloud.r-project.org")
#rstatix provides a pipe-friendly framework, coherent with the 'tidyverse' design philosophy, for performing basic statistical tests
library(rstatix)
# summary statistics for dependent variable depression based on the level of depression faced by people each year 
finalProject_data %>% group_by(Level_ofDepression1_3) %>% 
  get_summary_stats(Depressed_Yes, type="common")

finalProject_data %>% group_by(Covid_year) %>% 
  get_summary_stats(Trouble_concentratingOnThings, type="common")

#more summary statistics involving factor variable level of depression faced among people each year
library(ggplot2)
ggplot(finalProject_data, aes(x = Level_ofDepression1_3, 
                                      y = Depressed_Yes, col = Level_ofDepression1_3)) + 
  geom_boxplot(outlier.shape = NA) + geom_jitter(width = 0.02) + theme(legend.position="top")

#Checking to make sure ANCOVA assumptions are met
#linearity assumption
#the relationship between the covariates and at each group of the level of depression variable should be linear
ggplot(finalProject_data, aes(TroubleSleeping_OrsleepingTooMuch, 
                              Depressed_Yes, colour = Level_ofDepression1_3)) + geom_point(size = 3) + 
  geom_smooth(method = "lm", aes(fill = Level_ofDepression1_3), alpha = 0.1) + theme(legend.position="top")

ggplot(finalProject_data, aes(FeelingTired_OrhavingLittleEnergy, 
                              Depressed_Yes, colour = Level_ofDepression1_3)) + geom_point(size = 3) + 
  geom_smooth(method = "lm", aes(fill = Level_ofDepression1_3), alpha = 0.1) + theme(legend.position="top")

ggplot(finalProject_data, aes(Have_LittleInterestDoingThings, 
                              Depressed_Yes, colour = Level_ofDepression1_3)) + geom_point(size = 3) + 
  geom_smooth(method = "lm", aes(fill = Level_ofDepression1_3), alpha = 0.1) + theme(legend.position="top")

ggplot(finalProject_data, aes(Feelingbad_aboutYourself, 
                              Depressed_Yes, colour = Level_ofDepression1_3)) + geom_point(size = 3) + 
  geom_smooth(method = "lm", aes(fill = Level_ofDepression1_3), alpha = 0.1) + theme(legend.position="top")

#this scatterplot shows were show assumptions of linearity between the covariate and at each group of depression level
#might not be met, but it is mainly because of the outlier covid year of 2020
#we have an interaction term with this variable in the model to cover this issue
ggplot(finalProject_data, aes(Year_TwoYearInterval, 
                              Depressed_Yes, colour = Level_ofDepression1_3)) + geom_point(size = 3) + 
  geom_smooth(method = "lm", aes(fill = Level_ofDepression1_3), alpha = 0.1) + theme(legend.position="top")

#our depression model to be used for analysis
depression.model <- lm(Depressed_Yes~ Level_ofDepression1_3 + Year_TwoYearInterval+
                         TroubleSleeping_OrsleepingTooMuch + FeelingTired_OrhavingLittleEnergy+
                         Have_LittleInterestDoingThings + Feelingbad_aboutYourself+
                         Level_ofDepression1_3:Year_TwoYearInterval, data = finalProject_data)


#testing assumption of no interaction between the categorical independent variable and covariate
#also finding the significance of interaction term in the model  with all covariates
Anova(aov(depression.model, data = finalProject_data), type = 3)
#isolating the interaction term to make sure it still is not significant by itself against response variable
Anova(aov(Depressed_Yes~Level_ofDepression1_3*Year_TwoYearInterval,
          data =finalProject_data), type = 3)


#testing assumption of homogeneity of variances
#null hypothesis: "there is no difference in variance between sample groups"
res1<-depression.model$residuals
#checking distribution of residuals to determine which homogeneity of variances test to conduct (Bartlett or Levene)
hist(res1,main="Histogram of residuals",xlab="Residuals")
#Checking normality among residuals through The Shapiro-Wilk test
#Null hypothesis: data is drawn from a normal distribution
shapiro.test(resid(aov(depression.model, data = finalProject_data)))

#use bartlett test because data has been tested to be normally distributed but have slight inconsistencies
#in distribution of residuals not enough to say not normally distributed though
library(car)
bartlett.test(Depressed_Yes ~ Level_ofDepression1_3, data = finalProject_data)

#Peforming One-way ANCOVA Test on our depression model
anova_test(data = finalProject_data, formula=Depressed_Yes~ Level_ofDepression1_3 + Year_TwoYearInterval+
             TroubleSleeping_OrsleepingTooMuch + FeelingTired_OrhavingLittleEnergy+
             Have_LittleInterestDoingThings + Feelingbad_aboutYourself+
             Level_ofDepression1_3:Year_TwoYearInterval,
           type = 3, detailed = TRUE) # type 3 SS should be used in ANCOVA

#install.packages('emmeans')
library(emmeans)
#getting estimated marginal means also known as  least-squares means
#for statistically significant covariate variable - FeelingTired_OrhavingLittleEnergy
adjustMeans <- emmeans_test(data = finalProject_data, 
                            formula = Depressed_Yes ~ Level_ofDepression1_3, 
                            covariate = FeelingTired_OrhavingLittleEnergy)
get_emmeans(adjustMeans)

#perform the post-hoc test with the Benjamini-Hochberg Hockberg method 
emmeans_test(data = finalProject_data, 
             formula = Depressed_Yes~ Level_ofDepression1_3,
             covariate = FeelingTired_OrhavingLittleEnergy,
             p.adjust.method = "hochberg")