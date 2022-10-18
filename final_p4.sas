/**************************************************************************************************************************
Filename: C:\Users\Eric\Desktop\STAT 482\final_p4.sas
Written by: Eric Brown
Date: December 14, 2019

This program gathers data regarding Donner Party emigrants survival
rate when traveling through the Sierra Nevada during the 1840s. The
input data file contains information about the age and gender of the
emigrants as well. Once the data is read into the program, it uses the
logistic procedure to predict the survival of a Party member based on their
age and gender. 

Input: donner.dat - data file
Output: Logistic Regression analysis
produced via PROC LOGISTIC
**************************************************************************************************************************/

PROC FORMAT; *formatting data to show if the person survived or not and their gender;
  VALUE survival_fmt 0='No'
                     1='Yes';
  VALUE gender_fmt 0='Male'
                   1='Female';
RUN;

DATA donner; *reading in data after the column heading from the data file;
  infile 'C:\Users\esbro\Desktop\STAT 482\donner.dat' firstobs=2;
  input survival age gender;
  format survival survival_fmt. gender gender_fmt.;
RUN;

OPTIONS ls=94 ps=90 nodate nonumber;
PROC PRINT data=donner; *making sure data is displayed properly;
RUN;

PROC LOGISTIC data=donner DESCENDING;
  title 'Predicting Odds of Survival Using Logistic Regression';
  *creating a dummy variable for gender using male sex as the reference level;
  class gender (PARAM=REF REF='Male');
  model survival= age gender; *predicting survival based on age and gender;
RUN;
title;
QUIT;
