DATA statin;  
    DO SUBJ = 1 TO 20;   
        IF RANUNI(1557) LT .5 THEN GENDER = 'FEMALE';  
                              ELSE GENDER = 'MALE'; 
        IF RANUNI(0) LT .3 THEN DIET = 'HIGH FAT'; 
                           ELSE DIET = 'LOW FAT';      
        DO DRUG = 'A','B','C';     
                LDL = ROUND(RANNOR(1557)*20 + 110 
                             + 5*(DRUG EQ 'A') 
                             - 10*(DRUG EQ 'B') 
                             - 5*(GENDER EQ 'FEMALE')  
                             + 10*(DIET EQ 'HIGH FAT'));   
                HDL = ROUND(RANNOR(1557)*10 + 20      
                             + .2*LDL         
                             + 12*(DRUG EQ 'B'));   
                TOTAL = ROUND(RANNOR(1557)*20 + LDL + HDL + 50  
                             -10*(GENDER EQ 'FEMALE')     
                             +10*(DIET EQ 'HIGH FAT'));   
                OUTPUT;  
        END;  
    END;
RUN;

PROC PRINT data = statin NOOBS;
   title 'The statin data set';
RUN;

/**************************************************************************************************************************
Filename: C:\Users\Eric\Desktop\STAT 482\hw8_2.sas
Written by: Eric Brown
Date: November 24, 2019

This program uses a program shown and displayed in the problem to compare good
cholesterol (HDL) among the three drugs that each of the patients take during the study.
The program also transpose the tall statin data set into a fat data set called 
fatstatin that contains one observation for each subject, and four variables. From this
dataset the program compares the hdl values among the three drugs using the 
ANOVA procedure's REPEATED statement. 

Input: statin.sas7bdat - displayed in the problem
Output:One factor Comparsion Analysis, fatstatin.sas7bdat, & Three possible two-way comparison 
from statin.sas7bdat produced by PROC ANOVA
**************************************************************************************************************************/
Options ls=89 ps=94 nodate nonumber;
PROC ANOVA data=statin; * comparing hdl among the three drugs;
  TITLE 'ONE-WAY REPEATED MEASURES ANOVA';
  CLASS SUBJ DRUG; *subject and drug are each main effects & no interaction term between them;
  MODEL HDL= SUBJ DRUG;
  MEANS DRUG / SNK; *running Student-Newman_Keuls test;
RUN;
TITLE;

DATA fatstatin (drop=LDL HDL TOTAL GENDER DIET DRUG); *not including variables other then ones described in prob.;
  SET statin; *reading in data from the slim tall dataset;
  BY SUBJ; *sorting by subject to run conditional statements;
    if DRUG='A' then hdl1= HDL;
	else if DRUG='B' then hdl2= HDL;
	else if DRUG='C' then hdl3= HDL;
  if last.SUBJ then output; *only outputting until each subject's data is read in;
  retain hdl1 hdl2 hdl3;
RUN;

PROC PRINT data= fatstatin NOOBS;
  TITLE 'The fat statin tranposed data set';
RUN;
TITLE;

*comparing the hdl values among the three drugs using the ANOVA procedure's REPEATED statement;
PROC ANOVA data=fatstatin;
  Title 'One-way ANOVA Using the Repeated Statement';
  Model hdl1-hdl3= / NOUNI; *not conducting separate analysis for each of the three HDL varaibles;
  *no class statement means nothing to put on the right side of the equals sign;
  Repeated DRUG3 contrast (1) / NOM SUMMARY;
  Repeated DRUG3 contrast (2) / NOM SUMMARY;
  Repeated DRUG3 contrast (3) / NOM SUMMARY;
  *three levels of the DRUG and calling the repeated factor DRUG as well;
RUN;
TITLE;
