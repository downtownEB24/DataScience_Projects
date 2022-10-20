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
Filename: C:\Users\Eric\Desktop\STAT 482\hw8_4.sas
Written by: Eric Brown
Date: November 24, 2019

This programuses the statin data set from problem 8.2 and addes the variable GENDER to
the model. As well as test for GENDER and DRUG effects and GENDER by DRUG interaction.
It is understood that the variable DRUG is a repeated measure factor. And the program
uses PROC GLM to start the analysis since the design is unbalanced.

Input: statin.sas7bdat - displayed in previous problem
Output:Two-way ANOVA, Interaction Plot & Analysis
from statin.sas7bdat produced by PROC GLM, PROC MEAN, & PROC GPLOT
**************************************************************************************************************************/
Options ls=89 ps=94 nodate nonumber;
PROC GLM DATA=statin; *using glm instead of anova because design is unbalanced;
  Title 'Two-way ANOVA - Unbalanced Design';
  Class GENDER DRUG; *testing for these effects;
  Model HDL= GENDER | DRUG / SS3; *tell SAS to generate only the Type III sums of squares;
  Lsmeans GENDER | DRUG / PDIFF ADJUST=TUKEY;
RUN;
Title;

PROC MEANS DATA=statin NOPRINT NWAY; *used to get mean hdl value to plot the interaction graph;
  CLASS GENDER DRUG;
  VAR HDL;
  OUTPUT OUT=INTER
         MEAN=meanHdl;
RUN;

SYMBOL1 VALUE=CIRCLE COLOR=BLUE INTERPOL=JOIN;
SYMBOL2 VALUE=SQUARE COLOR=BLACK INTERPOL=JOIN;
PROC GPLOT DATA=INTER;
  TITLE 'Interaction Plot';
  PLOT meanHdl*DRUG=GENDER; 
*mean hdl on the y-axis, drug on the x-axis, and gender as the plotting symbol;
RUN;
TITLE;
