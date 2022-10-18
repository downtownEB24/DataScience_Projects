/**************************************************************************************************************************
Filename: C:\Users\Eric\Desktop\STAT 482\final_p1.sas
Written by: Eric Brown
Date: December 13, 2019

This program reads in data from a data file that contains info
regarding speed with which rats can negotiate a maze. The rats
are also grouped into three age groups (3,6, and 9) and two
genetic strains (A and B). Then the program conducts a two-way analysis
of variance with age and strain as the predictor and speed as the response
variable. Then the program creates an interaction plot with the average
speed and age.

Input: ratmaze.dat - data file
Output: two-way anova analysis and interaction plot
produced via PROC GLM, PPROC MEANS, & PROC GPLOT
**************************************************************************************************************************/
DATA ratmaze; *temporary dataset containing rat maze data;
  infile 'C:\Users\esbro\Desktop\STAT 482\ratmaze.dat';
  input age strain $ speed @@;
RUN;
OPTIONS ps=94 ls=98 nodate nonumber;
PROC PRINT data=ratmaze; *making sure dataset stored and displayed correctly since design is unbalanced;
run;

PROC GLM data=ratmaze; *can not use proc anova since design is not balanced;
  TITLE 'Two-way Analysis of Variance - Unbalanced Design';
  CLASS age strain;
  MODEL speed= age | strain / ss3; *producing only type III sum of squares;
  LSMEANS age | strain / PDIFF ADJUST=TUKEY; *produce least-square, adjusted means for main effects;
   *computes probabilities for all pairwise differences and adjustment for multiple comparisons;
RUN;

*restrict the output data set while getting cell means;
PROC MEANS data=ratmaze NWAY NOPRINT;
  CLASS age strain;
  VAR speed;
  OUTPUT OUT=MEANS MEAN=average_speed;
RUN;
TITLE;

SYMBOL1 V=SQUARE COLOR=BLUE I=JOIN;
SYMBOL2 V=CIRCLE COLOR=BLACK I=JOIN;
PROC GPLOT DATA=MEANS;
  TITLE 'Interaction Plot';
  PLOT average_speed * age = strain; *age is x-axis variable & average_speed y-axis variable;
RUN;
TITLE;
