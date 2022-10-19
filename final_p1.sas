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

/**************************************************************************************************************************
Filename: C:\Users\Eric\Desktop\STAT 482\final_p2.sas
Written by: Eric Brown
Date: December 13, 2019
This program gathers data regarding northern flicker birds. Pertaining
to their tail feathers. Some of the birds had one odd feather that
was different in length and/or color from the rest of their tail feathers.
So this program compares the yellowness of one typical feather against
the one odd feather of the same bird. Then concludes whether the
mean yellowness of the odd feather differs from the typical feather. 
Input: birds.dat - data file
Output: difference of means analysis
produced via PROC TTEST
**************************************************************************************************************************/

DATA birds; *gathering in data from input data file;
  infile 'C:\Users\esbro\Desktop\STAT 482\birds.dat';
  input birdLetter $ birdType $ featherLength;
RUN;

DATA analysis;
  SET birds;
    by birdLetter; *have to sort birds by bird type for paired ttest;
	*storing the feather length of each type of bird;
      if birdType= 'Typical' then typical_Len=featherLength;
	  else if birdType= 'Odd' then odd_Len=featherLength;
    if last.birdLetter then output;
    retain typical_Len odd_Len;
    drop birdType featherLength; *no longer need orignal feather data;
RUN;
OPTIONS ls=98 ps=95 nodate nonumber;
PROC PRINT data=analysis; *making sure data is displayed correctly;
RUN;

PROC TTEST data=analysis;
  TITLE 'Paired T-test of of Northern Flicker''s Feathers';
  PAIRED typical_Len * odd_Len; *comparing typical and odd feather lengths;
RUN;
TITLE;

/**************************************************************************************************************************
Filename: C:\Users\Eric\Desktop\STAT 482\final_p3.sas
Written by: Eric Brown
Date: December 15, 2019
This program determines what the kappa coefficient is when the cutoff values
are set at 0.4, 0.5, and 0.6. Then it suggests what level of agreement exists
between the two rates. Then the program does it all over again for cutoff
values of 0.2, 0.5, 0.8.
Input: created own data set based from problem description
Output: kappa coefficient caculation
produced via PROC FREQ
**************************************************************************************************************************/

%Let cutoff1=0.2;
%Let cutoff2=0.5;
%Let cutoff3=0.8;
DATA agree;
  y=RANUNI(456);

  Do subj= 1 to 100; *100 observations
  *using seed of 456 in RANUNI function;
    If RANUNI(456) lt &cutoff1 then do;
	*two character variables to calculate the Kappa coef. between them;
      rater1='Yes';
	  rater2='Yes';
	End;
	*second cutoff value least 0.4, but less than 0.5, ;
	Else if RANUNI(456) ge &cutoff1 and RANUNI(456) lt &cutoff2 then do;
      rater1='Yes';
      rater2='No'; 
	End;
	* least 0.5, but less than 0.6;
    Else if RANUNI(456) ge &cutoff2 and RANUNI(456) lt &cutoff3 then do;
      rater1='No';
      rater2='Yes'; 
	End;
	*greater than 0.6;
	Else if RANUNI(456) ge &cutoff3  then do;
	  rater1='No';
	  rater2='No';
	End;
	Output;
  End;
RUN;

PROC PRINT data=agree;
Run;

PROC FREQ data=agree;
  TITLE 'Computing Coefficient Kappa for Two Raters';
  Tables rater1 * rater2 / AGREE; *computing kappa coef.;
RUN;


*%createDATA(cutoff1=0.2 , cutoff2=0.5 , cutoff3=0.8);

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

/**************************************************************************************************************************
Filename: C:\Users\Eric\Desktop\STAT 482\final_p5.sas
Written by: Eric Brown
Date: December 12, 2019
This program uses data stored in the permanent dataset called qul. To if
a patient's qul score changes depending on how many office visits they 
attend through the study. Then the program tests if there is an association
between years (change) and qul score.
Input: qul.sas7bdat permanent dataset
Output: frequency and assocation test
produced via PROC FREQ
**************************************************************************************************************************/

Libname stat482 'C:\Users\esbro\Desktop\STAT 482'; *using libref to reference permanent dataset later on;
DATA studyData (keep=subj v_date first_visit last_visit years first_qul last_qul change score);
  length score $ 8; *making sure the formated score value is displayed correctly;
  set stat482.qul; 
  by subj; *sorting data by subject;
  retain first_visit last_visit first_qul last_qul; *have to retain the values to calculate correctly;
  if first.subj and not missing(qul_1)then do;
    first_visit=v_date; *this is first office visit;
	first_qul=qul_1; *first qul_1 score;
  end;
  if last.subj and not missing(qul_1) then do;
    last_visit=v_date; *last office visit;
	last_qul=qul_1; *last qul_1 score;
    years=ROUND(YRDIF(first_visit, last_visit,'ACTUAL')); *getting the number of years between office visits;
    change=first_qul - last_qul; *getting the change in qul_1 score;
	*displaying score correctly via character string value;
      if (change<0) then score='Better';
      else if (change>0) then score='Worse';
      else if (change=0) then score='NoChange';
    output;
  end;
RUN;

PROC PRINT DATA=studyData;
format first_visit last_visit mmddyy10.;
RUN;

PROC FREQ DATA=studyData; *do not need cumulative values;
  Tables years*score/ nopercent nocol chisq; *need chi-square statisitic for association level;
RUN;
