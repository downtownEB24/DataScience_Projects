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
