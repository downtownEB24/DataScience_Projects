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
