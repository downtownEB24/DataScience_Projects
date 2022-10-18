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
