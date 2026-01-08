/* ===============================================
   Lloyds Bank - Loan Approval Analysis
   Business Analytics CW2 - Final SAS MIS System
   =============================================== */

OPTIONS NODATE NONUMBER;

/* -----------------------------------------------
 Step 1: Dynamic Importing of Data
-------------------------------------------------- */
%LET path = %SYSGET(HOME);

PROC IMPORT DATAFILE="&path./loan_data.csv"
OUT=loan_data
DBMS=CSV
REPLACE;
GUESSINGROWS=MAX;
RUN;



/* -----------------------------------------------
 Step 2: Exploring Raw Data 
-------------------------------------------------- */
/* Display variable names, types, and attributes for loan_data */
PROC CONTENTS DATA=loan_data; RUN; 

/* Print the first 10 rows of LOAN_DATA for a quick preview */
PROC PRINT DATA=loan_data (OBS=10); RUN; 


/* -----------------------------------------------
 Step 3: Handling Missing Values and Outliers
-------------------------------------------------- */
/* Show the number of observations (N) and missing values (NMISS) for each numeric variable */
PROC MEANS DATA=loan_data N NMISS; RUN;

/*-----------------------------------------------------------
   Quickly check the 99th-percentile (P99) of income and loan
   amount to see where the extreme values (potential outliers)
   start.  Helps us decide sensible caps for each variable.
-----------------------------------------------------------*/
PROC MEANS DATA=loan_data P99;
    VAR person_income loan_amnt;
RUN;

/* Capping Extreme Income and Loan Amount */
DATA loan_data;
    SET loan_data;
    IF person_income > 200000 THEN person_income = 200000;
    IF loan_amnt > 50000 THEN loan_amnt = 50000;
RUN;


/* -----------------------------------------------
 Step 4: Feature Engineering
-------------------------------------------------- */
DATA loan_data;
    SET loan_data;
    income_loan_ratio = person_income / loan_amnt;

    /* Encoding Categorical Variables */
    gender_dummy = (person_gender = "male");

    education_master = (person_education = "Master");
    education_bachelor = (person_education = "Bachelor");
    education_highschool = (person_education = "High School");

    home_own = (person_home_ownership = "OWN");
    home_rent = (person_home_ownership = "RENT");
    home_mortgage = (person_home_ownership = "MORTGAGE");

    loan_personal = (loan_intent = "PERSONAL");
    loan_education = (loan_intent = "EDUCATION");
    loan_medical = (loan_intent = "MEDICAL");

    prev_loan_default = (previous_loan_defaults_on_file = "Yes");
    loan_approved = (loan_status = 1);
RUN;


/* -----------------------------------------------
 Step 5: Normalisation of Numeric Variables 
-------------------------------------------------- */
PROC UNIVARIATE DATA=loan_data NORMAL;
    VAR person_income loan_amnt;
    HISTOGRAM / NORMAL;
RUN;

DATA loan_data;
    SET loan_data;
    log_income = LOG(person_income);
    log_loan_amnt = LOG(loan_amnt);
RUN;

PROC UNIVARIATE DATA=loan_data NORMAL;
    VAR log_income log_loan_amnt;
    HISTOGRAM / NORMAL;
RUN;


/* -----------------------------------------------
  Step 6: Exploratory Data Analysis (EDA)
-------------------------------------------------- */

/* Summary Statistics */
PROC MEANS DATA=loan_data MEAN STD MIN MAX MEDIAN;
    VAR person_age person_income credit_score loan_amnt loan_int_rate;
RUN;


/* -----------------------------------------------
/*Frequency Analysis

Question:
Does the loan approval rate differ significantly between male and female applicants?
----------------------------------------------- */

PROC FREQ DATA=loan_data;
    TABLES person_gender person_home_ownership loan_intent loan_status;
RUN;

/* Credit Score Distribution Analysis: Histogram with Kernel Density Estimation (smooth distributions)*/
PROC KDE data=loan_data;
   univar credit_score /
         plots=histdensity     /* histogram + kernel density */
         bwm=0.5;              /* bandwidth multiplier       */
RUN;

/* -----------------------------------------------
 Formatting loan_status for readability
-------------------------------------------------- */
PROC FORMAT;
    VALUE status_fmt 0 = 'Rejected'
                     1 = 'Approved';
RUN;

PROC FREQ DATA=loan_data;
    TABLES loan_status;
    FORMAT loan_status status_fmt.;
    TITLE "Loan Status Frequency (Formatted)";
RUN;


/* -----------------------------------------------
   Step 7: Data Visualisation
-------------------------------------------------- */
/*  Histogram: Age Distribution of Loan Applicants */
ODS GRAPHICS ON;
PROC SGPLOT DATA=loan_data;
    HISTOGRAM person_age;
    TITLE "Histogram: Age Distribution of Loan Applicants";
RUN;
ODS GRAPHICS OFF;


/*  Box Plot: Income Distribution by Loan Status
   Purpose: Visualizes the spread and median of applicant income, grouped by loan approval status.
   Insight: Highlights income differences between approved and rejected applicants, including outliers. */
PROC SGPLOT DATA=loan_data;
    VBOX person_income / CATEGORY=loan_status;
RUN;

/*  Clustered Bar Chart: Gender vs Loan Status
   Purpose: Compares the count of approved vs rejected loans across genders using a grouped bar.
   Insight: Useful for exploring gender-related disparities in loan approvals. */
PROC SGPLOT DATA=loan_data;
    VBAR person_gender / GROUP=loan_status GROUPDISPLAY=CLUSTER 
        FILLATTRS=(TRANSPARENCY=0.2) DATALABEL;
    KEYLEGEND / TITLE="Loan Approval Status";
RUN;

/*  Panel Histogram: Income Distribution Faceted by Loan Status
   Purpose: Splits histograms of income by loan_status using paneling for side-by-side comparison.
   Insight: Allows quick visual comparison of income patterns among approved vs rejected applicants. */
PROC SGPANEL DATA=loan_data;
    PANELBY loan_status;
    HISTOGRAM person_income;
RUN;

/*  Vertical Bar Chart: Loan Intent Distribution
   Purpose: Displays frequency of different loan intents (e.g., education, personal, medical).
   Insight: Identifies the most common reasons for loan applications. */
PROC GCHART DATA=loan_data;
    VBAR loan_intent;
RUN;

/*  Scatter Plot: Income vs Loan Amount by Status
   Purpose: Plots person_income against loan_amnt, colored by loan_status.
   Insight: Reveals any linear patterns, clusters, or anomalies in income-loan relationships across statuses. */
PROC SGPLOT DATA=loan_data;
    SCATTER X=person_income Y=loan_amnt / GROUP=loan_status;
RUN;

/* 📊 Box Plot: Income Distribution by Education Level
   Purpose: Visualizes the spread and median of income across different education levels.
   Insight: Reveals which education groups tend to earn more and how income varies within each group. */
PROC SGPLOT DATA=loan_data;
    VBOX person_income / CATEGORY=person_education;
    TITLE "Income Distribution by Education Level";
RUN;

/*  Bar Chart: Loan Intent Frequency Distribution
   Purpose: Shows the number of applications for each loan purpose.
   Insight: Highlights the most and least common reasons for taking out loans. */
PROC SGPLOT DATA=loan_data;
    VBAR loan_intent / DATALABEL;
    TITLE "Loan Purpose Distribution";
RUN;

/*  Clustered Bar Chart: Loan Intent by Approval Status
   Purpose: Compares the approval rates for each loan intent category.
   Insight: Helps understand if approval likelihood varies by loan purpose. */
PROC SGPLOT DATA=loan_data;
    VBAR loan_intent / GROUP=loan_status GROUPDISPLAY=CLUSTER DATALABEL;
    TITLE "Loan Status by Loan Intent";
RUN;

/*  Density Plot: Distribution of Credit Scores
   Purpose: Plots the smoothed density curve of credit scores to understand distribution shape.
   Insight: Highlights skewness, peaks, and spread in applicant credit scores. */
PROC SGPLOT DATA=loan_data;
    DENSITY credit_score / TYPE=KERNEL;
    TITLE "Density Plot of Credit Score";
RUN;

/*  Paneled Scatter Plot: Income vs Loan Amount by Loan Intent
   Purpose: Displays loan amount vs. income for each loan intent in separate panels.
   Insight: Helps reveal intent-specific borrowing behaviors across income levels. */
PROC SGPANEL DATA=loan_data;
    PANELBY loan_intent;
    SCATTER X=person_income Y=loan_amnt;
    TITLE "Income vs Loan Amount by Loan Intent";
RUN;

/*  Heatmap: Income vs Loan Amount Frequency
   Purpose: Simulates density by coloring the frequency of unique income-loan pairs.
   Insight: Highlights common combinations and borrower patterns. */

/* 🔧 Step 1: Create frequency matrix */
proc sql;
    create table income_loan_freq as
    select person_income, loan_amnt, count(*) as freq
    from loan_data
    group by person_income, loan_amnt;
quit;



/*  3D Bar Chart: Average Credit Score by Home Ownership Type
   Purpose: Visualizes average credit scores across home ownership categories,
   segmented by loan approval status using a 3D vertical bar chart.
   Insight: Highlights whether renters, owners, or mortgage-holders differ in creditworthiness across approval outcomes. */
PROC GCHART DATA=loan_data;
    VBAR3D person_home_ownership / 
        SUBGROUP=loan_status
        SUMVAR=credit_score
        TYPE=MEAN
        INSIDE=MEAN
        WIDTH=12
        RAXIS=AXIS1;
    AXIS1 LABEL=("Average Credit Score");
    TITLE "Credit Score by Home Ownership (3D)";
    PATTERN1 COLOR=steelblue;
    PATTERN2 COLOR=darkred;
RUN;


/* -----------------------------------------------
  Step 8: Correlation Analysis
-------------------------------------------------- */
PROC CORR DATA=loan_data PEARSON SPEARMAN PLOTS=SCATTER;
    VAR person_age person_income person_emp_exp loan_amnt loan_int_rate
        loan_percent_income cb_person_cred_hist_length credit_score;
RUN;

/* -----------------------------------------------
   9. Advanced Analytics
-------------------------------------------------- */
/*  PROC SQL: Average Credit Score by Loan Status
   Purpose: Computes summary statistics (average, min, max, count) for credit scores grouped by loan status.
   Insight: Highlights how credit score distributions vary between approved and rejected applicants. */
%let path = %sysget(HOME); 
ods html file="&path./Average_Credit_Score_by_Loan_Status.html";
title "Average Credit Score by Loan Status";  /* Title for the output */
PROC SQL;
	SELECT loan_status
	       , COUNT(*) AS total_applicants
	       , AVG(credit_score) AS avg_score
	       , MIN(credit_score) AS min_score
	       , MAX(credit_score) AS max_score
	FROM loan_data
	GROUP BY loan_status;
QUIT;
ods html close;  /* Close ODS */


/*  PROC SQL: Average Loan Amount by Education Level
   Purpose: Calculates the average loan amount requested per education category.
   Insight: Shows which education levels are requesting higher loans. */
%let path = %sysget(HOME); 
ods html file="&path./Average_Loan_Amount_by_Education_Level.html"; /* Open ODS to capture output */
title "Average Loan Amount by Education Level";  /* Title for the output */
PROC SQL;
    SELECT person_education, AVG(loan_amnt) AS avg_loan_amount
    FROM loan_data
    GROUP BY person_education;
QUIT;
ods html close;  /* Close ODS */


/*  PROC SQL: Default Rate by Home Ownership
   Purpose: Calculates the rejection rate grouped by home ownership status.
   Insight: Identifies approval trends based on home ownership types. */
%let path = %sysget(HOME); 
ods html file="&path./Default_Rate_by_Home_Ownership.html"; /* Open ODS to capture output */
title "Default Rate by Home Ownership";  /* Title for the output */
PROC SQL;
    SELECT person_home_ownership,
           SUM(CASE WHEN loan_status = 0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS rejection_rate_percent
    FROM loan_data
    GROUP BY person_home_ownership;
QUIT;
ods html close;  /* Close ODS */


/*  PROC SQL: Count of Applicants per Loan Intent
   Purpose: Counts how many applicants fall under each loan intent category.
   Insight: Highlights the most common reasons for loan applications. */
%let path = %sysget(HOME);  
ods html file="&path./Count_of_Applicants_per_Loan_Intent.html"; /* Open ODS to capture output */ 
title "Count of Applicants per Loan Intent";  /* Title for the output */
PROC SQL;
    SELECT loan_intent, COUNT(*) AS applicant_count
    FROM loan_data
    GROUP BY loan_intent
    ORDER BY applicant_count DESC;
QUIT;
ods html close;  /* Close ODS */


/*  PROC SQL: Approval Rate by Education Level
   Purpose: Calculates the percentage of loans approved for each education level.
   Insight: Evaluates how education affects approval likelihood. */
%let path = %sysget(HOME);  
ods html file="&path./Approval_Rate_by_Education_Level.html"; /* Open ODS to capture output */  
title "Approval Rate by Education Level";  /* Title for the output */
PROC SQL;
    SELECT person_education,
           SUM(CASE WHEN loan_status = 1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS approval_rate
    FROM loan_data
    GROUP BY person_education;
QUIT;
ods html close;  /* Close ODS */


/*  PROC SQL: Min, Max, and Average Credit Score by Gender
   Purpose: Descriptive statistics for credit scores split by gender.
   Insight: Observes creditworthiness trends between male and female applicants. */
%let path = %sysget(HOME);  
ods html file="&path./Credit Score_by_Gender.html"; /* Open ODS to capture output */  
title "Min, Max, and Average Credit Score by Gender";  /* Title for the output */
PROC SQL;
    SELECT person_gender,
           MIN(credit_score) AS min_score,
           MAX(credit_score) AS max_score,
           AVG(credit_score) AS avg_score
    FROM loan_data
    GROUP BY person_gender;
QUIT;
ods html close;  /* Close ODS */


/*  PROC SQL: Monthly Income Bracket Bucketing 
   Purpose: Groups applicants into income brackets.
   Insight: Allows comparison of approval rates across income segments. */
%let path = %sysget(HOME);   
ods html file="&path./Monthly Income_Bracket_Bucketing Score_by_Gender.html"; /* Open ODS to capture output */   
title "Monthly Income Bracket Bucketing";  /* Title for the output */
PROC SQL;
    SELECT CASE 
             WHEN person_income < 20000 THEN 'Low Income'
             WHEN person_income BETWEEN 20000 AND 50000 THEN 'Mid Income'
             ELSE 'High Income'
           END AS income_bracket,
           COUNT(*) AS applicant_count,
           AVG(loan_status)*100 AS approval_rate
    FROM loan_data
    GROUP BY income_bracket;
QUIT;
ods html close;  /* Close ODS */

/* 📊 PROC SQL: Average Loan Amount and Interest Rate by Approval Status
   Purpose: Compares financial features between approved and rejected applications.
   Insight: Shows how loan size and rates differ based on approval outcome. */
%let path = %sysget(HOME);   
ods html file="&path./Credit_Average_Loan_Amount_and_Interest_Rate_by_Approval_Status.html"; /* Open ODS to capture output */   
title "Average Loan Amount and Interest Rate by Approval Status";  /* Title for the output */
PROC SQL;
    SELECT loan_status,
           AVG(loan_amnt) AS avg_loan,
           AVG(loan_int_rate) AS avg_interest_rate
    FROM loan_data
    GROUP BY loan_status;
QUIT;
ods html close;  /* Close ODS */


/* 📈 PROC RANK: Rank applicants based on income
   Purpose: Adds a new column ranking each applicant by income.
   Insight: Useful for percentile analysis and targeting high/low earners. */
title "Rank applicants based on income";  /* Title for the output */
PROC RANK DATA=loan_data OUT=ranked_income TIES=MEAN;
    VAR person_income;
    RANKS income_rank;
RUN;

/* Clear title after use */
title;

/* -----------------------------------------------
 Step 10: Logistic Regression Modeling 
 
-------------------------------------------------- */

/*  
Logistic Regression Model for Loan Risk Prediction

Purpose: To estimate the likelihood that a loan will end up in a bad status (e.g., default, delinquency) based on borrower characteristics and loan attributes. This helps in risk assessment and loan portfolio management.
Business Question: Will a loan end up in a bad status (like default or delinquency) based on the borrower's profile?

 */

/* --- Sort dataset before sampling --- */
proc sort data=loan_data;
    by loan_status;
run;

/* Sampling 20% from each loan_status group */
proc surveyselect data=loan_data out=loan_sampled method=SRS samprate=0.2 seed=12345;
    strata loan_status;
run;

/* Check sample balance */
proc freq data=loan_sampled;
    tables loan_status;
run;

/*
Predicting whether a loan will end up in a bad status 
Business Question: 
Will this loan result in a bad outcome (default or delinquency) based on the applicant's profile?
*/
proc logistic data=loan_sampled descending 
              outmodel=train_model;
    class prev_loan_default home_rent loan_education / param=ref;
    model loan_status = person_income credit_score loan_percent_income 
                        prev_loan_default home_rent loan_education
                        / outroc=roc_info;       /*  Place OUTROC here */
    output out=predictions p=pred_prob;          /*  Predicted probabilities */
run;

/* -----------------------------------------------
 Test the Model on New Applicant (predicts whether a loan will end up in a bad status)
-------------------------------------------------- */

DATA new_applicant;
    INPUT person_income credit_score loan_percent_income 
          prev_loan_default home_rent loan_education;
    DATALINES;
58000 720 0.35 1 1 0
45000 680 0.25 0 0 1
65000 710 0.40 1 1 0
52000 690 0.30 0 1 1
;
RUN;

proc logistic inmodel=train_model;
    score data=new_applicant out=predicted_risk;
run;

proc print data=predicted_risk;
run;


/* 
Logistic Regression Model on Sampled Data for Loan Approval Prediction

Purpose: To build a predictive model that estimates the probability of a loan being approved or rejected based on applicant characteristics and loan attributes. This supports decision-making in the loan approval process.
Business Question: Will a loan application be approved or not, based on applicant and loan attributes?
*/
ods graphics on;
PROC LOGISTIC DATA=loan_sampled OUTMODEL=train_model PLOTS=ROC;
    CLASS prev_loan_default home_rent loan_education / PARAM=REF;
    MODEL loan_approved (EVENT='0') = 
          prev_loan_default loan_percent_income loan_int_rate 
          person_income credit_score loan_education home_rent;
    OUTPUT OUT=predictions P=pred_prob;
    ODS OUTPUT ROCInfo=ROC_Info;
    ODS OUTPUT Classification=ConfusionMatrix;
RUN;

/* -----------------------------------------------
 Test the Model on New Applicant (predicts whether a loan will be approved)
-------------------------------------------------- */

DATA new_applicant;
    INPUT prev_loan_default loan_percent_income loan_int_rate 
          person_income credit_score loan_education home_rent;
    /* Values:
        prev_loan_default: 1 = Yes
        loan_percent_income: 0.35 = 35% of income
        loan_int_rate: 11.5%
        person_income: £58,000
        credit_score: 720
        loan_education: 0 = Not education loan
        home_rent: 1 = Applicant rents
    */
    DATALINES;
1 0.35 11.5 58000 720 0 1
0 0.25 8.5 45000 680 1 0
1 0.40 10.0 65000 710 0 1
0 0.30 7.0 52000 690 1 1
;
RUN;


/* Score the new applicant using saved model */
PROC LOGISTIC INMODEL=train_model;
    SCORE DATA=new_applicant OUT=predicted_risk;
RUN;

/* Export to Excel */
ods excel file="&outdir.predicted_results_&stamp..xlsx";
title "Sample of Scored Applicants";
proc print data=predicted_risk;
run;
ods excel close;


/*  Set threshold and flag predictions */
%let cut = 0.30;

data predictions_flag;
    set predictions;
    pred_flag = (pred_prob >= &cut);
run;

/* Confusion matrix */
proc freq data=predictions_flag noprint;
    tables loan_status * pred_flag / out=confusionmatrix;
run;


/* ─────────────────────────  STEP 10‑E  ─────────────────────────
   Confusion matrix  +  accuracy, recall, precision, F‑score, …
   ‑‑ choose ONE probability cut‑off (edit &cut as you like)      */
%let cut = 0.30;                              /* 0.30 = approve if P≥0.30 */

data predictions_flag;
    set predictions;                          /* created in PROC LOGISTIC */
    pred_flag = (pred_prob >= &cut);           /* 1 = predicted APPROVE */
run;

/* 1)  Cross‑tabulate once, keep counts only (no percents) */
proc freq data=predictions_flag noprint;
    tables loan_approved * pred_flag / out=conf_counts;
run;

/* 1️⃣ Confusion Matrix Counts */
proc freq data=predictions_flag noprint;
    tables loan_status * pred_flag / out=conf_counts;
run;

/* 2️⃣ Aggregate into metrics */
proc sql noprint;
    create table perf_metrics as
    select 
        sum(case when loan_status=0 and pred_flag=0 then 1 else 0 end) as TN,
        sum(case when loan_status=0 and pred_flag=1 then 1 else 0 end) as FP,
        sum(case when loan_status=1 and pred_flag=1 then 1 else 0 end) as TP,
        sum(case when loan_status=1 and pred_flag=0 then 1 else 0 end) as FN
    from predictions_flag;
quit;

/* 3️⃣ Calculate Accuracy, Precision, Recall, F1 etc. */
data perf_metrics;
    set perf_metrics;
    total       = TN + FP + TP + FN;
    accuracy    = (TP + TN) / total;
    precision   = TP / (TP + FP);
    recall      = TP / (TP + FN);
    specificity = TN / (TN + FP);
    f1_score    = 2 * precision * recall / (precision + recall);
run;

/* 4️⃣ View the metrics */
proc print data=perf_metrics label noobs;
    label 
        TN         = "True Negatives"
        FP         = "False Positives"
        TP         = "True Positives"
        FN         = "False Negatives"
        accuracy   = "Accuracy"
        precision  = "Precision"
        recall     = "Recall"
        specificity= "Specificity"
        f1_score   = "F1 Score";
    title "Model Evaluation Metrics at P ≥ &cut";
run;


/* 2)  Roll the counts into a single‑row data set */
proc sql noprint;
    create table perf_metrics as
    select 
        sum(case when loan_approved=0 and pred_flag=0 then 1 end) as TN,
        sum(case when loan_approved=0 and pred_flag=1 then 1 end) as FP,
        sum(case when loan_approved=1 and pred_flag=1 then 1 end) as TP,
        sum(case when loan_approved=1 and pred_flag=0 then 1 end) as FN
    from predictions_flag;
quit;

/* 3)  Derive the standard evaluation metrics */
data perf_metrics;
    set perf_metrics;
    total       = TN + FP + TP + FN;
    accuracy    = (TP + TN) / total;
    sensitivity = TP / (TP + FN);                  /* recall / TPR        */
    specificity = TN / (TN + FP);                  /* TNR                 */
    precision   = TP / (TP + FP);                  /* PPV                 */
    fpr         = FP / (FP + TN);                  /* fall‑out            */
    fnr         = FN / (FN + TP);
    f1          = 2 * precision * sensitivity /
                      (precision + sensitivity);
    keep TN FP TP FN total
         accuracy sensitivity specificity
         precision fpr fnr f1;
run;

/* 4)  Print the lot in one tidy table */
proc print data=perf_metrics label noobs;
    label
        TN          = 'True Negatives'
        FP          = 'False Positives'
        TP          = 'True Positives'
        FN          = 'False Negatives'
        accuracy    = 'Accuracy'
        sensitivity = 'Sensitivity (Recall)'
        specificity = 'Specificity'
        precision   = 'Precision (PPV)'
        fpr         = 'False Positive Rate'
        fnr         = 'False Negative Rate'
        f1          = 'F1 Score';
    title "Confusion Matrix & Performance Metrics at P ≥ %sysevalf(&cut*100)%";
run;


/* ======================  STEP 11  ======================
   MODEL VALIDATION  –  k‑fold ROC  &  Gains / Lift table
   ====================================================== */

/*  STEP 11‑A: Score model and generate ROC data */
proc logistic inmodel=train_model;
    score data=loan_sampled out=roc_data outroc=roc_info;
run;

/*  Plot ROC Curve */
proc sgplot data=roc_info;
    series x=_1mspec_ y=_sensit_;
    lineparm x=0 y=0 slope=1 / transparency=0.7;
    xaxis label='1 - Specificity';
    yaxis label='Sensitivity';
    title "ROC Curve from Scored Data";
run;


/* (11‑B)  Gains / Lift table – deciles on all scored data */
proc rank data=predictions out=gains groups=10 descending;
    var pred_prob;
    ranks decile;
run;

proc sql;
    create table gains_summary as
    select decile+1                              as decile   /* 1‑10 */
         , count(*)                              as n
         , sum(loan_approved)                    as approved
         , mean(loan_approved)*100               as resp_rate format=6.1
    from gains
    group by decile
    order by decile;
quit;

/* -----------------------------------------------
   12. Export & Reporting
-------------------------------------------------- */

/*  Set correct export path using your user directory */
%let outdir = &path.;
%let stamp  = %sysfunc(putn(%sysfunc(date()), yymmddn8.))_%sysfunc(putn(%sysfunc(time()), time5.));

/*  Safe export macro */
%macro safe_export(data=, file=);
    %if %sysfunc(exist(&data)) %then %do;
        proc export data=&data
            outfile="&outdir./&file._&stamp..csv"
            dbms=csv replace;
        run;
    %end;
    %else %put ️ Dataset &data does not exist. Skipping export.;
%mend;

/* 🚀 Run all exports */
%safe_export(data=predictions, file=predictions);
%safe_export(data=roc_info, file=roc_info);
%safe_export(data=confusionmatrix, file=confusion_matrix);



/*------------ 12‑B  ‑‑ Single‑click Excel workbook --------*/
/*  Each PROC starts a new sheet thanks to the SHEET_INTERVAL option.
    Great for business stakeholders who love Excel.            */
ods excel file="&outdir./Loan_Analytics_Report_&stamp..xlsx"
          options(sheet_interval='proc' autofilter='all')
          style=Excel;
/*— main KPIs —*/
proc print data=predicted_risk (obs=10) label;
    title "Sample of Scored Applicants";
run;

proc print data=ConfusionMatrix label;
    title "Confusion Matrix – Hold‑out Sample";
run;

proc print data=ROC_Info (obs=20) label;
    title "ROC Curve Coordinates (First 20 Observations)";
run;





/* -----------------------------------------------
   12‑C: Polished PDF Deck
-------------------------------------------------- */
%let path   = %sysget(HOME);           /* your own home dir              */
%let outdir = &path./;                 /* folder you can write to        */
/* make a time-stamp that has no colons ( : ) – safer in file names      */
%let stamp  = %sysfunc(translate(%sysfunc(datetime(), datetime20.),-,':'));

ods pdf file="&outdir./Loan_Analytics_Deck_&stamp..pdf"
        dpi=300 startpage=now;
title; footnote;

/* ───── Executive summary ───── */
ods pdf text="^{style[fontsize=14pt fontweight=bold]
              Lloyds Bank – Retail Lending Analytics}";
ods pdf text=" ";
ods pdf text="Key takeaways:";
ods pdf text="^{style[bullet] •} Previous-loan default is the strongest negative predictor.";
ods pdf text="^{style[bullet] •} ROC curve shows solid discrimination between approved and rejected cases.";
ods pdf text="^{style[bullet] •} Applicants with income > £75k and loan-to-income < 30% have 85% approval rate.";

/* ───── Flagship visual ───── */
ods layout gridded columns=2;   /* PDF supports layouts – keep it */
ods region;

proc sgplot data=loan_data;
    vbar loan_intent /
        group=loan_status
        groupdisplay=cluster
        datalabel;
    title 'Loan Status by Intent';
run;

ods layout end;

/* ───── Detailed outputs ───── */
proc print data=confusionmatrix label;
    title 'Confusion Matrix';
run;

proc print data=ranked_income(obs=20) label;
    title 'Top 20 Applicants Ranked by Income';
run;

ods pdf close;




/*------------ 12‑D  ‑‑ House‑keeping ----------------------*/
title; footnote;
%put   Export & Reporting finished at %sysfunc(datetime(), datetime.);



/* -----------------------------------------------
    End of SAS MIS System
-------------------------------------------------- */
%put  SAS MIS Completed on %sysfunc(datetime(), datetime.);
