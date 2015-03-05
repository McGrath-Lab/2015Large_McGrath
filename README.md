# 2015Large_McGrath
Python scripts for solving the egglaying models

Run the script using : python fit_Egglaying.py > output.csv

Input Data : 

This version does not need any input file, the data for the two strains, CX12311 and nurf-1 are already declared
in the script. This input data is the Egglaying rate at 5 time points viz. 2,5,27,51 and 72 hours. 

Output Data : 

1. Predicted Egglaying Rate for CX12311 and nurf-1 at time points from 0 to 200 hours for initial sperm, S0 = 300 and 450

2. Estimated fit parameters : ko for each strain and kc,kf shared for the two

3. Effect Sizes, i.e difference in the predicted Egglaying Rates between CX12311 and nurf-1

4. Coefficients for the model equation : intercept + k1*X1 + k2*X2 + ki*X1*X2

Modules Needed to run the script : 

lmfit : http://lmfit.github.io/lmfit-py/installation.html
numpy and scipy : http://www.scipy.org/scipylib/download.html
