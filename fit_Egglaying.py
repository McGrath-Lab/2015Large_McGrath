## Loading the required libraries
from scipy.integrate import ode
from scipy.optimize import leastsq
import numpy as np  
import math
from lmfit import minimize, Parameters, Parameter, fit_report

########################################################################
##                         MODEL FUNCTION                             ##
##              Takes in current time, state and parameters           ##
##	          and outputs the change in state values              ##		  
##							              ##			
########################################################################

def egglayingModel1(t,stateVar,params):
        ## t is not used but automatically sent by the ODE solver
        ## stateVar is a list of states organized as [S1,O1,E1,S2,O2,E2....SN,ON,EN]
        ## params is a dictionary of Parameter objects,  keyed as = ko1,ko2...koN, kf, kc
        ## The model is:
    	## dE = kf*O*S
    	## dS = -dE
    	## dO = ko - kc*O - dE

        num_strains = len(stateVar) / 3
        diffs = []

        for i in range(0,num_strains) :
                dE = params['kf'].value*stateVar[1+3*i]*stateVar[3*i]
                dS = -dE
                dO = params['ko'+str(i)].value - params['kc'].value*stateVar[1+3*i] - dE
                diffs.extend([dS,dO,dE])
	return diffs

########################################################################
##                          MODEL SOLVER                              ##
##             Solves ODE and outputs list of egg-laying rates        ##
##							              ##			
########################################################################

def solveODE(params, model_function, integrator, delta_t, final_t):
        ## params is a dictionary of Parameter objects,  keyed as = ko1,ko2...koN, kf, kc, S0
        ## assumes E and O start at 0 at t=0  and S is given by S0 in params dictionary
        ## model_function is a function that outputs state changes given the current t, state, and parameters
        ## integrator is one of the possible integrators used by ode()
        ## delta_t indicates the timestep to integrate ODEs
        ## final_t indicates how long to integrate to

        ## Set initial states and initialize data structures. Assumes ko is in ko1
        initial_states = [params['S0'].value,0,0]
        y = {}
        pr_data=[]

        ## Initialize ODE solver
        r = ode(model_function).set_integrator(integrator, nsteps = 160000) 
        r.set_initial_value(initial_states,0).set_f_params(params)

        ## Start integrator. r.y[0] condition to avoid overflow errors
        k = 0
        while r.successful() and r.t < final_t and r.y[0] > .000001:
                r.integrate(r.t+delta_t)
                y[k] = r.y
                #y is [S1,O1,E1...Sn,On,En]
                k+=1

        ## If integrator stops before final t, fill out remaining values
        while r.t<final_t:
                y[k] = [0, y[k-1][1], y[k-1][2]]
                k = k+1
                r.t = r.t + delta_t
                
        ## Egg-laying rate at t=0 is 0
        pr_data.extend([0])

        ## Calculate egg-laying rate for remaining timepoints
        for i in range(1,len(y)):
                #Assumes that E is third third data point
                e_prime = (y[i][2] - y[i-1][2])/delta_t
                pr_data.extend([e_prime])
                
        return(pr_data)

########################################################################
##			   RESIDUAL FUNCTION			      ##
##        Given parameters and data, calculates the residual          ##
##        are called from here, returns residuals                     ## 
########################################################################
def residual(params, model_function, data, integrator, delta_t, final_t):
        ## params is a Parameters object, 1 ko per strain. 1 kf, kc, and ks for all strains
        ## model_function is a function pointer for egg-laying model
        ## data is a dictionary of keys and lists [5 data points] with data['times'] = to the time points of the data
        ## inital_states are [S1,O1,E1,S2,O2,E2....SN,ON,EN] where N is >=1
        ## integtrator is a type of integrator used by ode
        ## delta_t specifies the size of the times step

        ## Initialize globals

        y = {}
        res = []

        ## Calculate number of data points and strains from data
        num_strains = len(data)-1
        num_data_points = len(data['times'])

        ## Initialize states
        initial_states = setStates(num_strains,[params['S0'].value,0,0])

        ## Initialize integrator
        r = ode(model_function).set_integrator(integrator, nsteps = 160000) 
        r.set_initial_value(initial_states,0).set_f_params(params)


        ## Solve ODE
        k = 0
        while r.successful() and r.t < final_t and (r.y[0] > .000001 or r.y[3] > .000001):
                r.integrate(r.t+delta_t)
                y[k] = r.y
                #y is [S1,O1,E1...Sn,On,En]
                k+=1

        while r.t<final_t:
                y[k] = []
                for i in range(0,num_strains):
                        y[k].extend([0, y[k-1][1], y[k-1][2]])
                k = k+1
                r.t = r.t + delta_t
        

        ## Calculate residuals
        j = 0
        for key in data:
                if key != 'times' :
                        for i in range(0,num_data_points):
                                time = data['times'][i]
                                index = int(time/delta_t)
                                #Assumes that E is third third data point
                                e_prime = (y[index][j*3+2] - y[index-1][j*3+2])/delta_t
                                res.extend([e_prime - data[key][i]])
                        j+=1

	return res


########################################################################
##			  PARAMETER FUNCTION                          ##
##          This function creates a dictionary of parameters          ##
########################################################################
def setParameters(default_params) : 
        ## default parameters is a dictionary of lists
        parms = Parameters()
        for key in default_params:
                if len(default_params[key]) == 2:
                        parms.add(key, value = default_params[key][0], vary = default_params[key][1])
                if len(default_params[key]) == 4:
                        parms.add(key, value = default_params[key][0], vary = default_params[key][1], min = default_params[key][2], max = default_params[key][3])

        return parms

########################################################################
##			  STATE FUNCTION			      ##
########################################################################
def setStates(num_strains, default_values = [300,0,0]) : 
        states = []
        for i in range(0,num_strains) :
        	states.extend(default_values)
        return states


########################################################################
##		         CALCULATE INTERACTIONS	        	      ##
## This function calculates the additive and interactive terms        ##
########################################################################
        
def calc_interaction(a,b,c,d):
        ## a = data for genotype [0,0]
        ## b =  data for genotype [1,0]
        ## c =  data for genotype [0,1]
        ## d =  data for genotype [1,1]
        ## E' = intercept + k1*X1 + k2*X2 + ki*X1*X2

        model = {}
        model['intercept'] = []
        model['k1'] = []
        model['k2'] = []
        model['ki'] = []

        for i in range(0,len(a)):
                model['intercept'].extend([a[i]])
                model['k1'].extend([b[i]-a[i]])
                model['k2'].extend([c[i]-a[i]])
                model['ki'].extend([d[i]-c[i]-b[i]+a[i]])
        return model

########################################################################
##			     MAIN FUNCTION                            ##
########################################################################

## Globals
final_t = 200
delta_t = .1
ODE_method = 'dopri'
minimize_method = 'leastsq'
mQTL = .1 # effect of modifier QTL on oocyte generation rate for Figure 4a

## CX12311 and NILnurf-1 egg-laying data
data = {}
data['times'] = [2,7,29,53,81]
data['CX12311'] = [0.04, 2.91, 6.72, 1.81, 0.22]
data['nurf-1'] = [0.00, 0.09, 3.65, 5.66, 3.92]

## Create parameter dictionary: fit individual kos to CX12311 and NIL nurf-1 and shared kf and kc values. 
## Initial sperm for both strain = 300
params = setParameters({'ko0': [1.5, True, .5, 4], 'ko1': [1.5, True, .5, 4],'kf': [.00048, True, .00001, .001], 'kc': [-0.165, True, -0.17, 0.1], 'S0':[300, False]})


## Calculate best fit parameters
out_data = minimize(residual, params, method = minimize_method, args=(egglayingModel1, data, ODE_method, delta_t, final_t))
print params


## Print out sum square of residuals to ascertain goodness of fit
print (np.sum(out_data.residual*out_data.residual))

## Calculate ODEs for best fits
params['ko0'].value = out_data.params['ko0'].value
params['kf'].value = out_data.params['kf'].value
params['kc'].value = out_data.params['kc'].value
CX12311 = solveODE(params, egglayingModel1, ODE_method, delta_t, final_t)
params['ko0'].value = out_data.params['ko1'].value
NILnurf1 = solveODE(params, egglayingModel1, ODE_method, delta_t, final_t)

## Model what happens when we increase initial sperm to 450
params['S0'].value = 450
params['ko0'].value = out_data.params['ko0'].value
CX12311_450 = solveODE(params, egglayingModel1, ODE_method, delta_t, final_t)
params['ko0'].value = out_data.params['ko1'].value
NILnurf1_450 = solveODE(params, egglayingModel1, ODE_method, delta_t, final_t)

## For figure 4a, model effect of mQTL on egg-laying using parameters from above
params['S0'].value = 300
params['ko0'].value = out_data.params['ko1'].value
a = solveODE(params, egglayingModel1, ODE_method, delta_t, final_t)
params['ko0'].value = out_data.params['ko1'].value + mQTL
b = solveODE(params, egglayingModel1, ODE_method, delta_t, final_t)
params['ko0'].value = out_data.params['ko0'].value
c = solveODE(params, egglayingModel1, ODE_method, delta_t, final_t)
params['ko0'].value = out_data.params['ko0'].value + mQTL
d = solveODE(params, egglayingModel1, ODE_method, delta_t, final_t)

## Calculate interaction term between nurf-1 deletion and the mQTL
model = calc_interaction(a,b,c,d)

## Print out data to stdout to import into excel for graphing.
print 'Time(h),CX12311,NILnurf1,CX12311_450,NILnurf1_450,'+str(out_data.params['ko1'].value)+','+str(out_data.params['ko1'].value+mQTL)+','+str(out_data.params['ko0'].value)+','+str(out_data.params['ko0'].value+mQTL)+',Intercept,k1,k2,ki'
for i in range(0,len(a)):
        print str(i*delta_t) + ',' + str(CX12311[i]) + ',' + str(NILnurf1[i]) + ',' + str(CX12311_450[i]) + ',' + str(NILnurf1_450[i]) + ',' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ',' + str(d[i]) + ',' + str(model['intercept'][i]) + ',' + str(model['k1'][i]) + ',' + str(model['k2'][i]) + ',' + str(model['ki'][i])
