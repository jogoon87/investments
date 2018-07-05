"""============================================================================
Copyright 2017 Joonglee Jo (jogoon87@outlook.com)
Description
-----------
	This module is a prototype for dyanmic asset allocation 
    under various asset dynamics
    [1] Dynamic Mean-Variance Asset Allocation
References
-----------
    [1] Basak and Chabakauri (2010), "Dynamic Mean-Variance Asset Allocation"
============================================================================"""

from math import *
from scipy import interpolate
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def Select_Model(Model_Name, Parameters):
    if(Model_Name == "Simulation"):
        return Simulation(Parameters)
    elif(Model_Name == "Liu(2001)"):
        return Liu2001(Parameters)
    elif(Model_Name == "GaussianReturn"):
        return GaussianReturn(Parameters)
    elif(Model_Name == "Vasicek1977"):
        return Vasicek1977(Parameters)        


def InsertDB_Portfolio(errcode, dbcursor, fund_id, asset_code, spot, ttm, optStockPath, optHedgingPath):
    if(errcode == 1):
        dbcursor.execute("INSERT INTO PORTFOLIO (FUND_ID, ASSET_CODE, SPOT, TTM, INVESTMENT, HEDGING)" \
                            "VALUES (?, ?, ?, ?, ?, ?)", \
                            (fund_id, asset_code, spot, ttm, optStockPath, optHedgingPath))
        dbcursor.commit()

class Simulation:
    def __init__(self, args):
        """ Arguemtns
            ---------
            Model : General, Liu2001, GaussianReturn, Vasicek1977
        """
        print("[Simulation] Warning: Coding in progress...\n")
        self.underasset = args[0]
        self.rf = args[1]
        self.s0 = args[2]
        self.x0 = args[3]
        self.beta = args[4]
        self.delta = args[5]
        self.gamma = args[6]
        self.rho = args[7] 
        self.nubar = args[8]
        self.xbar = args[9]
        self.speed = args[10]
        self.Name = "Liu(2001)" # temp
    def Get_Optima(self, state, ttm):
        if(self.Name == "Liu(2001)"):
            # Implementation of Proposition 2. of Basak and Chabakauri (2010), eq (26), p. 14.
            #print("[Simulation] Warning: Check out this function <Get_Optima>.\n")
            #return -1
            print("[Simulation] Monte-Carlo simultation started...\n")
            # additional arguments
            numSim = 100
            diffeps = 0.01 # epsilon for differential        
            RanNumsDif = np.random.randn(numSim, len(tsteps), 2, 2)   # random number generation
            integral = np.zeros((2, numSim))
            # calculate initial solutions only
            for n in range(numSim):
                StatePathDif = np.empty((len(tsteps), 2))
                StockPathDif = np.empty((len(tsteps), 2))
                StatePathDif[0][0] = Model.x0
                StatePathDif[0][1] = Model.x0 * (1 + diffeps)
                StockPathDif[0][0] = Model.s0
                StockPathDif[0][1] = Model.s0
                for t in range(1, len(tsteps)):
                    for i in range(2):
                        StatePathDif[t][i], StockPathDif[t][i] = Model.Get_Process(delt, \
                                                                                StatePathDif[t-1][i], \
                                                                                StockPathDif[t-1][i], \
                                                                                RanNumsDif[n][t][i][0], \
                                                                                RanNumsDif[n][t][i][1])
                        integral[i][n] += Model.Get_trapezoid(StatePathDif[t][i], StatePathDif[t-1][i], delt)
            # #-----------------------------------------------------------------
            diff = (integral.mean(1)[1] - integral.mean(1)[0]) / (StatePathDif[0][1] - StatePathDif[0][0])
            Optimal_Stock, Hedging_Demand_Ratio = Model.Get_Sim_Optima(diff, TTM) 
        else:
            print("[Simulation] Warning: <Get_Optima> is yet implemted for other models.\n")
            Optimal_Stock, Hedging_Demand_Ratio = 0, 0
        return 1, Optimal_Stock, Hedging_Demand_Ratio
        
class Liu2001:
    Name = "Liu2001"
    def __init__(self, args):
        """ Arguemtns
            ---------
            SV model parms of Liu (2001). See [1], eq. (45) on p. 25.
            rf : risk-free interest rate
            ttm : time-to-maturiy
            s0 : initial stock price
            x0 : initial state
            beta : elasticity of the market price of risk, delta*sqrt(x_t)
                    w.r.t. intantaneous stock return volatility, x_t^{0.5*beta)}
            delta : risk-premium scale parameters
            gamma : risk-aversion
            rho : correlation btn s and x
            nubar : vol of vol
            xbar : long-run variance
            speed : mean-reverting speed
            exret : excess return
        """
        self.underasset = args[0]
        self.rf = args[1]
        self.s0 = args[2]
        self.x0 = args[3]
        self.beta = args[4]
        self.delta = args[5]
        self.gamma = args[6]
        self.rho = args[7] 
        self.nubar = args[8]
        self.xbar = args[9]
        self.speed = args[10]

    def Get_Optima(self, state, ttm):
        """ This function returns closed-form optimal stock investment
            according to Liu(2001)'s stochastic volatility model.
            See p.25 of Basak and Chabakauri (2010)
        Input
        -----
            state : X, currenct state
            ttm : time-to-maturity(T-t)
        Return
        ------
            opitmal stock investment
            hedging demand ratio
        """
        Optimal_Stock = (self.delta/self.gamma) * state**((self.beta-1)/(2*self.beta)) * exp(-self.rf*ttm) \
                        * (1 - self.rho*self.nubar*self.delta * (1 - exp(-(self.speed+self.rho*self.nubar*self.delta)*ttm)) \
                        / (self.speed+self.rho*self.nubar*self.delta))
        Hedging_Numerator = self.rho*self.nubar*self.delta * (1 - exp(-(self.speed+self.rho*self.nubar*self.delta)*ttm)) \
                        / (self.speed+self.rho*self.nubar*self.delta)
        Hedging_Demand_Ratio = - Hedging_Numerator / (1.0 - Hedging_Numerator)  # hedging ratio
        return 1, Optimal_Stock, Hedging_Demand_Ratio
    
    def Get_Process(self, delt, curState, curStock, rand1, rand2):
        r = self.rf
        beta = self.beta
        delta = self.delta
        gamma = self.gamma
        nubar = self.nubar
        speed = self.speed
        Xbar = self.xbar
        rho = self.rho
        # [1] Euler scheme -------------------------------------------
        # 1) Variance(state) process
        nextState = curState + speed*(Xbar - curState)*delt + nubar*sqrt(curState*delt)*rand1
        # 2) Stock process
        nextStock = curStock * (1.0 + \
                                 (r + delta*curState**((1+beta)/(2*beta)))*delt \
                                 + curState**(1/(2*beta))*sqrt(delt)* (rho*rand1 + sqrt(1-rho*rho)*rand2) )
        # [2] Mileston -----------------------------------------------
        return nextState, nextStock

    def Get_Sim_Optima(self, diff, TTM):
        #-----------------------------------------------------------------
        x0 = self.x0
        r = self.rf
        beta = self.beta
        delta = self.delta
        gamma = self.gamma
        nubar = self.nubar
        speed = self.speed
        Xbar = self.xbar
        rho = self.rho
        OptStkDiff = x0**(-0.5/beta)*exp(-r*TTM) / gamma * (delta * sqrt(x0) - rho*nubar*diff)
        tmpnumer = rho*nubar*diff
        OptHedDiff = - tmpnumer / (delta*sqrt(x0) - tmpnumer)
        return OptStkDiff, OptHedDiff

    def Get_trapezoid(self, curState, preState, delt):
        #-----------------------------------------------------------------
        delta = self.delta
        return 0.5*delta*delta*(curState + preState)*delt

class GaussianReturn:
    def __init__(self, args):
        """ Arguemtns
            ---------
            Time-varying Gaussian mean returns. See [1], eq. (49) on p. 30.
            rf : risk-free interest rate
            s0 : initial stock price
            x0 : initial state
            sigma: stock volatility
            gamma : risk-aversion
            rho : correlation btn s and x
            nu : instantaneous variance of the state variave
            xbar : long-run variance
            speed : mean-reverting speed
        """
        self.underasset = args[0]
        self.rf = args[1]
        self.s0 = args[2]
        self.x0 = args[3]
        self.sigma= args[4]        
        self.gamma = args[5]
        self.rho = args[6] 
        self.nu = args[7]
        self.xbar = args[8]
        self.speed = args[9]
        
    def Get_Optima(self, state, ttm):
        """ This function returns closed-form optimal stock investment
            according to Liu(2001)'s stochastic volatility model.
            See p.25 of Basak and Chabakauri (2010)
        Input
        -----
            state : X, currenct state
            ttm : time-to-maturity(T-t)
        Return
        ------
            (stochastic) opitmal stock investment
            (mean) hedging demand ratio
        """
        Optimal_Stock = state/(self.gamma*self.sigma) * exp(-self.rf*ttm) \
                        - (self.rho*self.nu)/(self.gamma * self.sigma) * \
                        (self.speed * ( (1.0 - exp(-(self.speed + self.rho*self.nu)*ttm)) / (self.speed + self.rho*self.nu))**2 * self.xbar \
                         +  (1.0 - exp(-2.0*(self.speed + self.rho*self.nu)*ttm)) / (self.speed + self.rho*self.nu) * state ) * exp(-self.rf*ttm)
        Hedging_Numerator = (self.rho*self.nu) * \
                        (self.speed * ( (1.0 - exp(-(self.speed + self.rho*self.nu)*ttm)) / (self.speed + self.rho*self.nu))**2 \
                         +  (1.0 - exp(-2.0*(self.speed + self.rho*self.nu)*ttm)) / (self.speed + self.rho*self.nu) ) 
        Mean_Hedging_Demand_Ratio = - Hedging_Numerator / (1.0 - Hedging_Numerator)  # hedging ratio
        return 1, Optimal_Stock, Mean_Hedging_Demand_Ratio
    
    def Get_Process(self, delt, curState, curStock, rand1, rand2):
        r = self.rf        
        sigma = self.sigma        
        nu = self.nu
        speed = self.speed
        Xbar = self.xbar
        rho = self.rho
        # [1] Euler scheme -------------------------------------------
        # 1) Variance(state) process
        nextState = curState + speed*(Xbar - curState)*delt + nu*sqrt(curState*delt)*rand1
        # 2) Stock process
        nextStock = curStock * (1.0 + \
                                 (r + sigma*curState)*delt \
                                 + sigma*sqrt(delt)*(rho*rand1 + sqrt(1-rho*rho)*rand2) )
        return nextState, nextStock


class Vasicek1977:
    def __init__(self, args):
        """ Arguemtns
            ---------
            Vasicek (1977) stochastic interest rate model. See [1], eq. (63) on p. 37.
            ttm : time-to-maturiy
            s0 : initial stock price
            x0 : initial state(=r0)
            sigma: stock volatility
            gamma : risk-aversion
            rho : correlation btn s and r
            sigmar : instantaneous volatility of the interest rate
            rbar : long-run interest rate
            speed : mean-reverting speed
            mu : (constant) stock return
        """
        self.underasset = args[0]
        self.mu = args[1]
        self.s0 = args[2]
        self.x0 = args[3]
        self.sigma= args[4]        
        self.sigmar = args[5]
        self.gamma = args[6]
        self.rho = args[7]         
        self.rbar = args[8]
        self.speed = args[9]
        
    def Get_Optima(self, rt, ttm):
        """ This function returns closed-form optimal stock investment
            according to Liu(2001)'s stochastic volatility model.
            See p.25 of Basak and Chabakauri (2010)
        Input
        -----
            rt : r(t)
            ttm : time-to-maturity(T-t)
        Return
        ------
            (stochastic) opitmal stock investment
            (mean) hedging demand ratio
        """
        Optimal_Stock = (self.mu-rt)/(self.gamma*self.sigma**2) \
                        - (self.rho*self.sigmar)/(self.gamma * self.sigma) * \
                        ( self.speed * \
                            ( (1.0 - exp(-(self.speed - self.rho*self.sigmar/self.sigma)*ttm)) / \
                            (self.speed - self.rho*self.sigmar/self.sigma))**2 * (self.mu-self.rbar)/self.sigma \
                            +  (1.0 - exp(-2.0*(self.speed - self.rho*self.sigmar/self.sigma)*ttm)) / \
                            (self.speed - self.rho*self.sigmar/self.sigma) * (self.mu-rt)/self.sigma )
        Hedging_Numerator = (self.rho*self.sigmar) * ( self.speed * \
                            ( (1.0 - exp(-(self.speed - self.rho*self.sigmar/self.sigma)*ttm)) / \
                            (self.speed - self.rho*self.sigmar/self.sigma))**2 \
                            +  (1.0 - exp(-2.0*(self.speed - self.rho*self.sigmar/self.sigma)*ttm)) / \
                            (self.speed - self.rho*self.sigmar/self.sigma) )
        Mean_Hedging_Demand_Ratio = - Hedging_Numerator / (1.0 - Hedging_Numerator)  # hedging ratio
        return 1, Optimal_Stock, Mean_Hedging_Demand_Ratio

    def Get_Process(self, delt, curState, curStock, rand1, rand2):
        mu = self.mu        
        sigma = self.sigma      #  stock volatility
        sigmar = self.sigmar    #  interest-rate volatility
        speed = self.speed
        rbar = self.rbar
        rho = self.rho
        # [1] Euler scheme -------------------------------------------
        # 1) Interest-rate(state) process
        nextState = curState + speed*(rbar - curState)*delt + sigmar*sqrt(curState*delt)*rand1
        # 2) Stock process
        nextStock = curStock * (1.0 + \
                                 mu*delt \
                                 + sigma*sqrt(delt)*(rho*rand1 + sqrt(1-rho*rho)*rand2) )
        return nextState, nextStock

  