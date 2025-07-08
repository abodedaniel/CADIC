# -*- coding: utf-8 -*-
"""
Deploy 6G subnetworks in a factory environment inspired by 3GPP and 5G ACIA.
Mobility Model: Predefined factory map, where subnetworks move in the alleys.
Channel Model: Path loss, correlated shadowing, small-scale fading with Jake's Doppler model
"""
import numpy as np
import os.path
import sys
#import gymnasium as gym
from scipy.io import loadmat, savemat
from scipy.spatial.distance import cdist
from scipy.special import j0 as bessel
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt
import control_env
import torch
np.set_printoptions(linewidth=75*3)

class env_subnetwork():
    # =================== Initialisation of class =================== #
    def __init__(self, numCell=8, numDev=1, factoryarea = [30,30], dt=0.001, mobile=True, clutter='sparse', problem='channel', 
                 numSubbands=3, numerology=5, RB_per_Subband = 2, level=4, rate=10, pOut=0.01, steps=1001,
                 reward_type='rate', observation_type='I_minmax', obs_size=3,
                 fname='code/src/env/Factory_values.mat', seed=1):

        self.problem = problem
        self.fname = fname
        if seed is not None:
            np.random.seed(seed) # Everytime environment is called to reproduce from seed

        # Subnetworks and simulation
        self.numCell = numCell          # Number of subnetworks
        self.numDev = numDev                 # Number of devices in each subnetwork (NOT IMPLEMENTED!)
        self.cellDia = 4                # Radius of subnetworks [m]
        self.subnet_radius = self.cellDia/2
        self.minDistance = 0 #self.subnet_radius
        self.sampTime = dt              # Samplerate [s]
        self.transTime = dt             # Transition time [s]
        self.mobile = mobile
        self.numSteps = steps           # Total number of environment steps
        self.simTime = int(self.numSteps * self.transTime) # Total simulation time [s]
        self.reward_type = reward_type  # Reward type [string]
        self.observation_type = observation_type # Observation type [string]
        self.updateT = self.numSteps * self.sampTime # Update time [s]
        self.rate_list = [] #np.zeros((self.numSteps, self.numCell))
        
        # Requirements
        self.r_min = 10
        self.lambda_1 = 1
        self.lambda_2 = 1e-3

        # Mobility model and environment
        self.minDist = 1                # Minimum seperation distance [m]
        self.speed = 3                  # Robot movement speed [m/s]
        self.factoryArea = factoryarea    # Size of factory [m x m]
        self.deploy_length = self.factoryArea[0]

        # Channel model
        self.corrDist = 10              # Correlated shadowing distance [m]
        self.mapRes = 0.1               # Correlated shadowing map grid reolution [m]
        self.clutter = clutter
        self.init_clutter(self.clutter)      # Parameters based on factory environment
        self.mapX = np.arange(0,self.factoryArea[0]+self.mapRes,step=self.mapRes) # Map x-coordinates
        self.mapY =  np.arange(0,self.factoryArea[1]+self.mapRes,step=self.mapRes) # Map y-coordinates

        self.codeBlockLength = 1        # Length of coded blocks
        self.codeError = 1e-3           # Codeword decoding error probability
        self.qLogE = norm.ppf(norm.cdf(self.codeError)) * np.log10(np.e) # Constant for channel rate penalty
        self.numerology = numerology
        self.numerology_to_SCS = {0: 15e3, 1: 30e3, 2: 60e3, 3: 120e3, 4: 240e3, 5: 480e3, 6: 960e3} #Table 4.2.1 3GPP TS 38.211 V18.2.0
        self.num_of_SCS_per_RB = 12    ##Section 4.4.4.1 3GPP TS 38.211 V18.2.0
        # self.cqi_to_se = {0: 0.0, 1:0.1523, 2: 0.2344, 3: 0.3770, 4: 0.6016, 5: 0.8770, 6: 1.1758, 7: 1.4766, 8:
        #           1.9141, 9: 2.4063, 10: 2.7305, 11: 3.3223, 12: 3.9023, 13: 4.5234, 14: 5.1152, 15: 5.5547}
        # self.bw_to_PRBs = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100} # 3GPP document 36.213, Table 7.1.7.1-1, Table 7.1.7.2.1-1 and Table 7.1.7.2.2-1
        # Default bw: 180 kHz (1 RB)
        self.bw_per_RB = self.numerology_to_SCS[self.numerology] * self.num_of_SCS_per_RB #(Hz) #Depends on the numerology i.e. subcarrier spacing
        self.num_fadingblocks_per_subband = RB_per_Subband
        self.tx_bw_per_subband = self.bw_per_RB * self.num_fadingblocks_per_subband
        
        # Traffic model and radio communication
        #self.numChannels = 1            # Number of channels
        self.numSubbands = numSubbands          # Number of channel groups
        self.numLevels = level
        self.fc = 10e9                   # Carrier frequency [GHz]
        self.wave = 3e8 / self.fc       # Carrier wavelength [m]
        self.fd = self.speed / self.wave # Max Doppler frequency shift [Hz]
        #self.totBW = self.numSubbands*10*10**6# Total bandwidth [Hz]
        self.noiseFactor = 10           # Noise power factor
        self.noisePower = 10**((-174+self.noiseFactor+10*np.log10(self.bw_per_RB))/10)
        self.txPow = -10                # TX power
        self.rho =  np.float32(bessel(2*np.pi*self.fd*self.sampTime)) # Fading parameter
        #self.bw_to_PRBs = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}
        #self.totalbw = 3 #MHz
        #self.channelBW = (self.totalbw * 10**6) /(self.numSubbands*self.numChannels) # Bandwidth of channels
        
        #self.intPowMax = -14.096877011071049 if self.clutter == 'dense' else -11.634749182802036
        
        self.intPowMax =  -20 if 'dense' in self.clutter else -20 #-52.66387821883611   #-48.10078099565358
        self.intPowNorm = self.intPowMax - 10*np.log10(self.noisePower) # Min-max normalisation constant
        self.indPowMin = - 90
        self.indPowNorm = -30 + 90
        self.sum_int_history = np.empty([self.numCell,self.numSubbands,1])

        self.sinrMax = 43.95373571433217 if 'dense' in self.clutter else 43.73062810851178
        self.sinrMin = 11.380395703242927 if 'dense' in self.clutter else 4.784518853326086
        self.sinrNorm = self.sinrMax - self.sinrMin

        # Action space
        self.powerLevels = np.array([self.txPow]) # Initial power levels for subnetworks [dBm]
        self.channels = np.array([i for i in range(self.numSubbands)]) # Set of channels
        self.combAction = np.array(np.meshgrid(self.channels, self.powerLevels)).T.reshape(-1,2) # Combined action array
        self.obs_size = obs_size
        self.multichannel_action = {0:[0,0,0],1:[1,1,1],2:[2,2,2],3:[0,1,1],4:[0,2,2],5:[1,2,2],6:[0,1,2]}
        self.centralized_action = {0:np.array([0,0,0,0]),1:np.array([1,0,0,0]),2:np.array([1,1,0,0]),3:np.array([1,1,1,0]),4:np.array([1,0,1,0])}
        self.n_actions = 5

        # Optimisation constraints and reward function
        #self.reqRate = np.full(self.numCell, rate) # Minimum throughputs (Requirement) [Mbps]
        #self.pOutage = np.full(self.numCell, pOut) # Outage probabilities (Requirement) 
        #self.Pmax = 0 # Maximum sum of transmission power [dBm]
        #self.w_next = np.zeros([self.numCell, self.numDev]) # Initial weight factors for reliability reward signals
        #self.p_next = np.zeros(self.numCell) # Initial weight factors for power reward signals

        self.rate_req = np.full([self.numCell, self.numDev], 11)
        self.SINR_req = 10 * np.log10(2**self.rate_req - 1)
        self.Pmax = 0 # [dBm]
        self.Pmin = 0 # [dBm]

        print(f'Env: code/src/env/infactory_env.py')
        print(f'     Factory with n={self.numCell} subnetworks and m={self.numDev} devices in {self.clutter} clutter.')
        print(f'     {self.problem.capitalize()} allocation for k={self.numSubbands} channels and u={self.numLevels} power levels.')
        print(f'     Action space for channels={self.channels} and power={self.powerLevels}.')

        if self.problem == 'joint':
            self.Plevels = np.linspace(start=self.Pmin, stop=self.Pmax, num=self.numLevels)
            self.comb_act = np.array(np.meshgrid(np.arange(self.numSubbands), self.Plevels)).T.reshape(-1,2)
            print(f'     Joint={self.comb_act}.')

        self.generate_factory_map()
        #Intialize plants
        self.initialize_plants()
    
    def initialize_plants(self):
        self.activity_indicator = np.ones((self.numSteps, self.numCell))
        self.buffer_size_list = []
        self.subn_plant_control = control_env.WNCSEnv(ext_env = self, n_UEs=self.numCell, bw_per_RB=self.bw_per_RB)
        self.subn_plant_control.reset()
        self.plant_states_list = [] #np.zeros((self.numSteps, self.numCell, 4))
        self.force_list = []
        self.lqr_list = []
        self.lqr = 0
        
        


    def reset(self):
        self.initialize_plants()
        self.generate_mobility()
        self.generate_channel()
        self.sum_int_history = np.empty([self.numCell,self.numSubbands,1])

    def init_clutter(self, clutter):
        """
        Get clutter parameters based on sparse or dense scenario.
        """
        self.clutType = clutter         # Type of clutter (sparse or dense)
        self.shadStd = [4.0]            # Shadowing std (LoS)
        if clutter == 'sparse0':
            self.clutSize = 10.0        # Clutter element size [m]  
            self.clutDens = 0.1         # Clutter density [%]
            self.shadStd.append(5.7)    # Shadowing std (NLoS)
        elif clutter == 'sparse':
            self.clutSize = 10.0        # Clutter element size [m]  
            self.clutDens = 0.2         # Clutter density [%]
            self.shadStd.append(5.7)    # Shadowing std (NLoS)
        elif clutter == 'sparse2':
            self.clutSize = 10.0        # Clutter element size [m]  
            self.clutDens = 0.35         # Clutter density [%]
            self.shadStd.append(5.7)    # Shadowing std (NLoS)
        elif clutter == 'dense0':
            self.clutSize = 2.0         # Clutter element size [m]  
            self.clutDens = 0.45         # Clutter density [%]
            self.shadStd.append(7.2)    # Shadowing std (NLoS)
        elif clutter == 'dense':
            self.clutSize = 2.0         # Clutter element size [m]  
            self.clutDens = 0.6         # Clutter density [%]
            self.shadStd.append(7.2)    # Shadowing std (NLoS)
        elif clutter == 'dense2':
            self.clutSize = 2.0         # Clutter element size [m]  
            self.clutDens = 0.8        # Clutter density [%]
            self.shadStd.append(7.2)    # Shadowing std (NLoS)
        else:
            raise NotImplementedError

    # =================== Mobility Model Functions =================== #
    def generate_factory_map(self, fname='code/src/env/Factory_values.mat'):
        """
        Generate the predefined factory map. All intersections (path points) is defined
        as points in a plane, and the matrix links define the legal moves between points.
        A global parameter on the shadowing map will be fetched from a local file.
        
        Param :fname: File name for shadowing values (string)
        Return :path_points: Waypoints (list)
        Return :link: Legal moves between waypoints (np.array)
        """

        # Load shadowing values
        if 'cMapFFT' not in globals() and os.path.exists(f'./{self.fname}'):
            dic = loadmat(self.fname)
            global cMapFFT
            cMapFFT = dic['data']

    def generate_mobility(self):
        """
        Random generation of states for every timestep. Initialisation is random on legal
        map paths, where robots are seperated with minimum 3 meters. The task of a robot
        is random, based on legal moves. Collisions between robots with a common destination
        are avoided by stopping every robot within minimum seperation distance of the robot
        with the shortest distance to the destination.
                
        Param :path_points: Waypoints (list)
        Param :link: Legal moves between waypoints (np.array)
        Return :loc: State locations (np.array)
        """
        if self.mobile == False:    
            # Initialise random start positions
            N = round(self.updateT/self.sampTime)
            self.loc = np.zeros([2,self.numCell,N],dtype=np.float64)
            bound = self.deploy_length - self.cellDia
            X = np.zeros([self.numCell,1],dtype=np.float64)
            Y = np.zeros([self.numCell,1],dtype=np.float64)
            dist_2 = self.minDistance**2
            loop_terminate = 1
            nValid = 0
            while nValid < self.numCell and loop_terminate < 1e6:
                newX = bound*(np.random.uniform()-0.5)
                newY = bound*(np.random.uniform()-0.5)
                if all(np.greater(((X[0:nValid] - newX)**2 + (Y[0:nValid] - newY)**2),dist_2)):
                    X[nValid] = newX
                    Y[nValid] = newY
                    nValid = nValid+1
                loop_terminate = loop_terminate+1
            if nValid < self.numCell:
                print("Invalid number of subnetworks for deploy size")
                exit
            #Location of the access points
            X = X+self.deploy_length/2
            Y = Y+self.deploy_length/2
            AP_loc = np.concatenate((X, Y), axis=1)
            self.loc = np.dstack([AP_loc.T]*N) 
        else:
            X = np.zeros([self.numCell,1],dtype=np.float64)
            Y = np.zeros([self.numCell,1],dtype=np.float64)
            gwLoc = np.zeros([self.numCell,2],dtype=np.float64)
            XBound1 = self.factoryArea[0]-2*self.subnet_radius
            YBound1 = self.factoryArea[1]-2*self.subnet_radius
            dist_2 = self.minDistance**2
            loop_terminate = 1
            nValid = 0
            while nValid < self.numCell and loop_terminate < 1e6:
                newX = XBound1*(np.random.uniform()-0.5)
                newY = YBound1*(np.random.uniform()-0.5)
                if all(np.greater(((X[0:nValid] - newX)**2 + (Y[0:nValid] - newY)**2),dist_2)):
                    X[nValid] = newX
                    Y[nValid] = newY
                    nValid = nValid+1
                loop_terminate = loop_terminate+1
            if nValid < self.numCell:
                return -1
            gwLoc[:,0] = X.T
            gwLoc[:,1] = Y.T
            self.gwLoc = gwLoc
            Xtemp = gwLoc[:,0]+self.factoryArea[0]/2
            Ytemp = gwLoc[:,1]+self.factoryArea[1]/2
            Xway = self.factoryArea[0]/2
            Yway = self.factoryArea[1]/2
            N = round(self.updateT/self.sampTime)
            D = np.arctan2(Yway-Ytemp,Xway-Xtemp).reshape(-1)
            loc = np.zeros([2,self.numCell,N],dtype=np.float64)
            loc[0,:,0] = Xtemp
            loc[1,:,0] = Ytemp
            Imat = np.zeros([self.numCell,self.numCell],dtype=np.float64)
            np.fill_diagonal(Imat, 190*np.ones([self.numCell,1]))
            for n in range(1,N):
                Xtemp = Xtemp+np.multiply(np.cos(D),self.speed)*self.sampTime
                Ytemp = Ytemp+np.multiply(np.sin(D),self.speed)*self.sampTime
                loc_temp = np.array([Xtemp,Ytemp])
                dist_pw = cdist(loc_temp.T,loc_temp.T)
                dist_pw = dist_pw+Imat
                indx2,indx1 = np.where(dist_pw <= self.minDistance)
                indx11,elm = np.unique(indx1,return_index=True)
                #num_unique = np.shape(indx11)[0]
                D[indx11] = D[indx11]+np.pi #np.random.uniform(low=0.0,high=1.0,size=num_unique)*2*np.pi
                Xtemp[indx11] = Xtemp[indx11]+np.multiply(np.cos(D[indx11]),self.speed)*self.sampTime
                Ytemp[indx11] = Ytemp[indx11]+np.multiply(np.sin(D[indx11]),self.speed)*self.sampTime
                Xtemp[np.where(Xtemp < self.subnet_radius)] = self.subnet_radius
                Xtemp[np.where(Xtemp > XBound1)] = XBound1
                Ytemp[np.where(Ytemp < self.subnet_radius)] = self.subnet_radius
                Ytemp[Ytemp > XBound1] = XBound1
                D[np.where(Xtemp==self.subnet_radius)] = np.random.uniform(low=0.0,high=1.0,size=len(Xtemp[np.where(Xtemp==self.subnet_radius)]))*2*np.pi
                D[np.where(Xtemp==XBound1)] = np.random.uniform(low=0.0,high=1.0,size=len(Xtemp[np.where(Xtemp==XBound1)]))*2*np.pi
                D[np.where(Ytemp==self.subnet_radius)] = np.random.uniform(low=0.0,high=1.0,size=len(Ytemp[np.where(Ytemp==self.subnet_radius)]))*2*np.pi
                D[np.where(Ytemp==YBound1)] = np.random.uniform(low=0.0,high=1.0,size=len(Ytemp[np.where(Ytemp==YBound1)]))*2*np.pi
                loc[0,:,n] = Xtemp
                loc[1,:,n] = Ytemp
                #print(np.min(cdist(np.array([Xtemp,Ytemp]).T,np.array([Xtemp,Ytemp]).T)+Imat))
            self.loc = loc   

    def deploy_devices(self):
        """
        Deploy an equal number of devices for each subnetwork. Locations are static
        relative to the APs.

        Return :devLoc: Device locations (np.array)
        """
        devLoc = np.zeros((2, self.numCell * self.numDev), dtype=np.float64)
        for i in range(self.numDev):
            loc_angle = np.random.uniform(low=0.0, high=1.0, size=self.numCell) * 2 * np.pi
            dist_rand = np.random.uniform(low=1, high=self.subnet_radius, size=self.numCell)
            devLoc[:,i*self.numCell:(i+1)*self.numCell] = np.array([dist_rand * np.cos(loc_angle),
                                                                    dist_rand * np.sin(loc_angle)])
        return devLoc
    
    # =================== Channel Model Functions =================== #
    def channel_pathLoss(self, dist):
        """
        Calculate path loss of a link in factories based on 3GPP.

        Return :Gamma: Path loss (float)
        """
        PrLoS = np.exp(dist * np.log(1 - self.clutDens) / self.clutSize)
        NLoS = PrLoS <= (1 - PrLoS)
        idx = np.logical_not(np.eye(dist.shape[0]), dtype=bool)
        Gamma = np.zeros(dist.shape)
        Gamma[idx] = 31.84 + 21.5 * np.log10(dist[idx]) + 19 * np.log10(self.fc/1e9)
        Gamma_min = 31.84 + 21.5 * np.log10(1) + 19 * np.log10(self.fc/1e9) 
        if self.clutType == 'sparse':
            Gamma[NLoS] = np.max([Gamma[NLoS], 
                                33 + 25.5 * np.log10(dist[NLoS]) + 20 * np.log10(self.fc/1e9)], 
                                axis=0)
        elif self.clutType == 'dense':
            Gamma[NLoS] = np.max([Gamma[NLoS], 
                                33 + 25.5 * np.log10(dist[NLoS]) + 20 * np.log10(self.fc/1e9),
                                18.6 + 35.7 * np.log10(dist[NLoS]) + 20 * np.log10(self.fc/1e9)], 
                                axis=0)
        Gamma[Gamma < Gamma_min] = Gamma_min
        return Gamma#10**(Gamma/10)

    def init_shadow(self):
        """
        Generate covariance map of correlated shadowing values.

        Return :map: Correlated shadowing map (np.array)
        """
        nx, ny = [len(self.mapX), len(self.mapY)]
        # Generate covariance map
        if 'cMapFFT' not in globals(): # Calculate this part only once
            print('Calculating new shadowing map.')
            cMap = np.zeros([nx, ny], dtype=np.float64)
            for x in range(nx):
                for y in range(ny):
                    cMap[x,y]= np.exp((-1) \
                                    * np.sqrt(np.min([np.absolute(self.mapX[0]-self.mapX[x]),
                                                    np.max(self.mapX)-np.absolute(self.mapX[0]-self.mapX[x])])**2 \
                                            + np.min([np.absolute(self.mapY[0]-self.mapY[y]),
                                                    np.max(self.mapY)-np.absolute(self.mapY[0]-self.mapY[y])])**2) \
                                    / self.corrDist)

            global cMapFFT
            cMapFFT = np.fft.fft2(cMap * self.shadStd[1])
        Z = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
        map = np.real(np.fft.fft2(np.multiply(np.sqrt(cMapFFT), Z) / np.sqrt(nx * ny))) * np.sqrt(2)
        return map

    def channel_shadow(self, allLoc, dist, map):
        """
        Compute channel correlated shadowing values

        Param :allLoc: Locations of APs and devices (np.array)
        Param :dist: Distance matrix for subnetworks (np.array)
        Param :map: Correlated shadowing map (np.array)
        Return :Psi: Shadowing value matrix (np.array)
        """
        # Convert locations to the shadowing map
        idx = np.array(np.round(allLoc[0,:], 1) / self.mapRes, dtype=int)
        idy = np.array(np.round(allLoc[1,:], 1) / self.mapRes, dtype=int)
        ids = np.ravel_multi_index([idx,idy], map.shape)

        # Calculate shadowing values
        f = (map.flatten()[ids]).reshape(1,-1)
        f_AB = np.add(f.T, f)
        temp = np.exp((-1) * dist / self.corrDist)
        Psi = np.multiply(np.divide(1 - temp, np.sqrt(2) * np.sqrt(1 + temp)), f_AB)  
        return Psi#10**(Psi/10)

    def channel_fading(self, direction='ul',time_index=0):
        """
        Compute channel fading values with Jake's doppler model.
        Rho is the precalculated zeroth order bessel function J0(2*pi*fd*td).

        Param :time_index: The instantanious time-slot (integer)
        Return :h: Small-scale fading values (np.array)
        """
        nTot = self.numCell * (1 + self.numDev)
        if direction == 'ul':
            if time_index == 0 or self.mobile == False:
                self.ul_h = np.sqrt(0.5 * (np.random.randn(nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband)**2 \
                                    + 1j * np.random.randn(nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband)**2))
            else:
                self.ul_h = self.ul_h * self.rho + np.sqrt(1. - self.rho**2) * 0.5 * (np.random.randn(nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband)**2\
                                                                            + 1j * np.random.randn(nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband)**2)
            return np.abs(self.ul_h)
        
        elif direction == 'dl':
            if time_index == 0 or self.mobile == False:
                self.dl_h = np.sqrt(0.5 * (np.random.randn(nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband)**2 \
                                    + 1j * np.random.randn(nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband)**2))
            else:
                self.dl_h = self.dl_h * self.rho + np.sqrt(1. - self.rho**2) * 0.5 * (np.random.randn(nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband)**2\
                                                                            + 1j * np.random.randn(nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband)**2)
            return np.abs(self.dl_h)


    def generate_channel(self):   #To revisit
        """
        Compute Rx powers for all timesteps based on channel model.
        """
        # Compute locations of all APs and devices
        UL_devLoc = self.deploy_devices()
        UL_dLoc = np.repeat(UL_devLoc[:,:,np.newaxis], self.numSteps, axis=2) \
             + np.repeat(self.loc, self.numDev, axis=1)
        UL_allLoc = np.concatenate((UL_dLoc, self.loc), axis=1) # [m1, m2, .., mM, n1, n2, ..., nN]

        DL_devLoc = self.deploy_devices()
        DL_dLoc = np.repeat(DL_devLoc[:,:,np.newaxis], self.numSteps, axis=2) \
             + np.repeat(self.loc, self.numDev, axis=1)
        DL_allLoc = np.concatenate((DL_dLoc, self.loc), axis=1) 

        map = self.init_shadow() # Initialise correlated shadowing map
        nTot = self.numCell * (1 + self.numDev)
        self.ul_rxPow = np.zeros([nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband, self.numSteps], dtype=np.float64)
        self.ul_dist = np.zeros([nTot, nTot, self.numSteps], dtype=np.float64)
        self.dl_rxPow = np.zeros([nTot, nTot, self.numSubbands,self.num_fadingblocks_per_subband, self.numSteps], dtype=np.float64)
        self.dl_dist = np.zeros([nTot, nTot, self.numSteps], dtype=np.float64)
        for time_index in range(self.numSteps):
            # Calculate distances
            ul_nLoc = UL_allLoc[:,:,time_index]
            self.ul_dist[:,:,time_index] = cdist(ul_nLoc.T, ul_nLoc.T)
            # Calculate general losses
            ul_Gamma = self.channel_pathLoss(self.ul_dist[:,:,time_index])
            ul_Psi = self.channel_shadow(ul_nLoc, self.ul_dist[:,:,time_index], map)
            ul_h = self.channel_fading('ul',time_index)
            # Calculate Rx power
            self.ul_rxPow[:,:,:,:,time_index] = np.expand_dims(ul_Psi + ul_Gamma, axis=(2,3)) + 2 * ul_h

            dl_nLoc = DL_allLoc[:,:,time_index]
            self.dl_dist[:,:,time_index] = cdist(dl_nLoc.T, dl_nLoc.T)
            # Calculate general losses
            dl_Gamma = self.channel_pathLoss(self.dl_dist[:,:,time_index])
            dl_Psi = self.channel_shadow(dl_nLoc, self.dl_dist[:,:,time_index], map)
            dl_h = self.channel_fading('dl',time_index)
            # Calculate Rx power
            self.dl_rxPow[:,:,:,:,time_index] = np.expand_dims(dl_Psi + dl_Gamma, axis=(2,3)) + 2 * dl_h

    # =================== Problem based step functions =================== #
    def channel_step(self, chl_action, time_index):

        #pow_action = self.powerLevels * np.ones(self.numCell)
        pow_action = self.Pmin * np.ones(self.numCell)
        ul_rxPow = self.ul_rxPow[:,:,time_index]

        # Interference power        
        intPow = np.zeros([self.numCell,self.numSubbands],dtype=np.float64)
        for k in range(self.numSubbands):
            indm = np.argwhere(chl_action == k)
            for n in range(self.numCell):
                intPow[n,k] = np.sum(10**((pow_action[n]-ul_rxPow[n,self.numCell+indm[indm !=n]])/10)) + self.noisePower

        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell, self.numSubbands], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        for n in range(self.numCell):
            for k in range(self.numSubbands):
                self.SINRAll[n,k] = (10**((pow_action[n] - ul_rxPow[n,self.numCell+n])/10)) / intPow[n,k]
            self.SINR[n] = self.SINRAll[n,chl_action[n]]
            self.sRate[n] = np.log2(1 + self.SINR[n])
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        self.rate_list[time_index] = 1/self.numSubbands * self.sRate

        return intPow
    

    def centralized_channel_step(self, action, time_index):
        if self.numCell != 4:
            raise NotImplementedError
        #print(action)
        if isinstance(action, np.ndarray):
            action = action.item()
        chl_action = self.centralized_action[action]
        pow_action = self.Pmin * np.ones(self.numCell)
        ul_rxPow = self.ul_rxPow[:,:,time_index]
        self.ch_choice = np.zeros((self.numCell,self.numSubbands))

        for n in range(self.numCell):
            self.ch_choice[n,chl_action[n]] = 1

        # Interference power        
        intPow = np.zeros([self.numCell,self.numSubbands],dtype=np.float64)
        for k in range(self.numSubbands):
            indm = np.argwhere(chl_action == k)
            for n in range(self.numCell):
                intPow[n,k] = np.sum(10**((pow_action[n]-ul_rxPow[n,self.numCell+indm[indm !=n]])/10)) + self.noisePower
        
        self.potentialIntMat = self.Pmin-ul_rxPow[0:self.numCell,self.numCell:]
        self.normPotentialIntMat = (self.potentialIntMat - self.indPowMin)/self.indPowNorm
        self.actualIntMat = np.zeros((self.numCell,self.numCell))
        for n in range(self.numCell):
            for m in range(self.numCell):
                if chl_action[n] == chl_action[m]:
                    self.actualIntMat = self.normPotentialIntMat[n,m]  
        
        self.IntDistMat = np.zeros((self.numCell,self.numCell-1))
        ul_dist = self.ul_dist[0:self.numCell, self.numCell:, time_index]
        for n in range(self.numCell):
            self.IntDistMat[n] = ul_dist[n,np.arange(self.numCell)!=n]


        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell, self.numSubbands], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        for n in range(self.numCell):
            for k in range(self.numSubbands):
                self.SINRAll[n,k] = (10**((pow_action[n] - ul_rxPow[n,self.numCell+n])/10)) / intPow[n,k]
            self.SINR[n] = self.SINRAll[n,chl_action[n]]
            self.sRate[n] = np.log2(1 + self.SINR[n])
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        self.rate_list[time_index] = 1/self.numSubbands * self.sRate
        return intPow
    

    def centralized_power_step(self, action, time_index):
        #print(action)
    
        chl_action = self.centralized_action[0]
        pow_action = 10*np.log10(action) #* np.ones(self.numCell)
        ul_rxPow = self.ul_rxPow[:,:,time_index]


        # Interference power        
        intPow = np.zeros([self.numCell,],dtype=np.float64)

        for n in range(self.numCell):
            m = np.arange(self.numCell)[np.arange(self.numCell)!=n]
            #print(ul_rxPow[n,self.numCell+m])
            intPow[n] = np.sum(10**((pow_action[m]-ul_rxPow[n,self.numCell+m])/10)) + self.noisePower
        
        self.potentialIntMat = intPow
        self.actualIntMat = intPow 
        
        self.IntDistMat = np.zeros((self.numCell,self.numCell-1))
        ul_dist = self.ul_dist[0:self.numCell, self.numCell:, time_index]
        for n in range(self.numCell):
            self.IntDistMat[n] = ul_dist[n,np.arange(self.numCell)!=n]

        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        for n in range(self.numCell):
            self.SINRAll[n] = (10**((pow_action[n] - ul_rxPow[n,self.numCell+n])/10)) / intPow[n]
            self.SINR[n] = self.SINRAll[n]
            self.sRate[n] = np.log2(1 + self.SINR[n])
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        self.rate_list[time_index] = self.sRate
        return intPow
    
    def heuristic_control_aware_round_robin_step(self, _SRate, time_index):
        active_set = self.activity_indicator[time_index] #You are not active if you are not transmitting or if you have failed
        #pow_action = 10*np.log10(action) #* np.ones(self.numCell)
        ul_rxPow = self.ul_rxPow[0:self.numCell,self.numCell:,time_index]
        num_active = int(np.sum(active_set))
        active_rxPow = np.zeros((num_active, num_active))
        valid_plants = np.arange(self.numCell)[active_set.astype(bool)]
        #print('valid plants', valid_plants)
        self.sRate = np.zeros([num_active],dtype=np.float64)
        for n in range(num_active):
            self.sRate[n] = _SRate[n]
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
       
        self.rate_list.append(self.sRate)
        self.mean_lqr_cost, self.lqr, self.plant_states, self.force, self.buffer_size,self.sensor_control_aoi, self.control_sensor_aoi = self.subn_plant_control.run(self.sRate, self.bw_to_PRBs[self.totalbw], self.numSubbands, np.ones_like(self.sRate))
        self.lqr_list.append(self.lqr)
        self.plant_states_list.append(self.plant_states)
        self.force_list.append(self.force)
        self.buffer_size_list.append(self.buffer_size)

    def heuristic_control_aware_centralized_power_step(self, action, time_index):
        self.numSubbands = 1
        active_set = self.activity_indicator[time_index] #You are not active if you are not transmitting or if you have failed
        pow_action = 10*np.log10(action) #* np.ones(self.numCell)
        ul_rxPow = self.ul_rxPow[0:self.numCell,self.numCell:,time_index]

        self.des_chgain = np.diag(ul_rxPow)
        num_active = int(np.sum(active_set))
        active_rxPow = np.zeros((num_active, num_active))
        valid_plants = np.arange(self.numCell)[active_set.astype(bool)]
        #print('valid plants', valid_plants)
        i = 0
        j = 0
        for n in valid_plants:
            for m in valid_plants:
                #print(i,j)
                active_rxPow[int(i),int(j)] = ul_rxPow[int(n),int(m)]
                j += 1
            i += 1
            j = 0
        #print(ul_rxPow)
        #print(active_rxPow)
        self.channel_gain = (10**(np.negative(active_rxPow)/10)) / self.noisePower
        # Interference power        
        intPow = np.zeros([num_active,],dtype=np.float64)

        for n in range(num_active):
            m = np.arange(num_active)[np.arange(num_active)!=n]
            #print(ul_rxPow[n,self.numCell+m])
            intPow[n] = np.sum(10**((pow_action[m]-active_rxPow[n,m])/10)) + self.noisePower
        
        self.potentialIntMat = intPow
        self.actualIntMat = intPow 
        
        # self.IntDistMat = np.zeros((self.numCell,self.numCell-1))
        # ul_dist = self.ul_dist[0:self.numCell, self.numCell:, time_index]
        # for n in range(self.numCell):
        #     self.IntDistMat[n] = ul_dist[n,np.arange(self.numCell)!=n]

        # SINR and channel rates
        self.SINRAll = np.zeros([num_active], dtype=np.float64)
        self.SINR = np.zeros([num_active],dtype=np.float64)
        self.sRate = np.zeros([num_active],dtype=np.float64)
        self.sRate_full = np.zeros([self.numCell],dtype=np.float64)
        for n in range(num_active):
            self.SINRAll[n] = (10**((pow_action[n] - active_rxPow[n,n])/10)) / intPow[n]
            self.SINR[n] = self.SINRAll[n]
            self.sRate[n] = np.log2(1 + self.SINR[n])
            self.sRate_full[valid_plants[n]] = np.log2(1 + self.SINR[n])
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        self.rate_list.append(self.sRate)
        #print('intpow ', intPow)
        #print(10**((pow_action[n] - active_rxPow)/10))
        #print('Original rate',self.sRate)
        
        self.mean_lqr_cost, self.lqr, self.plant_states, self.force, self.buffer_size,self.sensor_control_aoi, self.control_sensor_aoi = self.subn_plant_control.run(self.sRate_full, self.bw_to_PRBs[self.totalbw], self.numSubbands, self.numSubbands*np.ones_like(self.sRate_full))
        self.lqr_list.append(self.lqr)
        self.plant_states_list.append(self.plant_states)
        self.force_list.append(self.force)
        self.buffer_size_list.append(self.buffer_size)
        return intPow  

    def simple_control_aware_centralized_power_step(self, action, time_index):
        self.numSubbands = 1
        active_set = self.activity_indicator[time_index] #You are not active if you are not transmitting or if you have failed
        valid_plants = list(np.nonzero(active_set)[0])
        chgain_mat = torch.tensor((10**(self.ul_rxPow[0:self.numCell,self.numCell:,time_index]/10)).reshape([-1,self.numCell,self.numCell,1]))
        tx_powers = torch.zeros(self.numCell, dtype=torch.float64)
        tx_powers[valid_plants] = torch.tensor(action, dtype=torch.float64)
        tx_powers = tx_powers.reshape([-1,1,self.numCell,1])
        ul_rxPow = torch.divide(tx_powers.T,chgain_mat)
        num_active = int(np.sum(active_set))
        eye = torch.eye(self.numCell)
        power_mat = ul_rxPow.reshape([-1,self.numCell,self.numCell,1]) #same as power - active_rxpow
        Rcv_power = torch.mul(power_mat.squeeze(-1),eye) 
        #print(Rcv_power.shape)
        self.des_chgain = np.array(torch.sum(Rcv_power,dim=-1)).flatten()
        Interference_mat = torch.sum(torch.mul(power_mat.squeeze(-1),1-eye), dim=1) + self.noisePower
        
        sinr = torch.sum(torch.divide(Rcv_power,Interference_mat), dim=1)
        self.sRate = torch.log2(1 + sinr)
        self.SINR = 10 * np.log10(sinr)
        self.sRate = np.array(self.sRate)[0]
        self.rate_list.append(self.sRate)
        self.sRate_full = self.sRate
        #print(self.sRate)
        # pow_action = np.zeros(self.numCell)
        # pow_action[valid_plants] = action
        # subband_action = np.ones((self.numCell,self.numSubbands)) * np.array([[0,1,2]])
        # self.traffic_aware_joint_power_channel_step(pow_action, subband_action, time_index)

        self.mean_lqr_cost, self.lqr, self.plant_states, self.force, self.buffer_size,self.sensor_control_aoi, self.control_sensor_aoi = self.subn_plant_control.run(self.sRate, self.bw_to_PRBs[self.totalbw], self.numSubbands, self.numSubbands*np.ones_like(self.sRate_full))
        self.lqr_list.append(self.lqr)
        self.plant_states_list.append(self.plant_states)
        self.force_list.append(self.force)
        self.buffer_size_list.append(self.buffer_size)
        return Interference_mat     
    
    
    def traffic_aware_channel_step(self, chl_action, time_index):

        active_set = self.activity_indicator[time_index]
        pow_action = (10**(self.Pmin/10)) * active_set  #convert power to linear, only active subn has transmit power > 0
        
        ul_rxPow = self.ul_rxPow[:,:,time_index]
        #print(pow_action)
        # Interference power        
        intPow = np.zeros([self.numCell,self.numSubbands],dtype=np.float64)
        for k in range(self.numSubbands):
            b = []
            indm = np.argwhere(chl_action == k)
            indm_active = active_set[indm]
            for i in range(len(indm_active)):
                if indm_active[i] == 1:  #select only active subnetwork that are on channel k
                    b.append(indm[i])
            b = np.array(b, dtype=int)
            for n in range(self.numCell):
                intPow[n,k] = np.sum(10**(self.Pmin/10) / 10**((ul_rxPow[n,self.numCell+b[b !=n]])/10)) + self.noisePower

        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell, self.numSubbands], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        for n in range(self.numCell):
            for k in range(self.numSubbands):
                self.SINRAll[n,k] = (pow_action[n] / 10**((ul_rxPow[n,self.numCell+n])/10)) / intPow[n,k]
            self.SINR[n] = self.SINRAll[n,int(chl_action[n])]
            self.sRate[n] = np.log2(1 + self.SINR[n])
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        self.rate_list[time_index] = 1/self.numSubbands * self.sRate

        
        self.mean_lqr_cost, self.lqr, self.plant_states, self.buffer_size = self.subn_plant_control.run(self.sRate, self.bw_to_PRBs[self.totalbw], self.numSubbands, np.ones_like(self.sRate))
        #print(self.subn_plant_control.activity_indicator)
        self.lqr_list[time_index] = self.lqr
        self.plant_states_list[time_index] = self.plant_states
        self.buffer_size_list[time_index] = self.buffer_size
        self.intPow = intPow

        return intPow
    
    def traffic_aware_multi_channel_step(self,chl_action,time_index):
        active_set = self.activity_indicator[time_index]
               
        num_active = int(np.sum(active_set))
        pow_action = (10**(self.Pmin/10)) * np.ones(num_active)  #convert power to linear, only active subn has transmit power > 0
        num_selected_subbands = np.zeros(num_active)
        ul_rxPow = self.ul_rxPow[0:self.numCell,self.numCell:,time_index]
        multi_channel_decision = {} #Dictionary of subnetworks as key and operating subband or subbands as item
        valid_plants = np.arange(self.numCell)[active_set.astype(bool)]
        active_rxPow = np.zeros((num_active, num_active))

        for i in range(num_active):
            multi_channel_decision[i] = self.multichannel_action[chl_action[i]]
            if hasattr(multi_channel_decision[i], "__len__"):
                num_selected_subbands[i] = len(multi_channel_decision[i])
            elif multi_channel_decision[i] == -1:
                active_set[i] = 0
                num_selected_subbands[i] = 1
            else:
                num_selected_subbands[i] = 1
            #print(num_selected_subbands[i])
                
        i = 0
        j = 0
        for n in valid_plants:
            for m in valid_plants:
                #print(i,j)
                active_rxPow[int(i),int(j)] = ul_rxPow[int(n),int(m)]
                j += 1
            i += 1
            j = 0
        intPow = np.zeros([num_active,self.numSubbands],dtype=np.float64)
        #print('In 1 Multichannel decision = ',multi_channel_decision)
        for k in range(self.numSubbands):
            b = []
            indm = np.array([keys for keys in multi_channel_decision if np.any(multi_channel_decision[keys] == k)], dtype=np.int32)
            #print('In 1 indm = ',indm)
            indm_active = np.ones(num_active)[indm]
            #print('In 1 indm_active = ',indm_active)
            for i in range(len(indm_active)):
                if indm_active[i] == 1:  #select only active subnetwork that are on channel k
                    b.append(indm[i])
            b = np.array(b, dtype=int)
            for n in range(num_active):
                intPow[n,k] = np.sum(10**(self.Pmin/10) / 10**((active_rxPow[n, b[b !=n]])/10)) + self.noisePower
            #print('In 1 b = ',b)
        #print('In 1 ul_rxPow ', 10**(self.Pmin/10) / 10**((active_rxPow)/10))
        #print('In 1 IntPow ', intPow)

        # SINR and channel rates
        self.SINRAll = np.zeros([num_active, self.numSubbands], dtype=np.float64)
        self.SINR = np.zeros([num_active],dtype=np.float64)
        self.sRate = np.zeros([num_active],dtype=np.float64)
        for n in range(num_active):
            for k in range(self.numSubbands):
                self.SINRAll[n,k] = (10**(self.Pmin/10) / 10**((active_rxPow[n,n])/10)) / intPow[n,k]
            #print('all SINR ', self.SINRAll[n])
            #print('channel choice ',multi_channel_decision[n])
            
            self.SINR[n] = np.sum(self.SINRAll[n,multi_channel_decision[n]])
            self.sRate[n] = np.sum(np.log2(1 + self.SINRAll[n,multi_channel_decision[n]]))/num_selected_subbands[n]
            #print('In 1 - SINR', self.SINRAll[n,multi_channel_decision[n]])
            #print('Alt SE', np.sum(np.log2(1+self.SINRAll[n,multi_channel_decision[n]])))
            #print('Rate ', ((180000 * (15/3)*(1))  * self.sRate[n]))
            #print(' ')
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
        
        indx = np.array([keys for keys in multi_channel_decision if np.any(multi_channel_decision[keys] == -1)], dtype=np.int32) #select subnetwork that chose no channel, and set their rate to 0
        self.sRate[indx] = 0
        #print('Rate all',np.log2(1+ self.SINRAll))
        #print('selected sRate ',self.sRate)

        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        
        #self.rate_list.append((180000 * (self.bw_to_PRBs[self.totalbw]/self.numSubbands)*(num_selected_subbands)) * self.sRate)

        self.mean_lqr_cost, self.lqr, self.plant_states, self.force, self.buffer_size,self.sensor_control_aoi, self.control_sensor_aoi = self.subn_plant_control.run(self.sRate, self.bw_to_PRBs[self.totalbw], self.numSubbands, num_selected_subbands)
        #print(self.subn_plant_control.activity_indicator)
        self.lqr_list.append(self.lqr)
        self.plant_states_list.append(self.plant_states)
        self.force_list.append(self.force)
        self.buffer_size_list.append(self.buffer_size)
    
        self.intPow = intPow

        return intPow
    
    def simple_traffic_aware_single_channel_step(self,chl_action,time_index):
        active_set = self.activity_indicator[time_index]
        #print('ch ', chl_action)
        #chl_action = list(map(self.multichannel_action.get,chl_action))
        #print(chl_action.shape)
        out2 = np.eye(self.numSubbands)[chl_action.astype('int')] 
        #print(out2.shape)
        #print('out2 any shape',out2.any(axis=1).astype('int').shape)
        out2 = out2.any(axis=1).astype('int') * np.expand_dims(active_set,-1)
        num_selected_subbands = np.sum(out2,-1)   ## Number of selected subbands
        #print('new ch_action', out2)
        #print('new num selected subbands', num_selected_subbands)
        Q = torch.tensor(out2.reshape([-1,self.numCell,self.numSubbands]))
        eye = torch.eye(self.numCell)
        ul_rxPow = torch.tensor((10**(self.Pmin/10) / 10**((self.ul_rxPow[0:self.numCell,self.numCell:,time_index])/10)) * active_set) 
        #ul_rxPow = ul_rxPow.transpose()
        #print('rx pow from simple', ul_rxPow)
        power_mat = ul_rxPow.reshape([-1,self.numCell,self.numCell,1])
        Rcv_power = torch.mul(power_mat.squeeze(-1),eye)
        Interference_mat = torch.mul(power_mat.squeeze(-1),1-eye)
        #print('Interference mat ',Interference_mat)
        Rate = torch.zeros(self.numCell)
        int_k = torch.zeros(self.numCell,self.numSubbands)
        sinr_k = torch.zeros(self.numCell,self.numSubbands)
        Rate_ = torch.zeros(self.numCell,self.numSubbands)
        for i in range(self.numSubbands):
            int_k[:,i] = torch.sum(torch.mul(Q[:,:,i].unsqueeze(-1),Interference_mat),dim=1) + self.noisePower
            #print('int on k = ', i, ' is ', int_k[:,i])
            sinr_k[:,i] = torch.sum(torch.mul(Q[:,:,i].unsqueeze(-1),Rcv_power),dim=1)/(int_k[:,i])
            Rate_[:,i] = torch.log2(1+sinr_k[:,i])
            Rate += Rate_[:,i]
        #print('Rate per subband ', Rate_)
        #print('Rate ',Rate)
        #print('Noise ', self.noisePower)
        #print('new sinr pow ', sinr_k)
        self.sum_int_history = np.concatenate((self.sum_int_history, np.array(int_k).reshape(self.numCell, self.numSubbands,1)), axis=2)
        num_selected_subbands__ = np.copy(num_selected_subbands)
        num_selected_subbands__[num_selected_subbands__ == 0] = 1 #To prevent divide by 0
        self.sRate = np.array(Rate)/num_selected_subbands__
        self.sRate = self.sRate * active_set
        #if np.any(num_selected_subbands>1):
        #print('Rate per subband ', Rate_)
        #print('Rate ', Rate)
        #print('sRate ', self.sRate)
        #print('selected subbands', num_selected_subbands)
        # pow_action = np.ones(self.numCell)
        # self.traffic_aware_joint_power_channel_step(pow_action, chl_action, time_index)

        self.mean_lqr_cost, self.lqr, self.plant_states, self.force, self.buffer_size,self.sensor_control_aoi, self.control_sensor_aoi = self.subn_plant_control.run(self.sRate, self.bw_to_PRBs[self.totalbw], self.numSubbands, num_selected_subbands=num_selected_subbands)
        #print(self.subn_plant_control.activity_indicator)
        self.lqr_list.append(self.lqr)
        self.plant_states_list.append(self.plant_states)
        self.force_list.append(self.force)
        self.buffer_size_list.append(self.buffer_size)
        
        #print('int pow ', intPow)
        #print('sRate simple ',self.sRate)
        self.intPow = int_k
        #print('int k from simple', int_k)
        #print('Rate from simple', Rate)
        return 0
    
    def dl_traffic_aware_joint_power_channel_sinr_calc(self, dl_active_set):
        #active_set = self.activity_indicator[time_index]
        active_set = dl_active_set
        time_index = self.time_index
        active_set = self.activity_indicator[time_index]
        pow_action = self.pow_action
        chl_action = self.chl_action
        #print('ch ', chl_action)
        #chl_action = list(map(self.multichannel_action.get,chl_action))
        #print(chl_action.shape)
        self.numSubbands = 3
        valid_plants = list(np.nonzero(active_set)[0])
        not_valid_plants = list(np.nonzero(np.invert(active_set.astype('bool')))[0])
        out2 = np.eye(self.numSubbands)[chl_action.astype('int')] 
        #print(out2.shape)
        #print('out2 any shape',out2.any(axis=1).astype('int').shape)
        out2 = out2.any(axis=1).astype('int') * np.expand_dims(active_set,-1)
        num_selected_subbands = np.sum(out2,-1)   ## Number of selected subbands
        num_selected_subbands__ = np.copy(num_selected_subbands)
        num_selected_subbands__[num_selected_subbands__ == 0] = 1 #To prevent divide by 0
        pow_action = pow_action/(num_selected_subbands__*self.num_fadingblocks_per_subband)
        Q = torch.tensor(out2.reshape([-1,self.numCell,self.numSubbands]))
        chgain_mat = torch.tensor((10**(self.dl_rxPow[0:self.numCell,self.numCell:,:,:,time_index]/10)).reshape([-1,self.numCell,self.numCell,self.numSubbands, self.num_fadingblocks_per_subband,1]))
        pow_action[not_valid_plants] = 0
        tx_powers = torch.tensor(pow_action, dtype=chgain_mat.dtype)
        tx_powers = tx_powers.reshape([-1,1,self.numCell,1]).repeat(1,self.num_fadingblocks_per_subband, self.numSubbands,1,1).T.unsqueeze(2)
        eye = torch.eye(self.numCell).repeat(self.num_fadingblocks_per_subband, self.numSubbands,1,1).T
        dl_rxPow = torch.divide(tx_powers,chgain_mat)   ###Generate rx powers 
        num_active = int(np.sum(active_set))
        power_mat = dl_rxPow.reshape([-1,self.numCell,self.numCell,self.numSubbands,self.num_fadingblocks_per_subband,1]) #same as power - active_rxpow
        Rcv_power = torch.sum(torch.mul(power_mat.squeeze(-1),eye)[0,:,:], dim=1).unsqueeze(-1) 
        #print(Rcv_power.shape)
        self.des_chgain = Rcv_power[:,:,:,0]
        Interference_mat = torch.mul(power_mat.squeeze(-1),1-eye)

        Rate = torch.zeros(self.numCell)
        int_i = torch.zeros(self.numCell,self.numSubbands)
        int_i_k = torch.zeros(self.numCell,self.numSubbands,self.num_fadingblocks_per_subband)
        sinr_i_k = torch.zeros(self.numCell,self.numSubbands,self.num_fadingblocks_per_subband)
        sinr_i = torch.zeros(self.numCell,self.numSubbands)
        Rate_i_k = torch.zeros(self.numCell,self.numSubbands,self.num_fadingblocks_per_subband)
        Rate_i = torch.zeros(self.numCell,self.numSubbands)
        for i in range(self.numSubbands):
            for k in range(self.num_fadingblocks_per_subband):
                int_i_k[:,i,k] = torch.sum(torch.mul(Q[:,:,i].unsqueeze(-1),Interference_mat[:,:,:,i,k]),dim=1) + self.noisePower
                int_i[:,i] += int_i_k[:,i,k] * (1/self.num_fadingblocks_per_subband)
            #print('int on k = ', i, ' is ', int_k[:,i])
                sinr_i_k[:,i,k] = torch.mul(Q[:,:,i].unsqueeze(-1),Rcv_power[:,i,k,:])[0,:,0]/int_i_k[:,i,k]
                sinr_i[:,i] += sinr_i_k[:,i,k] * (1/self.num_fadingblocks_per_subband)
                Rate_i_k[:,i,k] = torch.log2(1+sinr_i_k[:,i,k])
                Rate_i[:,i] += Rate_i_k[:,i,k] * (1/self.num_fadingblocks_per_subband)
            Rate += Rate_i[:,i]
        self.dl_sinr_i_k = np.array(sinr_i_k)
        self.dl_avRate = Rate/num_selected_subbands__
        self.dl_avRate = self.dl_avRate.tolist()
        return self.dl_sinr_i_k

    def traffic_aware_joint_power_channel_step(self, pow_action, chl_action, time_index):
        active_set = self.activity_indicator[time_index]
        #dl_sinr_i_k = self.dl_traffic_aware_joint_power_channel_sinr_calc(pow_action, chl_action, time_index)
        self.time_index = time_index 
        self.pow_action = pow_action
        self.chl_action = chl_action
        #print('ch ', chl_action)
        #chl_action = list(map(self.multichannel_action.get,chl_action))
        #print(chl_action.shape)
        self.numSubbands = 3
        valid_plants = list(np.nonzero(active_set)[0])
        not_valid_plants = list(np.nonzero(np.invert(active_set.astype('bool')))[0])
        out2 = np.eye(self.numSubbands)[chl_action.astype('int')] 
        #print(out2.shape)
        #print('out2 any shape',out2.any(axis=1).astype('int').shape)
        out2 = out2.any(axis=1).astype('int') * np.expand_dims(active_set,-1)
        num_selected_subbands = np.sum(out2,-1)   ## Number of selected subbands
        num_selected_subbands__ = np.copy(num_selected_subbands)
        num_selected_subbands__[num_selected_subbands__ == 0] = 1 #To prevent divide by 0
        pow_action = pow_action/(num_selected_subbands__*self.num_fadingblocks_per_subband)
        Q = torch.tensor(out2.reshape([-1,self.numCell,self.numSubbands]))
        chgain_mat = torch.tensor((10**(self.ul_rxPow[0:self.numCell,self.numCell:,:,:,time_index]/10)).reshape([-1,self.numCell,self.numCell,self.numSubbands, self.num_fadingblocks_per_subband,1]))
        pow_action[not_valid_plants] = 0
        tx_powers = torch.tensor(pow_action, dtype=chgain_mat.dtype)
        self._tx_powers = list(np.array(tx_powers)[np.nonzero(np.array(tx_powers))])
        tx_powers = tx_powers.reshape([-1,1,self.numCell,1]).repeat(1,self.num_fadingblocks_per_subband, self.numSubbands,1,1).T.unsqueeze(2)
        eye = torch.eye(self.numCell).repeat(self.num_fadingblocks_per_subband, self.numSubbands,1,1).T
        ul_rxPow = torch.divide(tx_powers,chgain_mat)   ###Generate rx powers 
        num_active = int(np.sum(active_set))
        power_mat = ul_rxPow.reshape([-1,self.numCell,self.numCell,self.numSubbands,self.num_fadingblocks_per_subband,1]) #same as power - active_rxpow
        Rcv_power = torch.sum(torch.mul(power_mat.squeeze(-1),eye)[0,:,:], dim=1).unsqueeze(-1) 
        #print(Rcv_power.shape)
        self.des_chgain = Rcv_power[:,:,:,0]
        Interference_mat = torch.mul(power_mat.squeeze(-1),1-eye)

        Rate = torch.zeros(self.numCell)
        int_i = torch.zeros(self.numCell,self.numSubbands)
        int_i_k = torch.zeros(self.numCell,self.numSubbands,self.num_fadingblocks_per_subband)
        self.sinr_i_k = torch.zeros(self.numCell,self.numSubbands,self.num_fadingblocks_per_subband)
        self.sinr_i = torch.zeros(self.numCell,self.numSubbands)
        Rate_i_k = torch.zeros(self.numCell,self.numSubbands,self.num_fadingblocks_per_subband)
        Rate_i = torch.zeros(self.numCell,self.numSubbands)
        for i in range(self.numSubbands):
            for k in range(self.num_fadingblocks_per_subband):
                int_i_k[:,i,k] = torch.sum(torch.mul(Q[:,:,i].unsqueeze(-1),Interference_mat[:,:,:,i,k]),dim=1) + self.noisePower
                int_i[:,i] += int_i_k[:,i,k] * (1/self.num_fadingblocks_per_subband)
            #print('int on k = ', i, ' is ', int_k[:,i])
                self.sinr_i_k[:,i,k] = torch.mul(Q[:,:,i].unsqueeze(-1),Rcv_power[:,i,k,:])[0,:,0]/int_i_k[:,i,k]
                self.sinr_i[:,i] += self.sinr_i_k[:,i,k] * (1/self.num_fadingblocks_per_subband)
                Rate_i_k[:,i,k] = torch.log2(1+self.sinr_i_k[:,i,k])
                Rate_i[:,i] += Rate_i_k[:,i,k] * (1/self.num_fadingblocks_per_subband)
            Rate += Rate_i[:,i]
        
        self.ul_avRate = Rate/num_selected_subbands__
        #print('Rate per subband ', Rate_)
        #print('Rate ',Rate)
        #print('Noise ', self.noisePower)
        self.ul_avRate = self.ul_avRate.tolist()
        self.sum_int_history = np.concatenate((self.sum_int_history, np.array(int_i).reshape(self.numCell, self.numSubbands,1)), axis=2)
        
        # self.sRate = np.array(Rate)/num_selected_subbands__
        # self.sRate = self.sRate * active_set
        #print('JSP sRate ', self.sRate)
        #print('JSP selected subbands', num_selected_subbands__)
        #self.mean_lqr_cost, self.lqr, self.plant_states, self.force, self.buffer_size,self.sensor_control_aoi, self.control_sensor_aoi = self.subn_plant_control.run(self.sRate, self.bw_to_PRBs[self.totalbw], self.numSubbands, num_selected_subbands=num_selected_subbands)
        #print(self.subn_plant_control.activity_indicator)
        self.mean_lqr_cost, self.lqr, self.plant_states, self.force, self.buffer_size,self.sensor_control_aoi, self.control_sensor_aoi = self.subn_plant_control.run(np.array(self.sinr_i_k), num_selected_subbands=num_selected_subbands, subband_action=chl_action, num_RB_per_Subband=self.num_fadingblocks_per_subband)
        self.lqr_list.append(self.lqr)
        self.plant_states_list.append(self.plant_states)
        self.force_list.append(self.force)
        self.buffer_size_list.append(self.buffer_size)
        
        #print('int pow ', intPow)
        #print('sRate ',self.sRate)
        self.intPow = int_i
        #print('int k from simple', int_k)
        #print('Rate from simple', Rate)
        return 0



    def traffic_aware_multi_channel_step2(self,chl_action,time_index):
        active_set = self.activity_indicator[time_index]
        pow_action = (10**(self.Pmin/10)) * active_set  #convert power to linear, only active subn has transmit power > 0
        num_selected_subbands = np.zeros(self.numCell)
        ul_rxPow = self.ul_rxPow[:,:,time_index]
        multi_channel_decision = {} #Dictionary of subnetworks as key and operating subband or subbands as item
        for i in range(self.numCell):
            multi_channel_decision[i] = self.multichannel_action[chl_action[i]]
            if hasattr(multi_channel_decision[i], "__len__"):
                num_selected_subbands[i] = len(np.unique(multi_channel_decision[i])) 
            else:
                num_selected_subbands[i] = 1
        
        #print('ch_action ', chl_action)
        #print((10**(self.Pmin/10) / 10**((self.ul_rxPow[0:self.numCell,self.numCell:,time_index])/10)))
        #print('old ch_action', chl_action)
        #print('old num selected subbands', num_selected_subbands)
        intPow = np.zeros([self.numCell,self.numSubbands],dtype=np.float64)
        for k in range(self.numSubbands):
            b = []
            indm = np.array([keys for keys in multi_channel_decision if np.any(np.unique(multi_channel_decision[keys]) == k)], dtype=np.int32)
            indm_active = active_set[indm]
            #print('In 2 indm_active = ',indm_active)
            for i in range(len(indm_active)):
                if indm_active[i] == 1:  #select only active subnetwork that are on channel k
                    b.append(indm[i])
            b = np.array(b, dtype=int)
            #print('In 2 b = ', b)
            for n in range(self.numCell):
                intPow[n,k] = np.sum(10**(self.Pmin/10) / 10**((ul_rxPow[n,self.numCell+b[b !=n]])/10)) + self.noisePower
        #print('In 2 ul_rxPow ', 10**(self.Pmin/10) / 10**((ul_rxPow[0:self.numCell,self.numCell:])/10))
        
        # SINR and channel rates
        self.sum_int_history = np.concatenate((self.sum_int_history, intPow.reshape(self.numCell, self.numSubbands,1)), axis=2)
        
        self.SINRAll = np.zeros([self.numCell, self.numSubbands], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        for n in range(self.numCell):
            for k in range(self.numSubbands):
                self.SINRAll[n,k] = (10**(self.Pmin/10) / 10**((ul_rxPow[n,self.numCell+n])/10)) / intPow[n,k]
            self.SINR[n] = np.sum(self.SINRAll[n,np.unique(multi_channel_decision[n])])
            self.sRate[n] = np.sum(np.log2(1 + self.SINRAll[n,np.unique(multi_channel_decision[n])]))/num_selected_subbands[n]
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
            #print('In 2 - SINR', self.SINRAll[n,multi_channel_decision[n]])
        indx = np.array([keys for keys in multi_channel_decision if np.any(np.unique(multi_channel_decision[keys]) == -1)], dtype=np.int32) #select subnetwork that chose no channel, and set their rate to 0
        self.sRate[indx] = 0
        #print('old sinr ', self.SINRAll)

        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        #self.rate_list[time_index] = 1/self.numSubbands * self.sRate 
        #print('sRate old ', self.sRate)
        
        self.mean_lqr_cost, self.lqr, self.plant_states, self.force, self.buffer_size,self.sensor_control_aoi, self.control_sensor_aoi = self.subn_plant_control.run(self.sRate, self.bw_to_PRBs[self.totalbw], self.numSubbands, num_selected_subbands)
        #print(self.subn_plant_control.activity_indicator)
        self.lqr_list.append(self.lqr)
        self.plant_states_list.append(self.plant_states)
        self.force_list.append(self.force)
        self.buffer_size_list.append(self.buffer_size)
        
        #print('int pow ', intPow)
        #print('sRate ',self.sRate)
        self.intPow = intPow
        return intPow
        


    def channel_step_devices(self, chl_action, time_index):
        #pow_action = self.powerLevels * np.ones(self.numCell)
        pow_action = self.Pmin * np.ones(self.numCell)
        ul_rxPow = self.ul_rxPow[:,:,time_index]

        # Interference power        
        intPow = np.zeros([self.numCell,self.numDev,self.numSubbands],dtype=np.float64)
        for k in range(self.numSubbands):
            indm = np.argwhere(chl_action == k)
            _m = 0
            for n in range(self.numCell):
                for m in range(self.numDev):
                    intPow[n,m,k] = np.sum(10**((pow_action[n]-ul_rxPow[_m,self.numCell*self.numDev+indm[indm !=n]])/10)) + self.noisePower
                    _m += 1

        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell, self.numDev, self.numSubbands], dtype=np.float64)
        self.SINR = np.zeros([self.numCell, self.numDev],dtype=np.float64)
        self.sRate = np.zeros([self.numCell, self.numDev],dtype=np.float64)
        _m = 0
        for n in range(self.numCell):
            for m in range(self.numDev):
                for k in range(self.numSubbands):
                    self.SINRAll[n,m,k] = (10**((pow_action[n] - ul_rxPow[_m,self.numCell*self.numDev+n])/10)) / intPow[n,m,k]
                self.SINR[n,m] = self.SINRAll[n,m,chl_action[n]]
                self.sRate[n,m] = np.log2(1 + self.SINR[n,m])
                _m += 1
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        #print(self.sRate.shape)

        self.SINRAll = self.SINRAll[:,0,:]
        self.SINR = self.SINR[:,0]
        self.sRate = self.sRate[:,0]
        intPow = intPow[:,0,:]
        """# SINR and channel rates
        intPow = intPow[:,0,:]
        self.SINRAll = np.zeros([self.numCell, self.numSubbands], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        for n in range(self.numCell):
            for k in range(self.numSubbands):
                self.SINRAll[n,k] = (10**((pow_action[n] - ul_rxPow[n,self.numCell+n])/10)) / intPow[n,k]
            self.SINR[n] = self.SINRAll[n,chl_action[n]]
            self.sRate[n] = np.log2(1 + self.SINR[n])
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)"""
        
        return intPow

    def reform_lqr(self): #Because lqr was not returned for invalid plants 
        if len(self.lqr) == self.numCell:
            lqr = self.lqr
        else:
            lqr = np.zeros(self.numCell)
            #buffer_size = np.zeros(self.numCell)
            list_plants = np.arange(0,self.numCell,1)
            #print(list_plants.shape)
            valid_plants = np.array(self.subn_plant_control.valid_plants)
            #print(valid_plants.shape)
            invalid = list_plants[np.isin(list_plants, valid_plants, invert=True)]
            lqr[valid_plants] = self.lqr
            #buffer_size[valid_plants] = self.buffer_size
            #buffer_size[invalid] = 0
            lqr[invalid] = 10000
            #self.buffer_size = buffer_size
        return np.array(lqr)
    
    def joint_step(self, actions, time_index):

        chl_action = self.comb_act[actions][:,0].astype(int)
        pow_action = self.comb_act[actions][:,1]

        ul_rxPow = self.ul_rxPow[:,:,time_index]

        # Interference power        
        intPow = np.zeros([self.numCell,self.numSubbands],dtype=np.float64)
        for k in range(self.numSubbands):
            indm = np.argwhere(chl_action == k)
            for n in range(self.numCell):
                intPow[n,k] = np.sum(10**((pow_action[n]-ul_rxPow[n,self.numCell+indm[indm !=n]])/10)) + self.noisePower

        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell, self.numSubbands*self.numLevels], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        
        for n in range(self.numCell):
            i = 0
            for k in range(self.numSubbands):
                for u in range(self.numLevels):
                    self.SINRAll[n,i] = (10**((self.Plevels[u] - ul_rxPow[n,self.numCell+n])/10)) / intPow[n,k]
                    i += 1
            self.SINR[n] = self.SINRAll[n,actions[n]]
            self.sRate[n] = np.log2(1 + self.SINR[n])
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)

        return intPow

    # =================== Environment action =================== #
    #def step(self, time_index=0, chl_action=None, pow_action=None):
    def step(self, actions, time_index=0):
        """
        Perform a step in the environment, for a given time step and action

        Param :time_index: Integer time index (np.array)
        Param :chl_action: Integer channel actions (np.array)
        Param :pow_action: Float power levels (np.array)
        Return :obs: Observations, type defined with observation_type (np.array)
        Return :reward: Rewards, type defined with reward_type (np.array)
        Return :done: True if episode is done (list)
        Return :info: Additional information (dictionary)
        """
        #print(self.problem)
        # Decode input action
        if self.problem == 'channel':
            if self.numDev <= 1: # Remove this IF statement after development!
                #intPow = self.channel_step(actions, time_index)
                #intPow = self.traffic_aware_channel_step(actions, time_index)
                intPow = self.centralized_channel_step(actions,time_index)
            else:
                intPow = self.channel_step_devices(actions, time_index)
        elif self.problem == 'power':
            intPow = self.centralized_power_step(actions,time_index)

        elif self.problem == 'multichannel':
            intPow = self.traffic_aware_multi_channel_step(actions,time_index)
        elif self.problem == 'joint':
            intPow = self.joint_step(actions, time_index)
        else:
            raise NotImplementedError

        # Observation
        if self.observation_type == 'I':
            obs = 10.0*np.log10(intPow)
        
        elif self.observation_type == 'potential_indInt':
            obs = (self.potentialIntMat - self.indPowMin)/self.indPowNorm
        
        elif self.observation_type == 'actual_indInt':
            obs = self.actualIntMat
        
        elif self.observation_type == 'sumInt_dist':
            obs = np.concatenate(((10.0*np.log10(intPow) - 10.0*np.log10(self.noisePower)) / self.intPowNorm, self.IntDistMat/self.factoryArea[0]), axis =-1)
        
        elif self.observation_type == 'I_dist':
            obs = self.IntDistMat/self.factoryArea[0] 

        elif self.observation_type == 'sumInt_channel':
            obs = np.concatenate(((10.0*np.log10(intPow) - 10.0*np.log10(self.noisePower)) / self.intPowNorm, self.ch_choice), axis =-1)
        
        elif self.observation_type == 'I_minmax':
            obs = (10.0*np.log10(intPow) - 10.0*np.log10(self.noisePower)) / self.intPowNorm
        
        elif self.observation_type == 'I_minmax_lqr':
            lqr_norm =np.expand_dims(self.reform_lqr(),-1)/200
            obs = np.concatenate((lqr_norm,(10.0*np.log10(intPow) - 10.0*np.log10(self.noisePower)) / self.intPowNorm), axis = -1)

        elif self.observation_type == 'I_minmax_lqr_buffer':
            lqr_norm =np.expand_dims(self.reform_lqr(),-1)/200
            buffer_size = np.expand_dims(self.buffer_size,-1)/np.max(np.array(self.subn_plant_control.data_size))
            obs = np.concatenate((buffer_size, lqr_norm,(10.0*np.log10(intPow) - 10.0*np.log10(self.noisePower)) / self.intPowNorm), axis = -1)
        
        elif self.observation_type == 'I_minmax_lqr_buffer_aoi':
            lqr_norm =np.expand_dims(self.reform_lqr(),-1)/200
            buffer_size = np.expand_dims(self.buffer_size,-1)/np.max(np.array(self.subn_plant_control.data_size))
            aoi = np.expand_dims(self.subn_plant_control.sensor_control_aoi, -1)/10
            obs = np.concatenate((aoi, buffer_size, lqr_norm,(10.0*np.log10(intPow) - 10.0*np.log10(self.noisePower)) / self.intPowNorm), axis = -1)
        
        elif self.observation_type == 'sinr':
            obs = self.SINRAll
        elif self.observation_type == 'sinr_minmax':
            obs = (self.SINRAll - self.sinrMin) / self.sinrNorm
        else:
            raise NotImplementedError
        
        #print(self.SINR)
        
        # Reward signal
        if self.reward_type == 'rate':
            reward = self.sRate
        elif self.reward_type == 'lqr':
            reward = np.negative(self.reform_lqr())
        elif self.reward_type == 'sinr':
            reward = self.SINR
        elif self.reward_type == 'binary':
            alpha = 10
            idx = self.SINR <= self.SINR_req[:,0]
            reward = np.full(self.numCell, alpha, dtype=float)
            reward[idx] = (-1) * alpha
        elif self.reward_type == 'composite_reward':
            rate_sum = np.sum(self.sRate, axis=1)
            idx = rate_sum <= self.r_min
            reward = self.lambda_1 * rate_sum
            reward[idx] -= self.lambda_2 * (rate_sum[idx] - self.r_min)
        #elif self.reward_type == 'constrained':
        #    self.w_next = np.max([self.w_next + self.reqSINR - self.SINR, np.zeros(self.w_next.shape)], axis=0)
        #    self.p_next = np.max([self.p_next + self.numDev * 10**((pow_action)/10) - self.Pmax, np.zeros(self.p_next.shape)], axis=0)
        #    reward = np.sum(self.pRate, axis=1) - np.sum(self.w_next, axis=1) - self.p_next
        else:
            raise NotImplementedError

        # Supplementary outputs
        done = np.array([time_index >= (self.numSteps-1) for _ in range(self.numCell)]) # (Return True if episode is done)
        info = {} # No info
        #print(done)
        return obs, reward, done, info, self.sRate, self.lqr

# =================== Used for analysis of the environment =================== #
def centralizedColoring(interMat, numGroups): 
    """
    Compute action with centralised graph colouring

    Param :interMat: Receive power matrix (np.array)
    Return :act: CGC actions (np.array)
    """
    N = interMat.shape[0]
    G = nx.Graph()
    G.add_nodes_from(np.linspace(0,N-1,num=N))   
    for n in range(N):
        dn = interMat[n,:]
        Indx = sorted(range(N), key=lambda k: dn[k])
        for k in range(1, numGroups):
            G.add_edge(n, Indx[k]) 
    d = nx.coloring.greedy_color(G, strategy='connected_sequential_bfs', interchange=True)
    act = np.asarray(list(d.values()),dtype=int)
    idx = np.argwhere(act >= numGroups).flatten()
    act[idx] = np.random.choice(np.arange(numGroups), len(idx))
    return act

if __name__ == '__main__':

    #from util import *

    N=4
    M=1
    channels=2
    seed=123
    reward = 'rate'
    steps = 400
    num_of_episodes = 1000
    plant_states = np.zeros((num_of_episodes,steps,N,4))
    lqr_cost = np.zeros((num_of_episodes,steps,N))
    se = np.zeros((num_of_episodes,steps,N))
    activity_indicator = np.zeros((num_of_episodes, steps, N))

    env = env_subnetwork(numCell=N, numDev=M, numSubbands=channels, level=1, fname='Factory_values.mat', dt=0.005, steps=steps, reward_type=reward, clutter='sparse', observation_type='I', seed=seed)
    env.reset()
    env.generate_mobility()
    env.generate_channel()
    act = np.random.choice(channels, N)
    
    for episode in range(num_of_episodes):
        env = env_subnetwork(numCell=N, numDev=M, numSubbands=channels, level=1, fname='Factory_values.mat', dt=0.005, steps=steps, reward_type=reward, clutter='sparse', observation_type='I', seed=seed)
        env.generate_mobility()
        env.generate_channel()
        for time_step in range(steps):
            act = np.random.choice(channels, N)
            obs, rew, _, _ = env.step(actions=act, time_index=time_step)
        plant_states[episode] = env.plant_states_list
        lqr_cost[episode] = env.lqr_list
        se[episode] = env.rate_list
        activity_indicator[episode] = env.activity_indicator

    np.savez('./xInterfering_6subband_3MHz.npz',se = se, lqr =lqr_cost, states=plant_states, act=activity_indicator)

    

    # CHECK DEVICE AND ACCESS POINT DEPLOYMENT
    #N, M, seed = 10, 10, 123
    #check_devLoc(N=N, M=M, seed=seed, timestep=0)
    #check_devLoc(N=N, M=M, seed=seed, timestep=np.random.randint(200))
    #check_devLoc(N=N, M=M, seed=seed, timestep=-1)

    """    
    #num_steps_tot = 10000
    #plot_environment_metrics(['sparse'], [20], N_plot_CDF=[True], num_steps_tot=num_steps_tot, max_delay=5, save_fig=False, fpath='../../../../results/', dt=0.01,steps=10)
    #plot_environment_metrics(['sparse'], [20], N_plot_CDF=[True], num_steps_tot=num_steps_tot, max_delay=10, save_fig=False, fpath='../../../../results/', dt=0.005,steps=200)
    #plot_environment_metrics(['sparse'], [20], N_plot_CDF=[True], num_steps_tot=num_steps_tot, max_delay=16, save_fig=False, fpath='../../../../results/', dt=0.008,steps=124)
    if False:
        max_delays = [1, 2, 10]#[n+1 for n in range(50)][::2] # 1, 3, 5, ..., 49 
        clutter_list = ['sparse', 'dense']
        N_list = [10, 20, 25, 50]
        N_plot_CDF = [True, True, False, True]#[True for _ in N_list]
        num_steps_tot = 1000000
        save_figs = True

        max_delay = plot_test_delay(max_delays, N=20, num_steps_tot=num_steps_tot, save_fig=save_figs)
        print(f'\nMax delay = {max_delay}')
        plot_environment_metrics(clutter_list, N_list, N_plot_CDF=N_plot_CDF, num_steps_tot=num_steps_tot, max_delay=10, save_fig=save_figs, steps=200, dt=0.005)
    """