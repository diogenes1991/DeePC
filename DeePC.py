class DeePC:

    def __init__(self,u_data,y_data):
        
        if len(u_data[0]) != len(y_data[0]):
            print("Error: Incompatible reference and data!")
            print("The input data has",len(u_data[0]),"components")
            print("The reaction data has",len(y_data[0]),"components")
            return None
        self.T   = len(u_data[0])
        self.n_u = len(u_data)
        self.n_y = len(y_data)
        self.u_data = u_data
        self.y_data = y_data
        self.bound = False

    def SetBoundaries(self,LOW,HIG):
        if (len(LOW)==self.n_u and len(HIG)==self.n_u):
            self.lbu = LOW
            self.ubu = HIG
            self.bound = True

    def CreateHankelMatrices(self,L):
        
        ##
        ##   T > (n_u + 1)(T_ini + T_f + n(B)) - 1 
        ## Hankel matrices are (n_data*L) x (T-L+1) Matrices
        ##

        Hankel_U = []
        for shift_i in range(L):
            for input_u in range(self.n_u):
                row = []
                for data_i in range(self.T-L+1):
                    row.append(self.u_data[input_u][data_i+shift_i])
                Hankel_U.append(row)
        self.Hankel_u = np.array(Hankel_U)
        
        Hankel_Y = []
        for shift_i in range(L):
            for input_y in range(self.n_y):
                row = []
                for data_i in range(self.T-L+1):
                    row.append(self.y_data[input_y][data_i+shift_i])
                Hankel_Y.append(row)
        self.Hankel_y = np.array(Hankel_Y)
    
    def RunSolver(self,Time_Horizon,u_ini,y_ini,reference,u_N,y_N):
        
        ##
        ##  u_ini and y_ini needs to be in the format 
        ##  u_ini = [[u0_0,u1_0,...,um_0],...,[u0_Tini,u1_Tini,...,um_Tini]]
        ##  y_ini = [[y0_0,y1_0,...,ym_0],...,[y0_Tini,y1_Tini,...,ym_Tini]]
        ##

        ##
        ##  Here we will setup the CASADI objects for the 
        ##  recursive solver
        ##

        if(len(reference[0])<Time_Horizon):
            print("Maximal recursion depth reached, exiting the Solver")
            return True

        ##
        ##   Check that the initial state is compatible
        ##

        if (len(u_ini)!=len(y_ini)):
            print("Incompatible initial data!")
            return False

        ##
        ##   Next we attempt to create the Hankel Matrices with 
        ##   dimension Tini + Time_Horizon and check 
        ##   if they have maximal row rank
        ##

        self.CreateHankelMatrices(len(u_ini)+Time_Horizon)
        if (np.linalg.matrix_rank(self.Hankel_u) != self.Hankel_u.shape[0]):
            print("Error: The data is not rich enough to look so much into the future")
            print(self.Hankel_u)
            print("Has dimension",self.Hankel_u.shape[0],"but rank",np.linalg.matrix_rank(self.Hankel_u))
            return False

        ##
        ##   Cost and Penalty matrices, indentities for now
        ##

        Q = 10000000*np.identity(self.n_y)
        R = 0.00000001*np.identity(self.n_u)

        T_ini = len(u_ini)
        
        opt_x_dpc = struct_symMX([
            entry('g',   shape=(self.T-T_ini-Time_Horizon+1)),
            entry('u_N', shape=(self.n_u), repeat=Time_Horizon),
            entry('y_N', shape=(self.n_y), repeat=Time_Horizon)
        ])
        

        opt_x_num_dpc = opt_x_dpc(0)
        
        Chi_Sq = 0
        
        for k in range(Time_Horizon):
            for input_u_i in range(self.n_u):
                for input_u_j in range(self.n_u):
                    Chi_Sq += opt_x_dpc['u_N',k,input_u_i]*R[input_u_i][input_u_j]*opt_x_dpc['u_N',k,input_u_j]
            
            #Chi_Sq += 0.01*sum1(opt_x_dpc['u_N',k])**2
            
            for input_y_i in range(self.n_y):
                for input_y_j in range(self.n_y):
                    Chi_Sq += (opt_x_dpc['y_N',k,input_y_i]-reference[input_y_i][k])*Q[input_y_i][input_y_j]*(opt_x_dpc['y_N',k,input_y_j]-reference[input_y_j][k])
        
            #Chi_Sq += sum1(opt_x_dpc['y_N',k])**2
        
        u_past   = self.Hankel_u[:T_ini*self.n_u]
        u_future = self.Hankel_u[T_ini*self.n_u:]
        y_past   = self.Hankel_y[:T_ini*self.n_y]
        y_future = self.Hankel_y[T_ini*self.n_y:]

        SplitHankel = vertcat(u_past,y_past,u_future,y_future)
        RHS         = vertcat(*u_ini,*y_ini,*opt_x_dpc['u_N'],*opt_x_dpc['y_N'])
        Constraint  = SplitHankel@opt_x_dpc['g']-RHS
        
        ##
        ##   Create lower and upper bound structures and set all values to plus/minus infinity.
        ##

        lbx_dpc = opt_x_dpc(-np.inf)
        ubx_dpc = opt_x_dpc(np.inf)

        if self.bound:
            lbx_dpc['u_N'] = DM(self.lbu)
            ubx_dpc['u_N'] = DM(self.ubu)
        
        ##
        ##   Definition of the Solver
        ##

        nlp = {'x' : opt_x_dpc, 'f' : Chi_Sq, 'g' : Constraint}
        S_dpc = nlpsol('S', 'ipopt', nlp)
        
        r = S_dpc(lbg=0, ubg=0, lbx=lbx_dpc, ubx=ubx_dpc)
        opt_x_num_dpc.master = r['x'] 
        
        ##
        ##   Setup of the rerusive problem
        ##

        u_N_star = horzcat(*opt_x_num_dpc['u_N']).full().T
        y_N_star = horzcat(*opt_x_num_dpc['y_N']).full().T
        
        for data in u_N_star:
            u_N.append(data.tolist())

        for data in y_N_star:
            y_N.append(data.tolist())
