clear all
close all
clc

% -------------------------
% Linear system parameters
% -------------------------
% stiffness
k=1;
% coupling stiffness
kc=0.1;
% damping
D=0.02;
% mass
m=1;

% Number of oscillators
dim=1;

% Building the matrices
M=m.*eye(dim);
if dim==1
    K=1;
else 
    K=spdiags(ones(dim,1)*[-kc k+kc -kc], [-1 0 1], sparse(dim,dim));
end

if dim>2
    K=K+diag([0 kc.*ones(1,dim-2) 0]);
end

C=D.*eye(dim);
% state space matrix
A = [zeros(dim) eye(dim);...
    -M\K    -M\C];

% -------------------------
% External harmonic forcing
% -------------------------

% amplitude of harmonic forcing
fa=0.1;
% frequency and period of harmonic forcing 
Om=1.2;%1.9;
T=2*pi/Om;
% direction of the harmonic forcing
f=fa.*ones(dim,1);%[1;zeros(dim-1,1)];% 
% forcing in state space form
G=@(t)[zeros(dim,1); M\f].*sin(Om*t);

% -------------------------
% Noise terms
% -------------------------
% noise intensity
sigma=0.01;
% direction of the noise
fb=sigma.*[zeros(dim,1); M\ones(dim,1)]; %eye(dim)
% number of independedn Gaussian white noise terms
N_gwn=size(fb,2);


% -------------------------
% Nonlinearity
% -------------------------
kappa=0.5*ones(dim,1); %0.5
S = @(x)kappa.*x.^3;  
% Nonlinearity in state space
NL=@(z)[zeros(dim,1);-M\S(z(1:dim))];

% right hand side of deterministic dynamical system dx= f(x,t)dt
RHS=@(t,z) A*z + NL(z) + G(t);

% options for ode-solver
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

% initial condition
x0= [repmat(-1.2235 ,dim,1); repmat(4.2724,dim,1)];

% transients
[~, z_trans] = ode45(@(t,z)RHS(t,z), [0 500*T],x0,opts); % Transients

% periodic orbit with high amplitude
z0=z_trans(end,:)';
[~, z_PO] = ode45(@(t,z)RHS(t,z), [0 T], z0);

% transients
[~, z_trans] = ode45(@(t,z)RHS(t,z), [0 500*T],zeros(2*dim,1),opts); % Transients

% periodic orbit with low amplitude
z0=z_trans(end,:)';
[~, z_PO_0] = ode45(@(t,z)RHS(t,z), [0 T], z0);

% Plot periodic orbits
figure
plot(z_PO(:,1),z_PO(:,dim+1))
hold on
plot(z_PO(1,1),z_PO(1,dim+1),'dg')
plot(z_PO_0(:,1),z_PO_0(:,dim+1))


 
%%
% -------------------------
% Start parallel pool
% -------------------------
if isempty(gcp('nocreate'))==1
    parpool
end
c = gcp; 
nw = c.NumWorkers;
%parpool(c)
%%
% -------------------------
% Euler-Maruyama approximation of SDE
% -------------------------

% Set number of samples
N_smpl=10^3;
% Noise directions
B=@(t,x) fb ;
% set initial condition
% we start forward Euler approximation from high energy orbit
z0=z_PO(1,:).'; 

% final time as and integer multiple of periodis
Nperiod=1;


% number of time steps per period
N_smpl_per_T=[1 5 10 100 1000 10000 100000 200000];

EM_means=zeros(2*dim,length(N_smpl_per_T));
EM_vars=zeros(2*dim,2*dim,length(N_smpl_per_T));




for iter_time_step=1:length(N_smpl_per_T)

% time step
dt=T/N_smpl_per_T(iter_time_step);
% Initialize final smaple locations
X1end=zeros(2*dim,N_smpl);

% Start measuring run time
tic
% Integrate trajectories to obtain samples at t=Nperiod*T
parfor ii=1:N_smpl
    
    [X ,~]=EM_end(RHS,B,Nperiod*N_smpl_per_T(iter_time_step), z0,dt,N_gwn);
    X1end(:,ii)=X.';
   % Generarte some output
    if ii==floor(ii/100)*100
        ii/100
    end
end
% Euler Maruyama run time
EM_clock=toc

EM_means(:,iter_time_step)=mean(X1end,2);
for jj=1:2*dim
    for ii=jj+1:2*dim
        Cov_EM([ii jj],[ii jj])=cov(X1end(ii,:),X1end(jj,:));
    end
end
EM_vars(:,:,iter_time_step)=Cov_EM;
end



%%
% -------------------------
% Small noise integrator
% -------------------------



% Set time step of SNI-approximation 
N_taus=[1 2 3 4 5 10]* Nperiod;

SNI_means=zeros(2*dim,length(N_taus));
SNI_vars=zeros(2*dim,2*dim,length(N_taus));

for iter_time_step=1:length(N_taus)

% Initialize final sample distribution
X2=zeros(2*dim,N_taus(iter_time_step),N_smpl);
% Time step of SNI-approximation Tend=tau=Nperiod*T/N_tau;
Tend=Nperiod*T/N_taus(iter_time_step);
% Initial conditions
X0=z_PO(1,:).'; 
% tolerance
tol=10^-15;
tic
N_tau=N_taus(iter_time_step);
parfor ii=1:N_smpl
    
    x0=X0;
    for np=1:N_tau
        % Compute x(np*Tend;x0,(np-1)*Tend) and the linearized flow map
        % DF_((np-1)*Tend)^(np*Tend)(x0)
        % instead of computing the full matrix DF, we compute V(t) cf. Eq. (20) 
        [t, z] = ode45(@(t,z)duff_stochint(t,z,dim,N_gwn,M,C,K,kappa,f,Om), (np-1)*Tend+[0 Tend], [x0 ;  fb(:)] ,opts); %;zeros(4*dim^2,1)
        
        % Initialize DF*B*(DF*B)^T
        Kt=zeros(2*dim,2*dim,length(t));
        % Retrive matrix V from ode45 integration size(V(t))=2*dim x N_gwn
        V=reshape(z(:,2*dim+1:end),length(t),2*dim,N_gwn);
        for tt=1:length(t)
        tmp=reshape(V(tt,:,:),2*dim,N_gwn);
        % Compute DF*B*(DF*B)^T at the time t=tt;
        Kt(:,:,tt)=tmp*tmp.';
        end
        % Integrate Kt to obtain variances, i.e. Sigma in Eq. (11)
        Vars=trapz(t,Kt,3);
        % Decompose eigenvectors of Variance matrix for inversion
        [V,L]=eig(Vars);
        % Eigevalues greater than zero
        idxs=diag(L)>tol;
        sigs=zeros(2*dim,1);
        % Sample L-dimensional normal distribution, where L is the number
        % of non-zero eigenvalues of the varaince matrix
        tmp_smpl=randn(sum(idxs),1);
        % Rescale samples from normal distribution to obtain samples with
        % variance Vars
        sigs(idxs)=tmp_smpl.*sqrt(diag(L(idxs,idxs)));
        % Transform samples into non-diagonalized form
        x0=z(end,1:2*dim).'+(V*sigs);
        % Instead of decomposing Vars by hand one can also use the build-in
        % command:          
        % x0=mvnrnd(z(end,1:2*dim),Vars).';
        % However, to use mvnrnd Vars needs to be positive definite
        
        % Write sample in X0
        X2(:,np,ii)=x0;
    end
    % Generarte some output
    if ii==floor(ii/100)*100
        ii/100
    end
end
% run-time for SNI-approximation
PS_clock=toc

SNI_means(:,iter_time_step)=mean(squeeze(X2(:,end,:)),2);
for jj=1:2*dim
    for ii=jj+1:2*dim
        Cov_SNI([ii jj],[ii jj])=cov(squeeze(X2(ii,end,:)),squeeze(X2(jj,end,:)));
    end
end
SNI_vars(:,:,iter_time_step)=Cov_SNI;
end 



%%

for iter_time_step=1:length(N_smpl_per_T)
    var_err=abs(EM_vars(:,:,iter_time_step)-EM_vars(:,:,end));
    EM_err_vars(iter_time_step)=  (max(var_err(:)));
    
end
for iter_time_step=1:length(N_taus)
    var_err=abs(SNI_vars(:,:,iter_time_step)-EM_vars(:,:,end));
    SNI_err_vars(iter_time_step)=  (max(var_err(:)));
end
figure
loglog(N_smpl_per_T,vecnorm(EM_means-EM_means(:,end)),'x')
hold on
loglog(N_smpl_per_T,EM_err_vars,'s')
grid on
xlabel('time steps per period T')
ylabel('convergence error')

figure
semilogy(N_taus,vecnorm(SNI_means-EM_means(:,end)),'x')
hold on
semilogy(N_taus,SNI_err_vars,'s')
grid on
legend('error mean','error variance')
ylabel('convergence error')


xlabel('time steps per period T')