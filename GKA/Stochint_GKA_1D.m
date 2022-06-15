clear all
close all

% -------------------------
% This script carries out the Gaussian kernel approximation (GKA) described in
% section 5. After defining the parameters and generating smaples we
% proceed with the GKA after line 170.
% -------------------------


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
fb=sigma.*[zeros(dim); M\ones(dim)]; %eye(dim)
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

 
 %%
 
% -------------------------
% Small noise integrator to generat3 10^6 samples after one period
% For comments please see SNI_eval_osci_chain.m
% -------------------------
Nperiod=1;
N_smpl=10^6;
tol=10^-15;
N_tau=5*Nperiod;

Tend=Nperiod*T/N_tau;
X0=z_PO(1,:).';

tic
X0s=repmat(X0,1,N_smpl);
Vars=zeros(2*dim,2*dim,N_smpl);
X0s_nostep=repmat(X0,1,N_smpl);
parfor ii=1:N_smpl
    X0s(:,ii)=X0;
    for np=1:N_tau
    
    [t, z] = ode45(@(t,z)duff_stochint(t,z,dim,N_gwn,M,C,K,kappa,f,Om), (np-1)*Tend+[0 Tend], [X0s(:,ii) ;  fb(:)] ,opts); %;zeros(4*dim^2,1)
    Kt=zeros(2*dim,2*dim,length(t));
    V=reshape(z(:,2*dim+1:end),length(t),2*dim,N_gwn);
    for tt=1:length(t)
        tmp=reshape(V(tt,:,:),2*dim,N_gwn);
        Kt(:,:,tt)=tmp*tmp.';
    end
    Vars(:,:,ii)=trapz(t,Kt,3);
   [V,L]=eig(squeeze(Vars(:,:,ii)));
    idxs=diag(L)>tol;
    sigs=zeros(2*dim,1);
    tmp_smpl=randn(sum(idxs),1);
    Red_var_inv=eye(sum(idxs))/L(idxs,idxs);
    
    sigs(idxs)=tmp_smpl.*sqrt(diag(L(idxs,idxs)));
    X0s_nostep(:,ii)=z(end,1:2*dim).';
    X0s(:,ii)= z(end,1:2*dim).'+(V*sigs);
    end
    if ii==floor(ii/1000)*1000
        ii/1000
    end
 
end
%size(X0s)

 toc

%%

% -------------------------
% GAUSSIAN KERNEL APPROXIMATION (GKA)
% -------------------------

% Computing sample mean
X_mean=squeeze(mean(X0s(:,:),2));

% Setting upper and lower limits for grid
X_min=X_mean-3*std(X0s(:,:),0,2);
X_max=X_mean+3*std(X0s(:,:),0,2);
grid_dist=(X_max-X_min)./2/100;

% Generatin grid for displacement and velocity of the first mass
Y1=linspace(X_min(1),X_max(1),100);
Y2=linspace(X_min(1+dim),X_max(1+dim),100);
[Y1,Y2]=meshgrid(Y1,Y2);
Y1_vec=Y1(:).';
Y2_vec=Y2(:).';


% Initializing the smaple sizes
N_smpls=[100 10^3 10^4 10^5 10^6];
% Initializing pdf
prob_plot=zeros(length(N_smpls),100,100);


for ll=1:length(N_smpls)
    % Current smaple size
    N_smpl=N_smpls(ll);
    % Distributing the workload across the workers
    % Each worker will compute a pdf from a subset of the samples. The
    % individual pdfs from each worker are then summed to generate the
    % final pdf. 
    N1=floor(N_smpl/nw);
    N2=mod(N_smpl,nw);
    ids1=[1:N2+1 repmat(N2+1,1,nw-N2-1)];
    ids2=(0:nw-1).*N1;
    % Worker with number n will compute pdf for samples w_idx(n,1) to w_idx(n,2)
    w_idx=zeros(2,nw);
    w_idx(1,:)=ids1+ids2;
    w_idx(2,1:end-1)=w_idx(1,2:end)-1;
    w_idx(2,end)=N_smpl;
    % Initializing the individual pdfs.
    prob_df=zeros(nw,length(Y1_vec));

tol2=10^-6;

for ii=1:nw
    for jj=w_idx(1,ii):w_idx(2,ii)
        % Variance for samle jj at final time
        Vars_tmp=squeeze(Vars(:,:,jj));
        % Diagonalize variance
        [V,L]=eig(Vars_tmp);
        % Indices with non zero variance
        idxs= diag(L)>tol;
        %L =L+  diag(idxs).*min(L(diag(diag(L)>tol))); %
        lengY=length(Y1_vec);
        % The center point of each grid point
        center_point=[Y1_vec;repmat(X_mean(2:dim),1,lengY);Y2_vec;repmat(X_mean(dim+2:2*dim),1,lengY)];
        % Transform into coordinates such that variance is diagonal 
        grid_points_proj=V(:,idxs).'*center_point;
        % Compute pseudo inverse of the diagonalized variances
        Red_var_inv=eye(sum(idxs))/L(idxs,idxs);
        % The center of the jj-th Gaussian is X0s_nostep(:,jj). This is now transformed into diagonal coordinates.  
        proj_mean=V(:,idxs).'*X0s_nostep(:,jj);
        % Distance between the center of the jj-th Gaussian and each grid
        % point in transfomred coordinates.
        proj_grid_minus_mean=grid_points_proj-repmat(proj_mean,1,lengY);
        % Compute Gaussian distribution in diagonalized coordinates
        efcn_exp=-0.5.*sum(Red_var_inv*(proj_grid_minus_mean.^2));%
        % Value of the Gaussian pdf at each grid point
        tmp_pdf =(1/sqrt((2*pi)^(2*dim)*det(L(idxs,idxs)))).*exp(efcn_exp);
        tmp_pdf = tmp_pdf./(tol2^(2*dim-sum(idxs)));
        tmp_pdf(vecnorm(V(:,not(idxs)).'*center_point,'Inf')>tol2)=0;
        
        % Adding pdf 
        prob_df(ii,:)= prob_df(ii,:)+tmp_pdf; %
%         if jj==floor(jj/1000)*1000
%             jj/1000
%         end
         
    end
   
end
prob_plot_tmp=sum(prob_df)/N_smpl;

prob_plot(ll,:,:)=reshape(prob_plot_tmp,100,100);
ll
end

%%

% Generating pdf by counting number of smaples in the discretized volume.
Ns=zeros(size(Y1));
for ii=1:length(Y1(:,1))
    for jj=1:length(Y2(1,:))
       dist=X0s -repmat([Y1(ii,jj); X_mean(2:dim);Y2(ii,jj); X_mean(dim+2:2*dim)],1,N_smpl);
       Ns(ii,jj)=sum(sum(abs(dist)<grid_dist)==2*dim)/(2^(2*dim)*prod(grid_dist)*N_smpl);
        
    end
end

pdf_MC_interp= griddedInterpolant(Y1',Y2',Ns');
p_mean_MC=pdf_MC_interp(X_mean(1:dim:2*dim).');
 
 

% Plotting both distributions
figure
surf(Y1,Y2,squeeze(prob_plot(end,:,:)))
hold on


plot3(Y1,Y2,Ns,'or')
%plot3(Y1,Y2,Ns,'dg')
xlabel('Position x_1')
ylabel('Velocity v_1')
zlabel('pdf')

legend(' GKA','Monte Carlo')
% Relative error
rel_err =sum(sum(abs(Ns -squeeze(prob_plot(end,:,:))).^2))/sum(sum(abs(Ns).^2))

%%
% Generating pdf by counting number of smaples in the discretized volume 
% considering only 10^3 samples.
Ns3=zeros(size(Y1));
for ii=1:length(Y1(:,1))
    for jj=1:length(Y2(1,:))
       dist=squeeze(X0s (:,1:10^3))-repmat([Y1(ii,jj);X_mean(2:dim);Y2(ii,jj);X_mean(dim+2:end)],1,10^3);
       Ns3(ii,jj)=sum(sum(abs(dist)<grid_dist)==2*dim)/(2^(2*dim)*prod(grid_dist)*10^3);
        
    end
end

% Generating pdf by counting number of smaples in the discretized volume 
% considering only 10^4 samples.
Ns4=zeros(size(Y1));
for ii=1:length(Y1(:,1))
    for jj=1:length(Y2(1,:))
       dist=squeeze(X0s (:,1:10^4))-repmat([Y1(ii,jj);X_mean(2:dim);Y2(ii,jj);X_mean(dim+2:end)],1,10^4);
       Ns4(ii,jj)=sum(sum(abs(dist)<grid_dist)==2*dim)/(2^(2*dim)*prod(grid_dist)*10^4);
        
    end
end

% Generating pdf by counting number of smaples in the discretized volume 
% considering only 10^5 samples.
Ns5=zeros(size(Y1));
for ii=1:length(Y1(:,1))
    for jj=1:length(Y2(1,:))
       dist=squeeze(X0s (:,1:10^5))-repmat([Y1(ii,jj);X_mean(2:dim);Y2(ii,jj);X_mean(dim+2:end)],1,10^5);
       Ns5(ii,jj)=sum(sum(abs(dist)<grid_dist)==2*dim)/(2^(2*dim)*prod(grid_dist)*10^5);
        
    end
end



figure
 
subplot(2,3,[1 4])
surf(Y1,Y2,squeeze(prob_plot(end,:,:)))
hold on


plot3(Y1,Y2,Ns5,'or')
xlabel('Position','Fontsize',16)
ylabel('Velocity','Fontsize',16)
zlabel('probability density','Fontsize',16)
lg=legend(' GKA 10^5 samples','Monte Carlo 10^5 samples');
set(lg,'Location','northeast','Fontsize',16)
axis tight


subplot(2,3,2)

surf(Y1,Y2,squeeze(prob_plot(end-1,:,:)))
hold on


plot3(Y1,Y2,Ns,'or')
xlabel('Position')
ylabel('Velocity')
zlabel('probability density')
lg=legend(' GKA 10^4 samples','Monte Carlo 10^5 samples');
set(lg,'Location','northeast','Fontsize',16)
axis tight

subplot(2,3,3)

surf(Y1,Y2,squeeze(prob_plot(end-2,:,:)))
hold on


plot3(Y1,Y2,Ns,'or')
xlabel('Position')
ylabel('Velocity')
zlabel('probability density')
lg=legend(' GKA 10^3 samples','Monte Carlo 10^5 samples');
set(lg,'Location','northeast','Fontsize',16)
axis tight

subplot(2,3,5)

surf(Y1,Y2,squeeze(prob_plot(end,:,:)))
hold on


plot3(Y1,Y2,Ns4,'or')
xlabel('Position')
ylabel('Velocity')
zlabel('probability density')
lg=legend(' GKA 10^5 samples' ,'Monte Carlo 10^4 samples');
set(lg,'Location','northeast','Fontsize',16)
axis tight

subplot(2,3,6)

surf(Y1,Y2,squeeze(prob_plot(end,:,:)))
hold on


plot3(Y1,Y2,Ns3,'or')
xlabel('Position')
ylabel('Velocity')
zlabel('probability density')
lg=legend('GKA 10^5 samples','Monte Carlo 10^3 samples');
set(lg,'Location','northeast','Fontsize',16)
axis tight



%%

% Computing and plotting relative errors
err3=sum(sum(abs(Ns-Ns3).^2))/sum(sum(abs(Ns).^2));
err4=sum(sum(abs(Ns-Ns4).^2))/sum(sum(abs(Ns).^2));
err5=sum(sum(abs(Ns-Ns5).^2))/sum(sum(abs(Ns).^2));
err5_GM=sum(sum(abs(Ns-squeeze(prob_plot(end-1,:,:))).^2))/sum(sum(abs(Ns).^2));
err4_GM=sum(sum(abs(Ns-squeeze(prob_plot(end-2,:,:))).^2))/sum(sum(abs(Ns).^2));
err3_GM=sum(sum(abs(Ns-squeeze(prob_plot(end-3,:,:))).^2))/sum(sum(abs(Ns).^2));
err2_GM=sum(sum(abs(Ns-squeeze(prob_plot(end-4,:,:))).^2))/sum(sum(abs(Ns).^2));
 
figure
loglog(10.^[ 5  4  3],[err5 err4 err3],'o')
hold on
loglog(10.^[5 4 3 2  ],[err5_GM err4_GM err3_GM   err2_GM ],'x')
xlabel('samples size')
ylabel('relative L_2 error')
grid on
legend('Monte Carlo','GKA')