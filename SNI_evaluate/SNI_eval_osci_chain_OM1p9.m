clear all
close all
clc


% -------------------------------
%   Same file as SNI_eval_osci_chain.m 
%   The only difference is that the frequency of the external harmonic
%   forcing is set to Om=1.9
%
%   For documentation please have a look at SNI_eval_osci_chain.m 
%
% -------------------------------
dim=1;
k=1;
kc=0.1;
D=0.02;
fa=0.1;
m=1;
sigma=0.01;


M=m.*eye(dim);
if dim==1
    K=1;
else 
    K=spdiags(ones(dim,1)*[-kc k+kc -kc], [-1 0 1], sparse(dim,dim));
end

if dim>2
    K=K+diag([0 kc.*ones(1,dim-2) 0]);
end
    
kappa=0.5*ones(dim,1); %0.5
C=D.*eye(dim);
f=fa.*ones(dim,1);%[1;zeros(dim-1,1)];% 
fb=sigma.*[zeros(dim,1); M\ones(dim,1)]; %eye(dim)

N_gwn=size(fb,2);
Om=1.9; %1.2
T=2*pi/Om;

A = [zeros(dim) eye(dim);...
    -M\K    -M\C];
S = @(x)kappa.*x.^3;  



NL=@(z)[zeros(dim,1);-M\S(z(1:dim))];
G=@(t)[zeros(dim,1); M\f].*sin(Om*t);
RHS=@(t,z) A*z + NL(z) + G(t);
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

 x0= [repmat(-2.646 ,dim,1); repmat(0.8564,dim,1)];

[~, z_trans] = ode45(@(t,z)RHS(t,z), [0 500*T],x0,opts); % Transients

z0=z_trans(end,:)';
[~, z_PO] = ode45(@(t,z)RHS(t,z), [0 T], z0);

[~, z_trans] = ode45(@(t,z)RHS(t,z), [0 500*T],zeros(2*dim,1),opts); % Transients

z0=z_trans(end,:)';
[~, z_PO_0] = ode45(@(t,z)RHS(t,z), [0 T], z0);

figure
plot(z_PO(:,1),z_PO(:,dim+1))
hold on
plot(z_PO(1,1),z_PO(1,dim+1),'dg')
plot(z_PO_0(:,1),z_PO_0(:,dim+1))


%%


Nperiod=1;
N_smpl_per_T=2*10^6;
dt=T/N_smpl_per_T;
t=0:dt:T;
B=@(t,x) 0.*fb ;
z0=z_PO(1,:).';%z0_low ;
% tic
% [X ,~]=EulerMaruyama(RHS,B,Nperiod*N_smpl_per_T, z0,dt);
% toc
tic
[X2 ,~]=EM_end(RHS,B,Nperiod*N_smpl_per_T, z0,dt,N_gwn);
toc
[~, z_ode] = ode45(@(t,z)RHS(t,z), [0 Nperiod*T], z0,opts);

figure
%plot(X(1,end),X(dim+1,end),'xk')
hold on
plot(X2(1),X2(dim+1),'xk')
plot(z_ode(end,1),z_ode(end,dim+1),'dg')


err=norm(z_ode(end,:)-X2.')
%%
c = gcp;%parcluster('local'); % build the 'local' cluster object
nw = c.NumWorkers;
%parpool(c)
%%
N_smpl=10^3;
B=@(t,x) fb ;
%z0_low ;


X1end=zeros(2*dim,N_smpl);
tic



parfor ii=1:N_smpl
    %SDE=sde(RHS,B,'StartState',z0);
    
    [X ,~]=EM_end(RHS,B,Nperiod*N_smpl_per_T, z0,dt,N_gwn);
    X1end(:,ii)=X.';
    % X1_no_mean=X1-repmat(smpl_mean,smpl_length,1);
    % X1_vars=zeros(2*dim,2*dim,smpl_length);
    % X1_means(:,:,ii)=smpl_mean;
    % X11_var(:,ii)=(X1_no_mean(:,1).^2);
    % X22_var(:,ii)=(X1_no_mean(:,2).^2);
    % X12_var(:,ii)=(X1_no_mean(:,1).*X1_no_mean(:,2));
    
    if ii==floor(ii/100)*100
        ii/100
    end
end
EM_clock=toc

%%
N_tau=5* Nperiod;
X2=zeros(2*dim,N_tau,N_smpl);

Tend=Nperiod*T/N_tau;
X0=z_PO(1,:).'; 
tol=10^-15;
tic
parfor ii=1:N_smpl
    
    x0=X0;
    for np=1:N_tau
        
        [t, z] = ode45(@(t,z)duff_stochint(t,z,dim,N_gwn,M,C,K,kappa,f,Om), (np-1)*Tend+[0 Tend], [x0 ;  fb(:)] ,opts); %;zeros(4*dim^2,1)
        Kt=zeros(2*dim,2*dim,length(t));
        V=reshape(z(:,2*dim+1:end),length(t),2*dim,N_gwn);
        for tt=1:length(t)
            tmp=reshape(V(tt,:,:),2*dim,N_gwn);
        Kt(:,:,tt)=tmp*tmp.';
        end
         Vars=trapz(t,Kt,3);
        [V,L]=eig(Vars);
        idxs=diag(L)>tol;
        sigs=zeros(2*dim,1);
        tmp_smpl=randn(sum(idxs),1);
        sigs(idxs)=tmp_smpl.*sqrt(diag(L(idxs,idxs)));
        
        x0=z(end,1:2*dim).'+(V*sigs);
%          
       % x0=mvnrnd(z(end,1:2*dim),Vars).';
        X2(:,np,ii)=x0;
    end
    if ii==floor(ii/100)*100
        ii/100
    end
end
PS_clock=toc

figure
plot(X1end(1,:), X1end(dim+1,:),'xk')
hold on
plot(squeeze(X2(1,end,:)),squeeze(X2(dim+1,end,:)),'sg')
plot(z0(1),z0(dim+1),'or')
xlabel('Position')
ylabel('Velocity')
lg=legend('Euler-Maruyama','SNI \tau=T/5','x0');
set(lg,'Location','NorthWest')
speed_up=EM_clock/PS_clock
rel_diff_mean=max(mean(X1end,2)-mean(squeeze(X2(:,end,:)),2)) 
%./mean(X1end(1:dim:end,:),2)
Cov_EM=zeros(2*dim,2*dim);
Cov_SNI=zeros(2*dim,2*dim);
for jj=1:2*dim
    %Cov_EM(jj,jj)=cov(X1end(jj,:));
    %Cov_SNI(jj,jj)=cov(squeeze(X2(jj,end,:)));
    for ii=jj+1:2*dim
        Cov_EM([ii jj],[ii jj])=cov(X1end(ii,:),X1end(jj,:));
        %Cov_EM(jj,ii)=Cov_EM(ii,jj);
        Cov_SNI([ii jj],[ii jj])=cov(squeeze(X2(ii,end,:)),squeeze(X2(jj,end,:)));
       % Cov_SNI(jj,ii)=Cov_SNI(ii,jj);
        %rell_diff_std=(std(X1end(1:dim:end,:),0,2)-std(squeeze(X2(1:dim:end,end,:)),0,2))
    end
end
var_err=abs((Cov_EM-Cov_SNI));
max(var_err(:))
%./std(X1end(1:dim:end,:),0,2)
%%
%tmp_vars=zeros(2*dim,2*dim,N_tau);

%%
