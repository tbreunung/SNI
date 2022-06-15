clear all
close all
clc


dim=100;

% Set false if parameters are loaded from file, true if they need to be
% generated.
new_pars=false;

% Set ture if parameters should be saved. 
save_pars=false;


% source file to load or write parameters
src=['osci_array_rand_par_N=' num2str(dim) '.mat'];



if new_pars==true
    m=0.5+1.5*rand(dim,1);
    ks=0.5+1.5*rand(dim,1);
    kc=0.05+0.15*rand(dim,1);
    cs=0.01+rand(dim,1)*0.02;
    
    kappa=0.1*rand(dim,1);
    
    fa=0.1*randn(dim,1);
    f_m=randn(dim,1);
    Om=1+randn(1); %1.2
    
    x0= randn(2*dim,1);
    
    sigma=0.01;
    if save_pars==true
        save(src,'m','ks','kc','cs','kappa','fa','f_m','Om','x0','sigma')
    end
else
    load(src)
end



% -------------------------------
%   The remainder is the same as SNI_eval_osci_chain.m, only that the 
%   parameter vallues differ.
%
%   For documentation please have a look at SNI_eval_osci_chain.m 
%
% -------------------------------



M=diag(m);
if dim==1
    K=ks;
else 
    K=spdiags([-kc ks+kc -kc], [-1 0 1], sparse(dim,dim));
end

if dim>2
    K=K+diag([0 ; kc(2:end-1); 0]);
end
    
 %0.5
C=diag(cs);
 
fb=sigma.*[zeros(dim,1); M\f_m]; %eye(dim)

N_gwn=size(fb,2);

T=2*pi/Om;

A = [zeros(dim) eye(dim);...
    -M\K    -M\C];
S = @(x)kappa.*x.^3;  

NL=@(z)[zeros(dim,1);-M\S(z(1:dim))];
G=@(t)[zeros(dim,1); M\fa].*sin(Om*t);
RHS=@(t,z) A*z + NL(z) + G(t);
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

%%
Nperiod=1;
 
[~, z_ode] = ode45(@(t,z)RHS(t,z), [0 Nperiod*T], x0,opts);

N_smpl_per_T=5*10^5;
dt=T/N_smpl_per_T;
B=@(t,x) 0.*fb ;
%[X2 ,~]=EulerMaruyama(RHS,B,Nperiod*N_smpl_per_T, x0,dt);

%N_smpl_per_T=2*10^5;
%dt=T/N_smpl_per_T;
tic
[X2_end ,~]=EM_end(RHS,B,Nperiod*N_smpl_per_T, x0,dt,N_gwn);
toc

figure
%plot(X(1,end),X(dim+1,end),'xk')

%plot(X2(1,:),X2(dim+1,:))
hold on
plot(z_ode(:,1),z_ode(:,dim+1))
legend('Euler','ode45')
err=norm(z_ode(end,:)-X2_end.')
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
    
    [X ,~]=EM_end(RHS,B,Nperiod*N_smpl_per_T,x0,dt,N_gwn);
    X1end(:,ii)=X.';
    % X1_no_mean=X1-repmat(smpl_mean,smpl_length,1);
    % X1_vars=zeros(2*dim,2*dim,smpl_length);
    % X1_means(:,:,ii)=smpl_mean;
    % X11_var(:,ii)=(X1_no_mean(:,1).^2);
    % X22_var(:,ii)=(X1_no_mean(:,2).^2);
    % X12_var(:,ii)=(X1_no_mean(:,1).*X1_no_mean(:,2));
    
    if ii==floor(ii/10)*10
        ii/10
    end
end
EM_clock=toc

%%
N_tau=5* Nperiod;
X2=zeros(2*dim,N_tau,N_smpl);

Tend=Nperiod*T/N_tau;
 
tol=10^-15;
tic
parfor ii=1:N_smpl
    
    x_tau=x0;
    for np=1:N_tau
        
        [t, z] = ode45(@(t,z)duff_stochint(t,z,dim,N_gwn,M,C,K,kappa,fa,Om), (np-1)*Tend+[0 Tend], [x_tau ;  fb(:)] ,opts); %;zeros(4*dim^2,1)
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
        
        x_tau=z(end,1:2*dim).'+(V*sigs);
%          
       % x0=mvnrnd(z(end,1:2*dim),Vars).';
        X2(:,np,ii)=x_tau;
    end
    if ii==floor(ii/100)*100
        ii/100
    end
end
PS_clock=toc
speed_up=EM_clock/PS_clock

figure
plot(X1end(1,:), X1end(dim+1,:),'xk')
hold on
plot(squeeze(X2(1,end,:)),squeeze(X2(dim+1,end,:)),'sg')
xlabel('Position')
ylabel('Velocity')
lg=legend('Euler-Maruyama','SNI \tau=T/5','x0');
set(lg,'Location','NorthWest')
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
