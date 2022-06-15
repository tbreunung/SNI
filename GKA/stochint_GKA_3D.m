clear all
close all

dim=3;
k=1;
kc=0.1;
D=0.01;
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
fb=sigma.*[zeros(dim); M\eye(dim)]; %eye(dim)

N_gwn=size(fb,2);
Om=1.2;%1.9;
T=2*pi/Om;

A = [zeros(dim) eye(dim);...
    -M\K    -M\C];
S = @(x)kappa.*x.^3;  



NL=@(z)[zeros(dim,1);-M\S(z(1:dim))];
G=@(t)[zeros(dim,1); M\f].*sin(Om*t);
RHS=@(t,z) A*z + NL(z) + G(t);
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

%[-1.499;  1.343]
%x0= [repmat(-1.2235 ,dim,1); repmat(4.2724,dim,1)];
x0= [repmat(-1.499 ,dim,1); repmat(1.343,dim,1)];

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
 N_smpl=10^3;

Nperiod=1;
N_smpl_per_T=5*10^4;
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

% B=@(t,x) fb ;
% %z0_low ;
% 
% 
% X1end=zeros(2*dim,N_smpl);
% tic
% 
% 
% 
% parfor ii=1:N_smpl
%     %SDE=sde(RHS,B,'StartState',z0);
%     
%     [X ,~]=EM_end(RHS,B,Nperiod*N_smpl_per_T, z0,dt,N_gwn);
%     X1end(:,ii)=X.';
%     % X1_no_mean=X1-repmat(smpl_mean,smpl_length,1);
%     % X1_vars=zeros(2*dim,2*dim,smpl_length);
%     % X1_means(:,:,ii)=smpl_mean;
%     % X11_var(:,ii)=(X1_no_mean(:,1).^2);
%     % X22_var(:,ii)=(X1_no_mean(:,2).^2);
%     % X12_var(:,ii)=(X1_no_mean(:,1).*X1_no_mean(:,2));
%     
%     if ii==floor(ii/100)*100
%         ii/100
%     end
% end
% EM_clock=toc

%%
% N_tau=5* Nperiod;
% X2=zeros(2*dim,N_tau,N_smpl);
% 
% Tend=Nperiod*T/N_tau;
% X0=z_PO(1,:).';
% tic
% parfor ii=1:N_smpl
%     
%     x0=X0;
%     for np=1:N_tau
%         
%         [t, z] = ode45(@(t,z)duff_stochint(t,z,dim,N_gwn,M,C,K,kappa,f,Om), (np-1)*Tend+[0 Tend], [x0 ;  fb(:)] ,opts); %;zeros(4*dim^2,1)
%         Kt=zeros(2*dim,2*dim,length(t));
%         V=reshape(z(:,2*dim+1:end),length(t),2*dim,N_gwn);
%         for tt=1:length(t)
%             tmp=reshape(V(tt,:,:),2*dim,N_gwn);
%         Kt(:,:,tt)=tmp*tmp.';
%         end
%         Vars=trapz(t,Kt,3);
%          %reshape(z(end,2*dim+(2*dim)*m+1:end),2*dim,2*dim);
%         x0=mvnrnd(z(end,1:2*dim),Vars).';
%         X2(:,np,ii)=x0;
%     end
%     if ii==floor(ii/100)*100
%         ii/100
%     end
% end
% PS_clock=toc
% 
% figure
% plot( X1end(1,:), X1end(dim+1,:),'xk')
% hold on
% plot(squeeze(X2(1,end,:)),squeeze(X2(dim+1,end,:)),'sg')
% plot(z0(1),z0(dim+1),'or')
% xlabel('Position')
% ylabel('Velocity')
% legend('Euler-Maruyama','Proposed scheme','Periodic orbit')
% 
% speed_up=EM_clock/PS_clock
% rel_diff_mean=(mean(X1end(1:dim:end,:),2)-mean(squeeze(X2(1:dim:end,end,:)),2))./mean(X1end(1:dim:end,:),2)
% rell_diff_std=(std(X1end(1:dim:end,:),0,2)-std(squeeze(X2(1:dim:end,end,:)),0,2))./std(X1end(1:dim:end,:),0,2)
%%
%tmp_vars=zeros(2*dim,2*dim,N_tau);
% N_smpl=10^5;
% tol=10^-15;
% N_tau=5*Nperiod;
% 
% Tend=Nperiod*T/N_tau;
% X0=z_PO(1,:).';
% 
% tic
% X0s=repmat(X0,1,N_smpl);
% Vars=zeros(2*dim,2*dim,N_smpl);
% for np=1:N_tau
% parfor ii=1:N_smpl
% %     exit_flag=false;
% %    
%      %x0=X0;
% %     while exit_flag==false
%          
%             
%             [t, z] = ode45(@(t,z)duff_stochint(t,z,dim,N_gwn,M,C,K,kappa,f,Om), (np-1)*Tend+[0 Tend], [X0s(:,ii) ;  fb(:)] ,opts); %;zeros(4*dim^2,1)
%             Kt=zeros(2*dim,2*dim,length(t));
%             V=reshape(z(:,2*dim+1:end),length(t),2*dim,N_gwn);
%             for tt=1:length(t)
%                 tmp=reshape(V(tt,:,:),2*dim,N_gwn);
%                 Kt(:,:,tt)=tmp*tmp.';
%             end
%             Vars(:,:,ii)=trapz(t,Kt,3);
%             X0s(:,ii)=z(end,1:2*dim).';
%            % if ii==floor(ii/100)*100
%            %     ii/100
%            % end
% end
% %size(X0s)
% X0s_nostep=X0s;
% N_s=randi(N_smpl,N_smpl,1);
% parfor ii=1:N_smpl
%     [V,L]=eig(squeeze(Vars(:,:,N_s(ii))));
%     idxs=diag(L)>tol;
%     sigs=zeros(2*dim,1);
%     tmp_smpl=randn(sum(idxs),1);
%     Red_var_inv=eye(sum(idxs))/L(idxs,idxs);
%     
%     %prob_along_path(np,ii)=(1/sqrt((2*pi)^sum(idxs)*det(L(idxs,idxs))))*exp(-0.5*(tmp_smpl.'*tmp_smpl));
%     sigs(idxs)=tmp_smpl.*sqrt(diag(L(idxs,idxs)));
%     
%     X0s(:,ii)= X0s(:,ii)+(V*sigs);
%     %mvnpdf(x0.',z(end,1:2*dim),Vars)
% %     X3_wstep(:,np,ii)=x0;
% %     Vars(:,:,np,ii)=Vars_tmp;
% %     X3_nostep(:,np,ii)=z(end,1:2*dim);
% end
% %         dist=x0([2:dim dim+2:end])-X_mean([2:dim dim+2:2*dim]);
% %         if sum(abs(dist)<grid_dist([2:dim dim+2:2*dim]))==2*dim-2
% %             exit_flag=true;
% %         else
% %             x0=X0;
% %         end
%     
% np  
%  
% end
%  toc
 
 %%
N_smpl=10^5;
tol=10^-15;
N_tau=5*Nperiod;

Tend=Nperiod*T/N_tau;
X0=z_PO(1,:).';

tic
X0s=repmat(X0,1,N_smpl);
Vars=zeros(2*dim,2*dim,N_smpl);
X0s_nostep=repmat(X0,1,N_smpl);
parfor ii=1:N_smpl
    %     exit_flag=false;
    %
    X0s(:,ii)=X0;
    %     while exit_flag==false
    for np=1:N_tau
    
    [t, z] = ode45(@(t,z)duff_stochint(t,z,dim,N_gwn,M,C,K,kappa,f,Om), (np-1)*Tend+[0 Tend], [X0s(:,ii) ;  fb(:)] ,opts); %;zeros(4*dim^2,1)
    Kt=zeros(2*dim,2*dim,length(t));
    V=reshape(z(:,2*dim+1:end),length(t),2*dim,N_gwn);
    for tt=1:length(t)
        tmp=reshape(V(tt,:,:),2*dim,N_gwn);
        Kt(:,:,tt)=tmp*tmp.';
    end
    Vars(:,:,ii)=trapz(t,Kt,3);
    %X0s(:,ii)=z(end,1:2*dim).';
   [V,L]=eig(squeeze(Vars(:,:,ii)));
    idxs=diag(L)>tol;
    sigs=zeros(2*dim,1);
    tmp_smpl=randn(sum(idxs),1);
    Red_var_inv=eye(sum(idxs))/L(idxs,idxs);
    
    %prob_along_path(np,ii)=(1/sqrt((2*pi)^sum(idxs)*det(L(idxs,idxs))))*exp(-0.5*(tmp_smpl.'*tmp_smpl));
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
% figure
% %tpdf(end,ii)=[];
% plot(squeeze(X0s(1,:)),squeeze(X0s(dim+1,:)),'or')
% hold on
%  plot( X1end(1,:), X1end(dim+1,:),'xk')
%  hold on
%  plot(squeeze(X2(1,end,:)),squeeze(X2(dim+1,end,:)),'sg')
% xlabel('Position')
% ylabel('Velocity')
%%
%prob_path=prod(prob_along_path(1:N_tau-1,:),1);
%prob_path=prob_path./sum(prob_path);
X_mean=squeeze(mean(X0s(:,:),2));
X_min=X_mean-3*std(X0s(:,:),0,2);
X_max=X_mean+3*std(X0s(:,:),0,2);
grid_dist=(X_max-X_min)./2/100;
%grid_dist(1:dim:end)=grid_dist(1:dim:end)/10;

Y1=linspace(X_min(1),X_max(1),100);
Y2=linspace(X_min(1+dim),X_max(1+dim),100);


[Y1,Y2]=meshgrid(Y1,Y2);
Y1_vec=Y1(:).';
Y2_vec=Y2(:).';


  
N1=floor(N_smpl/nw);
N2=mod(N_smpl,nw);
ids1=[1:N2+1 repmat(N2+1,1,nw-N2-1)];
ids2=(0:nw-1).*N1;
w_idx=zeros(2,nw);
w_idx(1,:)=ids1+ids2;
w_idx(2,1:end-1)=w_idx(1,2:end)-1;
w_idx(2,end)=N_smpl;
prob_df=zeros(nw,length(Y1_vec));

tol2=10^-6;

parfor ii=1:nw
    for jj=w_idx(1,ii):w_idx(2,ii)
        Vars_tmp=squeeze(Vars(:,:,jj));
        [V,L]=eig(Vars_tmp);
        idxs= diag(L)>tol;
        %L =L+  diag(idxs).*min(L(diag(diag(L)>tol))); %
        lengY=length(Y1_vec);
        center_point=[Y1_vec;repmat(X_mean(2:dim),1,lengY);Y2_vec;repmat(X_mean(dim+2:2*dim),1,lengY)];
         
        grid_points_proj=V(:,idxs).'*center_point;
        Red_var_inv=eye(sum(idxs))/L(idxs,idxs);
        proj_mean=V(:,idxs).'*X0s_nostep(:,jj);
        proj_grid_minus_mean=grid_points_proj-repmat(proj_mean,1,lengY);
        efcn_exp=-0.5.*sum(Red_var_inv*(proj_grid_minus_mean.^2));%
        tmp_pdf =(1/sqrt((2*pi)^(2*dim)*det(L(idxs,idxs)))).*exp(efcn_exp);%
        
        tmp_pdf = tmp_pdf./(tol2^(2*dim-sum(idxs)));
        tmp_pdf(vecnorm(V(:,not(idxs)).'*center_point,'Inf')>tol2)=0;
        
        prob_df(ii,:)= prob_df(ii,:)+tmp_pdf; %
%         if jj==floor(jj/1000)*1000
%             jj/1000
%         end
         
    end
    
end
prob_plot_tmp=sum(prob_df)/N_smpl;

prob_plot=reshape(prob_plot_tmp,100,100);


%%
% Ns=zeros(size(Y1));
% for ii=1:length(Y1(:,1))
%     for jj=1:length(Y2(1,:))
%        dist=X0s -repmat([Y1(ii,jj); X_mean(2:dim);Y2(ii,jj); X_mean(dim+2:2*dim)],1,N_smpl);
%        Ns(ii,jj)=sum(sum(abs(dist)<grid_dist)==2*dim)/(2^(2*dim)*prod(grid_dist)*N_smpl);
%         
%     end
% end
% 
% pdf_MC_interp= griddedInterpolant(Y1',Y2',Ns');
% p_mean_MC=pdf_MC_interp(X_mean(1:dim:2*dim).');
% rel_err =sum(sum(abs(Ns -squeeze(prob_plot(end,:,:))).^2))/sum(sum(abs(Ns).^2))

Xred=zeros(2,N_smpl*dim);
for jj=1:dim
   Xred(:,1+(jj-1)*N_smpl:(jj)*N_smpl)=X0s(jj:dim:end,:);  
end
Ns2=zeros(size(Y1));
for ii=1:length(Y1(:,1))
    for jj=1:length(Y2(1,:))
      dist=Xred -repmat([Y1(ii,jj);Y2(ii,jj); ],1,N_smpl*dim);
      Ns2(ii,jj)=sum(sum(abs(dist)<grid_dist(1:dim:end))==2)/(2^(2)*prod(grid_dist(1:dim:end))*N_smpl*dim);
 
        
    end
end
pdf_MC_interp= griddedInterpolant(Y1',Y2',Ns2');
p_mean_red=pdf_MC_interp(X_mean(1:dim:2*dim).');
Ns_red=Ns2.*p_mean_red^(dim-1);
rel_err =sum(sum(abs(Ns_red -prob_plot).^2))/sum(sum(abs(Ns_red).^2))


%%
figure
surf(Y1,Y2,prob_plot)
hold on


plot3(Y1,Y2,Ns_red,'or')
%plot3(Y1,Y2,Ns,'dg')
xlabel('Position $q_1$','Interpreter','latex','FontSize',16)
ylabel('Velocity $\dot{q}_1$','Interpreter','latex','FontSize',16)
zlabel('probabiliy density','Interpreter','latex','FontSize',16)
axis tight
lg=legend(' GKA','Monte Carlo - uncoupled limit');
set(lg,'Location','NorthEast')
%%
