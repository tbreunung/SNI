clear all
close all

M=1;
K=1;
kappa=0.5;
C=0.02;
f=0.1;%0.1;
Om=1.2;
T=2*pi/Om;
sigma=0.01;

phi=0;

dim=1;

A = [zeros(dim) eye(dim);...
    -M\K    -M\C];
S = @(x) [kappa*x^3];
DS = @(x) [3*kappa*x^2];


NL=@(z)[zeros(dim,1);-M\S(z(1:dim))];
%DNL=@(z)[zeros(dim,2*dim);-M\DS(z(1:dim)) zeros(dim,dim)];
G=@(t)[zeros(dim,1); M\f].*sin(Om*t);
RHS=@(t,z) A*z +   G(t);
[~, z_trans_lin] = ode45(@(t,z)RHS(t,z), [phi phi+500*T], [0;  0]); % Transients
[~, z_lin] = ode45(@(t,z)RHS(t,z), [phi  phi+T], z_trans_lin(end,:)');

figure
plot(z_lin(:,1),z_lin(:,2))

hold on
RHS=@(t,z) A*z + NL(z) + G(t);
 
 
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

[~, z_trans_nl_low] = ode45(@(t,z)RHS(t,z), [phi+0 phi+500*T], [0;  0],opts); % Transients
z0_low=z_trans_nl_low(end,:)';
[~, z_nlin_low] = ode45(@(t,z)RHS(t,z), [phi+0  phi+T],z0_low );

 plot(z_nlin_low(:,1),z_nlin_low(:,2))


 % IC
 % phi=0     [-1.499;  1.343]
 % phi=T/4 all others  [1.186; 0.213]   
 % [-0.7965; 3.3752]
[~, z_trans_nl_high] = ode45(@(t,z)RHS(t,z), [phi+0 phi+500*T], [-1.2235 ;  4.2724]  ,opts); % Transients
z0_high=z_trans_nl_high(end,:)';
[~, z_nlin_high] = ode45(@(t,z)RHS(t,z), [phi+0  phi+T], z0_high);

plot(z_nlin_high(:,1),z_nlin_high(:,2))

legend('linear','low energy orbit','high energy orbit')

% figure
% plot(t,squeeze(Vars(1,1,:)),t,squeeze(Vars(2,1,:)),t,squeeze(Vars(2,2,:)))
% legend('K_{11}','K_{12}','K_{22}')

%%
compt_basin=false;
if compt_basin==true

 N_x1=-2:0.01:2;
 N_x2=-2:0.01:2;
 tol=0.1;
 map=-ones(length(N_x1),length(N_x2));
 for ii=1:length(N_x1)
     parfor jj=1:length(N_x2)
     [~, z_trans_nl] = ode45(@(t,z)RHS(t,z), [phi+0 phi+500*T], [N_x1(ii);  N_x2(jj)]); % Transients
     X_end(ii,jj,:)=z_trans_nl(end,:);
     if norm(z_trans_nl(end,:).'-z0_high,2)<tol
         map(ii,jj)=1
     else
         if norm(z_trans_nl(end,:).'-z0_low,2)<tol
              map(ii,jj)=0
         end
     end
         
     end
     ii
 end
 [NX1, NX2]=meshgrid(N_x1,N_x2);
 tmp1=X_end(:,:,1);
  tmp2=X_end(:,:,2);
  
  figure
%  plot(tmp1(:),tmp2(:),'xk')
  contourf(NX1,NX2,map,[-1 0 1])
  hold on
   plot(z0_high(1),z0_high(2),'dr')
   plot(z0_low(1),z0_low(2),'dg')
end 
 %%
Nperiod=1;

tic
N_smpl_per_T=10^4;
dt=T/N_smpl_per_T;
I=eye(2*dim);
t=phi:dt:phi+T;

B=@(t,x)  0.*[ 0; M\sigma];
RHS=@(t,z) A*z + NL(z) + G(t);

[X ,~]=EulerMaruyama(RHS,B,500*N_smpl_per_T, z0_high,dt);
[PO_EM ,TEM]=EulerMaruyama(RHS,B, N_smpl_per_T, X(:,end),dt);


PHI_EM=zeros(2*dim,2*dim,length(TEM));
PHI_EM(:,:,1)=eye(2*dim);
for tt=2:length(TEM)
    PHI_EM(:,:,tt)=(eye(2*dim)+(TEM(tt)-TEM(tt-1)).*(A+[0 0; -3.*kappa*PO_EM(1,tt-1).^2 0]))*PHI_EM(:,:,tt-1);
end 
 


[~,z2]=ode45(@(t,x)Duf_w_eq_of_var(t,x,M,C,K,Om,f,kappa),t,[z0_high; I(:)],opts );
% hold on
figure
plot(PO_EM(1,:),PO_EM(2,:))
hold on
plot(z2(:,1),z2(:,2))
legend('Euler-Majurama', 'ode45')
toc



 
%PHI_t0_T=reshape(z2(end,3:6),2*dim,2*dim);
PHI_t0_T=squeeze(PHI_EM(:,:,end));
%
PHI=zeros(2*dim,2*dim,1+(length(t)-1)*Nperiod);
Kt=PHI;
D=[  0; M\sigma].*[ 0 M\sigma] ;

 
PHI(:,:,1)=eye(2*dim);
Kt(:,:,1)=D;

for tt=2:length(t)
    PHI(:,:,tt)= squeeze(PHI_EM(:,:,tt));% reshape(z2(tt,3:6),2*dim,2*dim);%
    Kt(:,:,tt)=squeeze(PHI(:,:,tt))*D*squeeze(PHI(:,:,tt).');
    CT=D; 
    for nn=1:Nperiod-1
        CT=PHI_t0_T^(nn)*D*(PHI_t0_T^(nn)).';
        Kt(:,:,nn*(length(t)-1)+tt)=squeeze(PHI(:,:,tt))*CT*squeeze(PHI(:,:,tt)).';
    end
     
end



tlong=0:dt:Nperiod*T;

Vars=cumtrapz(tlong,Kt,3);

 
% for tt=1:length(t)
%     PHI(:,:,tt)= reshape(z2(tt,3:6),2*dim,2*dim);
%     Kt2(:,:,tt)=squeeze(PHI(:,:,tt))*CT*squeeze(PHI(:,:,tt).');  
% end
% Vars2=cumtrapz(t,Kt2,3);

toc
%%
%PO=z2(1:end,1:2);
PO=PO_EM.';
%PO=[PO; repmat(PO(1:end-1,:),Nperiod-1,1)];
tic
B=@(t,x)  [0; M\sigma] ;
z0=z0_high ;%PO_EM(:,1);%
N_smpl=10^4;
Ndelta= 10^2;
smpl_length=length(PO(:,1));



X1=zeros(2*dim,1+Nperiod*(N_smpl_per_T/Ndelta),N_smpl);
X1end=zeros(2*dim,N_smpl);
RHS=@(t,z) A*z + NL(z) + G(t+phi);

parfor ii=1:N_smpl 
%SDE=sde(RHS,B,'StartState',z0);
[X ,~]=EulerMaruyama(RHS,B,Nperiod*N_smpl_per_T, z0,dt);
X=X.';
%simByEuler(SDE,smpl_length-1,'DeltaTime',dt);
%X1(:,:,ii)=(X(1:Ndelta:end,:)-PO(1:Ndelta:end,:)).';
X1(:,:,ii)=(X(1:Ndelta:end,:)).';
%X1end(:,ii)=(X(end,:)-PO(end,:)).';
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



X1_mean=mean(X1,3);
X1_nomean=X1-repmat(X1_mean,1,1,N_smpl);
X11=mean(X1_nomean(1,:,:).^2,3);
X12=mean(X1_nomean(1,:,:).*X1_nomean(2,:,:),3);
X22=mean(X1_nomean(2,:,:).^2,3);
 
% X1_mean=mean(X1end,2);
% X1_nomean=X1end-repmat(X1_mean,1,N_smpl);
% X11=mean(X1_nomean(1,:).^2,2);
% X12=mean(X1_nomean(1,:).*X1_nomean(2,:),2);
% X22=mean(X1_nomean(2,:).^2,2);
Vars_MC=zeros(2*dim,2*dim,length(X11));
Vars_MC(1,1,:)=X11;
Vars_MC(1,2,:)=X12;
Vars_MC(2,1,:)=X12;
Vars_MC(2,2,:)=X22;
toc
 %squeeze(Vars(:,:,1+Ndelta:Ndelta:end))

for ii=1:length(Vars_MC(1,1,:))
err(:,:,ii)=abs(Vars(:,:,1+Ndelta*(ii-1))-Vars_MC(:,:,ii));%./abs(Vars_MC(:,:,ii)).*100;

end
%%
%close all
% figure 
% PO=PO_EM.';
% PO=[PO; repmat(PO(1:end-1,:),Nperiod-1,1)];
% plot(tlong,PO(:,1),'-',tlong,PO(:,2))
% hold on
% plot(tlong(1:Ndelta:end),X1_mean,'s')
% legend({'small noise prediction','position MC ', 'velocity MC'},'Location', 'northwest')
% xlabel('time')
% ylabel('mean')


figure
%subplot(2,1,1)
plot(tlong,squeeze(Vars(1,1,:)),'k' ,tlong ,squeeze(Vars(2,2,:)),'g',tlong,squeeze(Vars(1,2,:)),'b')
hold on
plot(tlong(1:Ndelta:end),X11,'--k',tlong(1:Ndelta:end),X22,'--g',tlong(1:Ndelta:end),X12,'--b')
legend({'$K_{qq}(t)$ small noise expansion', '$K_{\dot{q}\dot{q}}(t)$  small noise expansion','$K_{q\dot{q}}(t)$ small noise expansion',...
    '$K_{qq}(t)$ Monte Carlo','$ K_{\dot{q}\dot{q}}(t)$ Monte Carlo','$K_{q\dot{q}}(t)$ Monte Carlo'},'Location','northwest','Interpreter','latex')
xlabel('time')
ylabel('varinaces')
set(gca,'XTick',[0:T/4:T],'XTicklabels',{'0','T/4','T/2','3T/4','T'},'Xlim',[0 1.05*T])

% subplot(2,1,2)
% plot(tlong(1:Ndelta:end),squeeze(err(1,1,:)),tlong(1:Ndelta:end),squeeze(err(2,2,:)),tlong(1:Ndelta:end),squeeze(err(1,2,:)))  
% legend({' err. K_{xx}(t)', '  err. K_{vv}(t)',' err.  K_{vx}(t)'},'Location','northwest')
% %axis([tlong(1) tlong(end)  0 100])
% xlabel('time')
% ylabel('relative error')
%  

 
%%
figure
%contourf(NX1.',NX2.',map,[-1 0 1])
pl(1,:)=plot(squeeze(X1(1,end,:)),squeeze(X1(2,end,:)),'xk');
hold on
plot(X1_mean(1,end,:),X1_mean(2,end,:),'or');



s=0:0.01:2*pi;

Var_final=squeeze(Vars(:,:,end-1));
[vs,lams]=eig(Var_final);
v1=vs(:,1);
v2=vs(:,2);
std1=sqrt(lams(1,1));
std2=sqrt(lams(2,2));
Var_elip= 2*std1*cos(s).*v1+2*std2*sin(s).*v2;
 
pl(2,:)=plot(PO(end,1)+Var_elip(1,:),PO(end,2)+Var_elip(2,:),'--r','Linewidth',2);


 
   

%pl(3,:)= plot(z0_high(1),z0_high(2),'db');
pl(3,:)= plot(PO(end,1),PO(end,2),'dg');
%plot(z0(1),z0(2),'sb')
legend(pl,'Monte Carlo samples','Predicted Std. deviation', 'Low ampl. periodic orbit')
xlabel('Position')
ylabel('Velocity' )
axis equal
% [vs,lam]=eig(squeeze(K(:,:,end)))
%%
% Nperiod=10;
% 
% N_smpl_per_T=10^3;
% dt=T/N_smpl_per_T;
% 
% B=@(t,x)  0.*[0; M\sigma];
% RHS=@(t,z) A*z + NL(z) + G(t);
% 
% [X ,TEM]=EulerMaruyama(RHS,B,Nperiod*N_smpl_per_T, z0,dt);
% PO=z2(1:end,1:2);
% 
% PO=[PO; repmat(PO(1:end-1,:),Nperiod-1,1)];
% tlong=0:dt:Nperiod*T;
% 
% figure
% subplot(2,2,1)
% plot(TEM,X(1,:))
% hold on
% plot(tlong,PO(:,1))
% subplot(2,2,2)
% plot(TEM,X(1,:)-PO(:,1).')
% subplot(2,2,3)
% plot(TEM,X(2,:))
% hold on
% plot(tlong,PO(:,2))
% subplot(2,2,4)
% plot(TEM,X(2,:)-PO(:,2).')
%%
%  tic
% I=eye(2*dim);
% t=phi:dt:phi+Nperiod*T;
% [~,z2]=ode45(@(t,x)Duf_w_eq_of_var(t,x,M,C,K,Om,f,kappa),t,[z0; I(:)],opts);
% 
% PHI_t0_T=reshape(z2(end,3:6),2*dim,2*dim);
% 
% 
% PHI=zeros(2*dim,2*dim,length(t));
% Kt=PHI;
% Kt_lin=PHI;
% D=[0; M\sigma].*[0  M\sigma];
% for tt=1:length(t)
%   PHI(:,:,tt)= reshape(z2(tt,3:6),2*dim,2*dim);
%   Kt(:,:,tt)=squeeze(PHI(:,:,tt))*D*squeeze(PHI(:,:,tt).'); 
%  %  Kt_lin(:,:,tt)=expm(A.*t(tt))*D*expm(A.*t(tt)).';
%   
% end
% toc
% 
% 
% Vars_old=cumtrapz(t,Kt,3);


