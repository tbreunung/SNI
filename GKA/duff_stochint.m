function x_d = duff_stochint(t,x,dim,m ,M,C,K,kappa,f,Om)
% dim=3;
% k=1;
% kc=0.1;
% D=0.01*k;
% kappa=0.5.*ones(dim,1);
% ampl=0.1.*ones(dim,1);
% K=spdiags(ones(dim,1)*[-kc k+kc -kc], [-1 0 1], sparse(dim,dim))+diag([0 kc.*ones(1,dim-2) 0]);
% C=D.*eye(dim);
% M=eye(dim);
%Om = pars(1,:);

 
x_d=zeros(2*dim+2*dim*m,1); %+4*dim^2

A=[zeros(dim) eye(dim); -M\K -M\C];
 
F_nlin=[zeros(dim,1);-M\[kappa.*x(1:dim).^3]];
DF_nlin=[zeros(dim,2*dim);-M\diag(3.*[kappa.*x(1:dim).^2]) zeros(dim)];


x_d(1:2*dim,1)=A*x(1:2*dim)+F_nlin+[zeros(dim,1); M\f].*sin(Om*t);

 
tmp=(A+DF_nlin)*reshape(x(2*dim+1:2*dim+ 2*dim*m),2*dim,m);%([x(1:2).';  x(3:4).'  ]);
x_d(2*dim+1:2*dim+(2*dim)*m,1)=tmp(:);%reshape(tmp.',4,1);%

%
%tmp=reshape(x(2*dim+1:2*dim+(2*dim)*m),2*dim,m)*(reshape(x(2*dim+1:2*dim+(2*dim)*m),2*dim,m)).';%([x(1:2).';  x(3:4).'  ]);

%x_d(2*dim+1+(2*dim)*m:end,1)=tmp(:);


end