function x_d = duff_stochint(t,x,dim,m ,M,C,K,kappa,f,Om)


% Set output vector
x_d=zeros(2*dim+2*dim*m,1); %m= No. of Gaussian white noises

% State space matrix
A=[zeros(dim) eye(dim); -M\K -M\C];
% Nonlinearity in state space form 
F_nlin=[zeros(dim,1);-M\[kappa.*x(1:dim).^3]];
% Jacobian of the nonlinearity in state space form 
DF_nlin=[zeros(dim,2*dim);-M\diag(3.*[kappa.*x(1:dim).^2]) zeros(dim)];

% Trajectory
x_d(1:2*dim,1)=A*x(1:2*dim)+F_nlin+[zeros(dim,1); M\f].*sin(Om*t);

% Linearized flow map, resp. the vector V(t) cf. Eq. (20) 
tmp=(A+DF_nlin)*reshape(x(2*dim+1:2*dim+ 2*dim*m),2*dim,m);
x_d(2*dim+1:2*dim+(2*dim)*m,1)=tmp(:);



end