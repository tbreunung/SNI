function x_d =Duf_w_eq_of_var(t,x,M,C,K,Om,ampl,kappa)
dim=1;
A=[zeros(dim) eye(dim); -M\K -M\C];

F_nlin=[zeros(dim,1);-M\[kappa*x(1)^3]];
DF_nlin=[zeros(dim,2*dim);-M\[3*kappa*x(1)^2] 0];


x_d(1:2*dim,1)=A*x(1:2*dim)+F_nlin+[0; ampl]*sin(Om*t);

tmp=(A+DF_nlin)*reshape(x(2*dim+1:end),2*dim,2*dim);%[x(2*dim+1:3*dim+1)   x(4*dim+1:5*dim+1)  ]);%([x(1:2).';  x(3:4).'  ]);
x_d(2*dim+1:(2*dim)+(2*dim)^2,1)=tmp(:);%reshape(tmp.',4,1);%

    
end