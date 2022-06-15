function [Y, Ts] = EulerMaruyama (mu, sigma ,N, x0 ,h)
 
dim=length(x0);
Y = zeros ( dim ,N+1);
Ts = h.*(0:1:N);

 Y( :,1 ) = x0 ;
 %h = T/N;
 sqrth = sqrt (h) ;
 for n = 1 :N
 Y(:,n+1) = Y(:,n) + mu((n-1)*h,Y(:,n) ).* h + sigma ((n-1)*h,Y(:,n)).* sqrth.*randn ;
 end
 end