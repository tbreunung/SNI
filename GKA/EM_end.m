function [Y, Ts] = EM_end (mu, sigma ,N, x0 ,h,m)
Ts = h.*N;
 sqrth = sqrt (h) ;
 Y=x0;
 for n = 1 :N
 Y = Y + mu((n-1)*h,Y).* h + sqrth.*sigma ((n-1)*h,Y)* randn(m,1) ;
 end
 
 end