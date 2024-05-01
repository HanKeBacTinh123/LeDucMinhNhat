%%%%%%%%%%%%%%%%%%% 
%% LMS Algorithm example 8.6 %% 
%%%%%%%%%%%%%%%%%%% 
 
%----- Givens -----% 

a = 3.98;  %  a = 7.96*lambda/2 
%N = input('   How many element do you want in uniform linear array?   ');   % number of elements in array 
N=50; 
PhiS = 0; 
PhiI = -60; 
%----- Desired Signal & Interferer -----% 
T=1E-3; 
t=(1:100)*T/100; 
it=1:100; 
S=cos(2*pi*t/T); 
PhiS = PhiS*pi/180;                  % desired user AOA 
I = randn(1,100);   
PhiI = PhiI*pi/180;                    % interferer AOA 

%----- Create Array Factors for each user's signal for linear array -----% 
 
vS = []; vI = []; 
i=1:N; 
vS=exp(-1j*2*pi*a*(cos(PhiS-2*pi*(i-1)/N)-cos(2*pi*i/N))).'; 
vI=exp(-1j*2*pi*a*(cos(PhiI-2*pi*(i-1)/N)-cos(2*pi*i/N))).'; 
 
%----- Solve for Weights using LMS -----% 
 
w = zeros(N,1);     snr = 10;  % signal to noise ratio 
 X=(vS+vI); 
Rx=X*X'; 
mu=1/(4*real(trace(Rx))) 
%mu = input('What is step size?') 
wi=zeros(N,max(it)); 
for n = 1:length(S) 
    x = S(n)*vS + I(n)*vI; 
    %y = w*x.'; 
    y=w'*x; 
     
    e = conj(S(n)) - y;      esave(n) = abs(e)^2; 
%    w = w +mu*e*conj(x); 
    w=w+mu*conj(e)*x; 
    wi(:,n)=w; 
    yy(n)=y; 
end 
w = (w./w(1));    % normalize results to first weight 
 
%----- Plot Results -----% 
 
Phi = -pi/2:.01:pi/2; 
AF = zeros(1,length(Phi)); 
 
% Determine the array factor for linear array 
 
for i = 1:N 
    AF = AF + w(i)'.*exp(-1j*2*pi*a*(cos(Phi-2*pi*(i-1)/N)-cos(2*pi*i/N))); 
end 
 
figure 
plot(Phi*180/pi,abs(AF)/max(abs(AF)),'k') 
xlabel('AOA (deg)') 
ylabel('|AF_n|') 
axis([-90 90 0 1.1]) 
set(gca,'xtick',[-90 -60 -30 0 30 60 90]) 
grid on 
 
 
figure;plot(it,abs(wi(1,:)),'kx',it,abs(wi(2,:)),'ko',it,abs(wi(3,:)),'ks',it,abs(wi(4,:)),'k+',it,abs(wi(5,:)),'kd','markersize',2) 
xlabel('Iteration no.') 
ylabel('|weights|') 
figure;plot(it,esave,'k') 
xlabel('Iteration no.') 
ylabel('Mean square error')

% tính SLL%%%%
n=length(Phi);
% tìm giá trị max, max_2, SLL = max/max_2
A = zeros(1,n);
for i = 1:n
    A(i) = abs(AF(i))/max(abs(AF));
end
m1=max(A);
for j = 1:n
    if A(j) == m1;
        break;
    end
end
Dmax=j;
GocToi=Phi(Dmax)*180/pi;

max_2=0;
for i=2:(n-1)
 if ((A(i)-A(i-1))>=0)&((A(i)-A(i+1))>=0)& (A(i)~=m1)
     if A(i)>= max_2
         max_2 = A(i);
     end
 end
end 
 m2=max_2;
 SLL = m1/m2

%tính beamwidth%
 for k=2:Dmax
    if (A(k-1) < m1/(sqrt(2))) & (A(k) > m1/(sqrt(2)))
        break;
    end    
 end
 D1=k;
GocPhu1 = Phi(D1)*180/pi;
 for h=Dmax:(n-1)
    if (A(h) > m1/(sqrt(2))) & (A(h+1) < m1/(sqrt(2))) 
        break;
    end    
 end
 D2=h;
GocPhu2 = Phi(D2)*180/pi; 
Beamwidth = GocPhu2-GocPhu1 
