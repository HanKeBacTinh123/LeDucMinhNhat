%testGSC
%dữ liệu nhập vào cho lập trình
clear; clc;
M = 4;
g = [0;0;0]; 
theta = [-60;-50;-40];
theta0 = -60;
sv = [1, 1, 1];
nv = 0.1;
for i=1:length(theta)
    if theta(i)==theta0
        chiso = i;
    end;
end;
g(chiso) = 1;
%lấy tín hiệu âm thanh vào
[data, fs] = audioread('nhac.mp3');
[tapam1, fs1] = audioread('tapam1.mp3');
[tapam2, fs2] = audioread('tapam2.mp3');

%xác định N
N = length(data);

%xử lý tín hiệu âm thanh vào
kenhtraidata = data(:, 1); %chỉ lấy kênh trái của data
kenhtraitapam1 = tapam1(:, 1);
kenhtraitapam2 = tapam2(:, 1);

%đưa tín hiệu các âm thanh về cùng độ dài snapshot N
kenhtraidata = kenhtraidata(1:N);
kenhtraitapam1 = kenhtraitapam1(1:N);
kenhtraitapam2 = kenhtraitapam2(1:N);
input = [transpose(kenhtraidata);transpose(kenhtraitapam1);transpose(kenhtraitapam2)];

%thu nhiễu
nhieu = audioread('tiengmuaroi.mp3');
kenhtrainhieu = nhieu(:,1);
kenhtrainhieu = kenhtrainhieu(1:N);
nhieu = transpose(kenhtrainhieu);
giatrichuanhoa = max(abs(nhieu));
nhieuchuanhoa = nhieu ./ giatrichuanhoa;

d=length(theta);
C = []; %ma trận C của vector lái (ma trận ràng buộc tuyến tính)
A=[];
a=[1];
a0=[1];
U=[];
Utotal=zeros(N,M);
Mx=zeros(N,1);
Rs=zeros(M);%signal correlation matrix
if nv==0 
    nv1=0.00001;
else
    nv1=nv;
end;
noise=sqrt(nv1).*nhieuchuanhoa;%noise
Rn=nv1*eye(M);%noise correlation matrix
delta=0.5;% inter-element spacing over wavelength

%Find steering vectors and correlation matices
for jj=1:d
    for i=1:(M-1)
    a=[a exp(-1j*i*2*pi*delta*sin(theta(jj)*pi/180))];%steering vector
    end
if jj == chiso
    vectorlai = a;
    disp('Âm thanh cần phải thu');
    %audiowrite('TinhieumongmuonGSC.wav',transpose(input(jj, :)),fs);
%     soundsc(transpose(input(jj, :)),fs);
%     %biểu diễn tín hiệu gốc data
%     figure;
%     t = (1:N)/fs;
%     plot(t, transpose(input(jj, :)), 'color', 'b');
%     xlabel('Thời gian (giây)');
%     ylabel('Biên độ');
%     title('Đồ thị tín hiệu âm thanh cần thu');
%     grid;
end;
u= sqrt(sv(jj)).*input(jj, :);
u = imresize(u, [1, N]);
U=[U,transpose(u)];%matrix of beamformer inputs
Mx=Mx+transpose(u+noise);%mixed beamformer inputs plus noise
uu=transpose(u+noise)*a;%outter product giving a NxM matrix
Utotal=Utotal+uu;
a=transpose(a);
C = [C,a];
Rs=Rs+[a*sv(jj)*ctranspose(a)];
a=[1];
end;
%tin hieu tron lai mix
mix = transpose(Mx);
%pause(10);
disp('Âm thanh tại ngõ tới có cả nhiễu và can nhiễu');
%audiowrite('TinhieutrontaingovaoGSC.wav',real(mix), fs);
% soundsc(real(mix), fs);
% figure;
% t = (1:N)/fs;
% plot(t, mix, 'color', 'r');
% xlabel('Thời gian (giây)');
% ylabel('Biên độ');
% title('Đồ thị tín hiệu âm thanh trộn với tạp âm và nhiễu');
% grid;
%Find total correlation matrix
R=Rs+Rn;%total correlation matrix

%Tìm ma trận Ca:
Ca = null(ctranspose(C));

%Find optimum weight vector
wq = C*pinv(ctranspose(C)*C)*g;
wao = pinv(ctranspose(Ca)*R*Ca)*ctranspose(Ca)*R*wq;
wopt = wq - Ca*wao;

%Plot beam-pattern
aa=[1];
r=0.1; %Angle resolution, [deg.]
Cout = [];
for k=1:180/r; 
th(k)=-90+k*r;  
for i=1:(M-1)
    aa=[aa exp(-1j*i*2*pi*delta*sin(th(k)*pi/180))];%steering vector
end
aa=transpose(aa);
Cout = [Cout, aa];
aa=[1];
end;
g_out = ctranspose(Cout)*wopt;
figure;
plot([-(90-r):r:90],(abs(g_out).^2),'b','LineWidth',1.5);
xlabel('Angle [degree]');ylabel('Beam pattern');title(['Beam pattern of GSC Beamformer. Number of antennas: ',num2str(M),'. Signal variance: ', num2str(sv), '. Noise variance: ', num2str(nv), '. AoA: -60, -50, -40']);
grid;
%Plot beamformer output 
yy=[];
y=ctranspose(wopt)*transpose(Utotal);
yy=[yy,y];
k=find(theta==theta0);
startindex = floor(N/2);
endindex = startindex + 200;
yyplot = yy(startindex:endindex);
Mxplot = Mx(startindex:endindex);
signaloutput = U(:,k);
signaloutputplot = signaloutput(startindex:endindex);
figure;
plot(abs(yyplot),'b','LineWidth',1.5);hold;
plot(abs(signaloutputplot),'g--','LineWidth',2);
plot(abs(Mxplot),'r.-','LineWidth',1);hold
xlabel('Time samples');ylabel('Amplitude');title(['GSC Beamformer output. Number of antennas: ',num2str(M),'. Signal variance: ', num2str(sv), '. Noise variance: ', num2str(nv), '. AoA: -60, -50, -40']);
legend('Beamformer output','Desired signal','Beamformer input');
grid;
%plot output signal
figure;
t = (1:N)/fs;
plot(t, transpose(yy), 'color', 'g');
xlabel('Thời gian (giây)');
ylabel('Biên độ');
title('Đồ thị tín hiệu âm thanh sau khi xử lý');
grid;
soundngora = real(transpose(yy));
%pause(10);
disp('Âm thanh sau khi xử lý');
%audiowrite('TinhieutaingoraGSC_sosanh.wav',soundngora, fs);
%soundsc(soundngora, fs);
%Calculate Root Mean Square Error
output = yy;
desired = U(:,k);
e = output - transpose(desired);
MSE = abs(e).^2;
rmse = sqrt(sum(MSE)*(1/N));
disp(['rmse for nv = ', num2str(nv),' is: ',num2str(rmse)]);
disp('Kết thúc xử lý');
% %vẽ biểu đồ sự thay đổi rmse
% nv = 0;
% %tính toán rmse
% values_of_rmse1 = [];
% values_of_rmse2 = [];
% for i=1:50
%     nv = nv + 0.02;
%     d=length(theta);
%     C = []; %ma trận C của vector lái (ma trận ràng buộc tuyến tính)
%     A=[];
%     a=[1];
%     a0=[1];
%     U=[];
%     Utotal=zeros(N,M);
%     Mx=zeros(N,1);
%     Rs=zeros(M);%signal correlation matrix
%     if nv==0 
%         nv1=0.00001;
%     else
%         nv1=nv;
%     end;
%     noise=sqrt(nv1).*nhieuchuanhoa;%noise
%     Rn=nv1*eye(M);%noise correlation matrix
%     delta=0.5;% inter-element spacing over wavelength
%     %Find steering vectors and correlation matices
%     for jj=1:d
%         for i=1:(M-1)
%         a=[a exp(-1j*i*2*pi*delta*sin(theta(jj)*pi/180))];%steering vector
%         end
%     u= sqrt(sv(jj)).*input(jj, :);
%     u = imresize(u, [1, N]);
%     U=[U,transpose(u)];%matrix of beamformer inputs
%     Mx=Mx+transpose(u+noise);%mixed beamformer inputs plus noise
%     uu=transpose(u+noise)*a;%outter product giving a NxM matrix
%     Utotal=Utotal+uu;
%     a=transpose(a);
%     C = [C,a];
%     Rs=Rs+[a*sv(jj)*ctranspose(a)];
%     a=[1];
%     end;
%     %Find total correlation matrix
%     R=Rs+Rn;%total correlation matrix
%     %Tìm ma trận Ca:
%     Ca = null(ctranspose(C));
%     %Find optimum weight vector
%     wq = C*pinv(ctranspose(C)*C)*g;
%     wao = pinv(ctranspose(Ca)*R*Ca)*ctranspose(Ca)*R*wq;
%     wopt = wq - Ca*wao;
%     %Plot beam-pattern
%     aa=[1];
%     ag=[];%array gain
%     r=0.1; %Angle resolution, [deg.]
%     for k=1:180/r; 
%     th(k)=-90+k*r;  
%     for i=1:(M-1)
%         aa=[aa exp(-1j*i*2*pi*delta*sin(th(k)*pi/180))];%steering vector
%     end
%     aa=transpose(aa);
%     %Cout = [Cout, aa];
%     aa=[1];
%     end;
%     %Plot beamformer output
%     yy=[];
%     y=ctranspose(wopt)*transpose(Utotal);
%     yy=[yy,y];
%     k=find(theta==theta0);
%     %Calculate Root Mean Square Error
%     output = yy;
%     aftermix = transpose(Mx);
%     desired = U(:,k);
%     e1 = output - transpose(desired);
%     MSE1 = abs(e1).^2;
%     rmse = sqrt(sum(MSE1)*(1/N));
%     values_of_rmse1 = [values_of_rmse1 rmse];
%     e2 = output - aftermix;
%     MSE2 = abs(e2).^2;
%     rmse = sqrt(sum(MSE2)*(1/N));
%     values_of_rmse2 = [values_of_rmse2 rmse];
% end;
% figure;
% x = 0.02:0.02:1;
% y = values_of_rmse1;
% plot(x,y,'b','LineWidth',2);hold;
% xlabel('nv'); ylabel('RMSE');
% title('Đồ thị RMSE (gốc và sau xử lý) theo sự thay đổi noise variance');
% grid;
% 
% figure;
% x = 0.02:0.02:1;
% y = values_of_rmse2;
% plot(x,y,'b','LineWidth',2);hold;
% xlabel('nv'); ylabel('RMSE');
% title('Đồ thị RMSE (sau xử lý và tín hiệu trộn) theo sự thay đổi noise variance');
% grid;
% disp('Kết thúc tính toán vòng lặp');