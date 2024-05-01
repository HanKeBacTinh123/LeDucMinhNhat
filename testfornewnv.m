%testGSC
%dữ liệu nhập vào cho lập trình
M = 4;
g = [0;0;0]; 
phi = [-60;-150;120];
phi = phi * pi/180;
phi0 = -60*pi/180;
theta = [30;50;80];
theta = theta * pi/180;
theta0 = 30*pi/180;
sv = [1, 1, 1];
nv = 0.1;
for i=1:length(phi)
    if phi(i)==phi0 && theta(i)==theta0 
        chiso = i;
    end;
end;
g(chiso) = 1;
L = length(phi); %number of coming signals
%lấy tín hiệu âm thanh vào
[data, fs] = audioread('nhac.mp3');
[tapam1, fs1] = audioread('tapam1.mp3');
[tapam2, fs2] = audioread('tapam2.mp3');
data = data(1:length(data)/40,:);

%xác định N
N = length(data);
M_new = M * M;
N_new = N * (M_new - L);

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

%coding
d=length(phi);
C = []; %ma trận C của vector lái (ma trận ràng buộc tuyến tính)
M_new = M * M;
A=[];
a=[1];
a0=[1];
U=[];
Utotal=zeros(N,M_new);
Mx=zeros(N,1);
Rs=zeros(M_new);%signal correlation matrix
if nv==0 
    nv1=0.00001;
else
    nv1=nv;
end;
noise=sqrt(nv1).*nhieuchuanhoa;%noise
Rn=nv1*eye(M_new);%noise correlation matrix
delta=0.5;% inter-element spacing over wavelength

x_i = zeros(d,1);
y_i = zeros(d,1);
nuy_i = zeros(d,1);
v_i = zeros(d,1);

dx = [];
dy = [];
vitri_x = 0;
vitri_y = 0;
for i = 1:M
    dx = [dx vitri_x];
    vitri_x = vitri_x + 1;
    dy = [dy (i-1)*ones(1, M)];
    i = i+1;
end;
final_dx = dx;
for i = 1:(M-1)
    dx = flip(dx);
    final_dx = [final_dx dx];
    i = i + 1;
end;
dx = dy;
dy = final_dx;
D = vertcat(dx, dy);
dx = dx*delta;
dy = dy*delta;

%Find steering vectors and correlation matices
for jj=1:d
    x_i(jj) = cos(phi(jj))*sin(theta(jj));
    y_i(jj) = sin(phi(jj))*sin(theta(jj));
    nuy_i(jj) = -2*pi*x_i(jj);
    v_i(jj) = -2*pi*y_i(jj);
    for i=1:(M_new - 1)
    a=[a exp(1j*(dx(i+1)*nuy_i(jj) + dy(i+1)*v_i(jj)))];%steering vector
    end;
if jj == chiso
    vectorlai = a;
    %biểu diễn tín hiệu gốc data
    figure;
    t = (1:N)/fs;
    plot(t, transpose(input(jj, :)), 'color', 'b');
    xlabel('Thời gian (giây)');
    ylabel('Biên độ');
    title('Đồ thị tín hiệu âm thanh cần thu');
    grid;
end;
u= sqrt(sv(jj)).*input(jj, :);
u = imresize(u, [1, N]);
U=[U,transpose(u)];%matrix of beamformer inputs
Mx=Mx+transpose(u+noise);%mixed beamformer inputs plus noise
uu=transpose(u+noise)*a;%outter product giving a N x M_new matrix
Utotal=Utotal+uu;
a=transpose(a);
C = [C,a];
Rs=Rs+[a*sv(jj)*ctranspose(a)];
a=[1];
end;
%tin hieu tron lai mix
mix = transpose(Mx);

figure;
t = (1:N)/fs;
plot(t, mix, 'color', 'r');
xlabel('Thời gian (giây)');
ylabel('Biên độ');
title('Đồ thị tín hiệu âm thanh trộn với tạp âm và nhiễu');
grid;

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
r=1; %Angle resolution, [deg.]
G_out = [];

for h = 1:90/r
    x_new_i = zeros(360,1);
    y_new_i = zeros(360,1);
    nuy_new_i = zeros(360,1);
    v_new_i = zeros(360,1);
    th(h)=0+h*r;
    th(h)=th(h)*pi/180;
    Cout = [];
    for k=1:360/r
    ph(k)=-180+k*r;
    ph(k)=ph(k)*pi/180;
    x_new_i(k) = cos(ph(k))*sin(th(h));
    y_new_i(k) = sin(ph(k))*sin(th(h));
    nuy_new_i(k) = -2*pi*x_new_i(k);
    v_new_i(k) = -2*pi*y_new_i(k);
    for i=1:(M_new-1)
        aa=[aa exp(1j*(dx(i+1)*nuy_new_i(k)+dy(i+1)*v_new_i(k)))];%steering vector
    end
    aa=transpose(aa);
    Cout = [Cout, aa];
    aa=[1];
    end
    g_out = ctranspose(Cout)*wopt;
    G_out = [G_out g_out];
end

disp('Đồ thị biểu diễn độ lợi');
p = -(180-r):r:180;
tht = (0+r):r:90;
[p, tht] = meshgrid(p, tht);
z = abs(G_out);
figure;
mesh(((0+r):r:90), (-(180-r):r:180), z);
xlabel('Theta');
ylabel('Phi');
zlabel('g');
title(['Beam pattern of GSC Beamformer. Number of antennas: ',num2str(M_new),'. Signal variance: ', num2str(sv), '. Noise variance: ', num2str(nv), '. Theta: ', '. AoA(Phi,Theta): (-60, 30); (-150, 50); (120, 80)']);
grid;

figure;
contour(((0+r):r:90), (-(180-r):r:180), z);
xlabel('Theta');
ylabel('Phi');
zlabel('g');
title(['Beam pattern of GSC Beamformer. Number of antennas: ',num2str(M_new),'. Signal variance: ', num2str(sv), '. Noise variance: ', num2str(nv), '. Theta: ', '. AoA(Phi,Theta): (-60, 30); (-150, 50); (120, 80)']);
grid;

%Plot beamformer output 
yy=[];
y=ctranspose(wopt)*transpose(Utotal);
yy=[yy,y];
k=find(phi==phi0);
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
xlabel('Time samples');ylabel('Amplitude');
title(['Beam pattern of GSC Beamformer. Number of antennas: ',num2str(M_new),'. Signal variance: ', num2str(sv), '. Noise variance: ', num2str(nv), '. Theta: ', '. AoA(Phi,Theta): (-60, 30); (-150, 50); (120, 80)']);
legend('Beamformer output','Desired signal','Beamformer input');
grid;
% plot output signal
figure;
t = (1:N)/fs;
plot(t, transpose(yy), 'color', 'g');
xlabel('Thời gian (giây)');
ylabel('Biên độ');
title('Đồ thị tín hiệu âm thanh sau khi xử lý bởi GSC');
grid;
soundngora = real(transpose(yy));
%pause(10);
% audiowrite('TinhieutaingoraGSC_sosanh.wav',soundngora, fs);
%soundsc(soundngora, fs);
%Calculate Root Mean Square Error
output = yy;
desired = U(:,k);
e = output - transpose(desired);
MSE = abs(e).^2;
rmse = sqrt(sum(MSE)*(1/N));
disp(['Rmse for nv = ', num2str(nv),' by GSC is: ',num2str(rmse)]);

%Using NN for nv=0.1
nn_input_byNN = transpose(reshape(ctranspose(Ca)*transpose(Utotal), [N_new 1]));
desired_output_byNN = transpose(ctranspose(wq)*transpose(Utotal));

z1 = nn_input_byNN * W1 + b1;
a1 = sigmoid(z1);
z2 = a1 * W2 + b2;
predicted_byNN = sigmoid(z2);

wao_byNN = ctranspose(predicted_test * pinv(ctranspose(Ca)*transpose(Utotal)));
wopt_byNN = wq - Ca*wao_byNN;

%Plot beamformer output by NN
yy=[];
y=ctranspose(wopt_byNN)*transpose(Utotal);
yy=[yy,y];
k=find(phi==phi0);
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
xlabel('Time samples');ylabel('Amplitude');
title(['Beam pattern of GSC using NN Beamformer. Number of antennas: ',num2str(M_new),'. Signal variance: ', num2str(sv), '. Noise variance: ', num2str(nv), '. Theta: ', '. AoA(Phi,Theta): (-60, 30); (-150, 50); (120, 80)']);
legend('Beamformer output','Desired signal','Beamformer input');
grid;
% plot output signal
figure;
t = (1:N)/fs;
plot(t, transpose(yy), 'color', 'g');
xlabel('Thời gian (giây)');
ylabel('Biên độ');
title('Đồ thị tín hiệu âm thanh sau khi xử lý bởi GSC cải tiến bởi NN');
grid;
soundngora = real(transpose(yy));
%pause(10);
% audiowrite('TinhieutaingoraGSC_sosanh.wav',soundngora, fs);
%soundsc(soundngora, fs);
%Calculate Root Mean Square Error
output = yy;
desired = U(:,k);
e = output - transpose(desired);
MSE = abs(e).^2;
rmse = sqrt(sum(MSE)*(1/N));
disp(['Rmse for nv = ', num2str(nv),' by GSC using NN is: ',num2str(rmse)]);


%test for 10 random noise variance
disp(['Test for 10 new random noise variance']);
random_values = sort(0.005 + (1-0.005)*rand(1,10));
rmse_orgSigAndGSC = [];
rmse_orgSigAndNN = [];
rmse_desireResdAndNN = [];
for i=1:10
    nv = random_values(i);
    Utotal_nv = zeros(N,M_new);
    if nv==0 
        nv1=0.00001;
    else
        nv1=nv;
    end;
    noise=sqrt(nv1).*nhieuchuanhoa;%noise
    Rn=nv1*eye(M_new);%noise correlation matrix
    
    for jj=1:length(phi)
        uu_nv = transpose(transpose(U(:, jj)) + noise) * transpose(C(:, jj));
        Utotal_nv = Utotal_nv + uu_nv;
    end
    R_nv=Rs+Rn;%total correlation matrix
    
    %Find optimum weight vector for new noise variance
    wao_nv = pinv(ctranspose(Ca)*R_nv*Ca)*ctranspose(Ca)*R_nv*wq;
    wopt_nv = wq - Ca*wao_nv;
    
    %Plot beamformer output for new noise variance
    yy=[];
    y=ctranspose(wopt_nv)*transpose(Utotal_nv);
    yy=[yy,y];
    k=find(phi==phi0);
    startindex = floor(N/2);
    endindex = startindex + 200;
    yyplot = yy(startindex:endindex);
    Mxplot = Mx(startindex:endindex);
    signaloutput = U(:,k);
    signaloutputplot = signaloutput(startindex:endindex);
    % figure;
    % plot(abs(yyplot),'b','LineWidth',1.5);hold;
    % plot(abs(signaloutputplot),'g--','LineWidth',2);
    % plot(abs(Mxplot),'r.-','LineWidth',1);hold
    % xlabel('Time samples');ylabel('Amplitude');
    % title(['Beam pattern of GSC Beamformer. Number of antennas: ',num2str(M_new),'. Signal variance: ', num2str(sv), '. Noise variance: ', num2str(nv), '. Theta: ', '. AoA(Phi,Theta): (-60, 30); (-150, 50); (120, 80)']);
    % legend('Beamformer output','Desired signal','Beamformer input');
    % grid;
    %plot output signal
    % figure;
    % t = (1:N)/fs;
    % plot(t, transpose(yy), 'color', 'g');
    % xlabel('Thời gian (giây)');
    % ylabel('Biên độ');
    % title('Đồ thị tín hiệu âm thanh sau khi xử lý với nv mới');
    % grid;
    soundngora = real(transpose(yy));
    %pause(10);
    % audiowrite('TinhieutaingoraGSC_sosanh.wav',soundngora, fs);
    %soundsc(soundngora, fs);
    %Calculate Root Mean Square Error
    output = yy;
    desired = U(:,k);
    e = output - transpose(desired);
    MSE = abs(e).^2;
    rmse = sqrt(sum(MSE)*(1/N));
    disp(['Rmse tín hiệu gốc và tín hiệu tại ngõ ra với phương sai nhiễu ', num2str(nv),' là: ',num2str(rmse)]);
    rmse_orgSigAndGSC = [rmse_orgSigAndGSC rmse];
    %calculate using NN
    nn_input_nv = transpose(reshape(ctranspose(Ca)*transpose(Utotal_nv), [N_new 1]));
    desired_output = transpose(ctranspose(wq)*transpose(Utotal_nv));
    
    z1 = nn_input_nv * W1 + b1;
    a1 = sigmoid(z1);
    z2 = a1 * W2 + b2;
    predicted_test = sigmoid(z2);
    errors_test_nv = abs(desired_output - predicted_test);
    squarederrors_test_nv = (errors_test_nv).^2;
    sumsquareerrors_test_nv = sum(squarederrors_test_nv(:));
    mse_test_nv = sumsquareerrors_test_nv / numel(errors_test_nv);
    cost_test_nv = sqrt(mse_test_nv); % rmse of errors
    disp(['RMSE ngõ ra mong muốn d(n) và ngõ ra qua NN y(n) cho phương sai nhiễu ', num2str(nv), ' là: ', num2str(cost_test_nv)]);
    rmse_desireResdAndNN = [rmse_desireResdAndNN cost_test_nv];

    %calculate wopt by NN
    wao_NN = ctranspose(predicted_test * pinv(ctranspose(Ca)*transpose(Utotal_nv)));
    wopt_NN = wq - Ca*wao_NN;
    
    %Plot beamformer output for new noise variance
    yy_NN=[];
    y_NN=ctranspose(wopt_NN)*transpose(Utotal_nv);
    yy_NN=[yy_NN,y_NN];
    k=find(phi==phi0);
    startindex = floor(N/2);
    endindex = startindex + 200;
    yyNNplot = yy_NN(startindex:endindex);
    %MxNNplot = Mx(startindex:endindex);
    signaloutput = U(:,k);
    signaloutputplot = signaloutput(startindex:endindex);
    %Calculate Root Mean Square Error
    output = yy_NN;
    desired = U(:,k);
    e = output - transpose(desired);
    MSE = abs(e).^2;
    rmse_NN = sqrt(sum(MSE)*(1/N));
    disp(['Rmse tín hiệu gốc và tín hiệu tại ngõ ra qua NN với phương sai nhiễu ', num2str(nv),' là: ',num2str(rmse_NN)]);
    disp(['-----------------------------------------------']);
    rmse_orgSigAndNN = [rmse_orgSigAndNN rmse_NN];
end

figure;
x = random_values;
y1 = rmse_orgSigAndGSC;
plot(x,y1,'r.-','LineWidth',2);hold;
plot(x,y1, 'bx', 'MarkerSize', 8);
xlabel('Noise variance'); ylabel('RMSE Org Sig and Output by GSC');
title('Đồ thị RMSE (tín hiệu gốc và tín hiệu ngõ ra qua GSC) theo sự thay đổi phương sai nhiễu');
grid;

figure;
x = random_values;
y2 = rmse_orgSigAndNN;
plot(x,y2,'b','LineWidth',2);hold;
plot(x, y2, 'ro', 'MarkerSize', 8);
xlabel('Noise variance'); ylabel('RMSE Org Sig and Output by NN');
title('Đồ thị RMSE (tín hiệu gốc và tín hiệu ngõ ra qua GSC sử dụng NN) theo sự thay đổi phương sai nhiễu');
grid;

figure;
x = random_values;
y = rmse_desireResdAndNN;
plot(x,y,'g','LineWidth',2);hold;
plot(x, y, 'rx', 'MarkerSize', 8);
xlabel('Noise variance'); ylabel('RMSE Desired Res and By NN');
title('Đồ thị RMSE (ngõ ra mong muốn d(n) và ngõ ra qua NN y(n)) theo sự thay đổi phương sai nhiễu');
grid;