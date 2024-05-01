%testGSCusingNN
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
NN_input = [];
DESIRED_output = [];
L = length(phi); %number of coming signals
for i=1:length(phi)
    if phi(i)==phi0 && theta(i)==theta0 
        chiso = i;
    end;
end;
g(chiso) = 1;
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
for p = 1:200
    nv = 0.005*p;
    %coding
    d=length(phi);
    C = []; %ma trận C của vector lái (ma trận ràng buộc tuyến tính)
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
    end;
    u= sqrt(sv(jj)).*input(jj, :);
    u = imresize(u, [1, N]);
    U=[U,transpose(u)];%matrix of beamformer inputs
    Mx=Mx+transpose(u+noise);%mixed beamformer inputs plus noise
    uu=transpose(u+noise)*a;%outter product giving a M_new x N matrix
    Utotal=Utotal+uu;
    a=transpose(a);
    C = [C,a];
    Rs=Rs+[a*sv(jj)*ctranspose(a)];
    a=[1];
    end;
    %tin hieu tron lai mix
    mix = transpose(Mx);
    
    %Find total correlation matrix
    R=Rs+Rn;%total correlation matrix
    
    %Tìm ma trận Ca:
    Ca = null(ctranspose(C));
    
    %Find optimum weight vector
    wq = C*pinv(ctranspose(C)*C)*g;
    wao = pinv(ctranspose(Ca)*R*Ca)*ctranspose(Ca)*R*wq;
    wopt = wq - Ca*wao;
    nn_input = reshape(ctranspose(Ca)*transpose(Utotal), [N_new 1]);
    desired_output = ctranspose(wq)*transpose(Utotal);
    NN_input = [NN_input nn_input];
    DESIRED_output = [DESIRED_output desired_output];
end

NN_input = transpose(NN_input);
DESIRED_output = transpose(DESIRED_output);

%Initiate training and testing data
[rows, columns] = size(NN_input);
num_train = 0.5;
num_test = 1-num_train;
num_training_rows = round(num_train * rows);
training_indices = randperm(rows, num_training_rows);
test_indices = setdiff(1:rows, training_indices);

NN_input_training = NN_input(training_indices, :);
NN_input_test = NN_input(test_indices, :);
DESIRED_output_training = DESIRED_output(training_indices, :);
DESIRED_output_test = DESIRED_output(test_indices, :);

disp(['Starting calculate using neural network']);
tic;

%Building Neural Network
n_i = N_new;
n_h = 40;
n_o = N;

W1 = randn(n_i, n_h);
b1 = randn(1, n_h);
W2 = randn(n_h, n_o);
b2 = randn(1, n_o);

epochs = 5000;
learning_rate = 0.1;
costs = []; %list of costs for each epoch
Errors = [];

for e = 1:epochs
    %feed foward
    hidden_layer_input = NN_input_training * W1 + b1;
    hidden_layer_output = sigmoid(hidden_layer_input); % f(x) = sigmoid(x);
    output_layer_input = hidden_layer_output * W2 + b2;
    predicted_output = sigmoid(output_layer_input); 

    %cost function
    Errors = abs(DESIRED_output_training - predicted_output);
    squarederrors = (Errors).^2;
    sumsquareerrors = sum(squarederrors(:));
    mse = sumsquareerrors / numel(Errors);
    cost = sqrt(mse); % rmse of errors
    if mod(e, 200) == 0
        disp(['rmse for epoch ', num2str(e),' is: ',num2str(cost)]);
    end
    if cost <= 1e-6 
        cost = 0;
        disp(['rmse for epoch ', num2str(e),' of n_h ', num2str(n_h), ' is: ',num2str(cost)]);
        costs = [costs cost];
        break;
    end
    costs = [costs cost];

    %back propagation
    error = DESIRED_output_training - predicted_output;
    d_predicted_output = error .* ((sigmoid(predicted_output) .* (1 - sigmoid(predicted_output)))); % f'(x) = f(x)*(1-f(x))
    
    error_hidden_layer = d_predicted_output * transpose(W2);
    d_hidden_layer = error_hidden_layer .* ((sigmoid(hidden_layer_output) .* (1 - sigmoid(hidden_layer_output))));

    %update weights and bias
    W2 = W2 + transpose(hidden_layer_output) * d_predicted_output * learning_rate;
    b2 = b2 + sum(d_predicted_output) * learning_rate;
    W1 = W1 + transpose(NN_input_training) * d_hidden_layer * learning_rate;
    b1 = b1 + sum(d_hidden_layer) * learning_rate;
end
elapsedTime = toc;
disp(['Thời gian tính toán NN : ', num2str(elapsedTime), ' giây.']);

%calculate for testing data
z1 = NN_input_test * W1 + b1;
a1 = sigmoid(z1);
z2 = a1 * W2 + b2;
predicted_test = sigmoid(z2);
Errors_test = abs(DESIRED_output_test - predicted_test);
squarederrors_test = (Errors_test).^2;
sumsquareerrors_test = sum(squarederrors_test(:));
mse_test = sumsquareerrors_test / numel(Errors_test);
cost_test = sqrt(mse_test); % rmse of errors
disp(['RMSE khi dùng NN cho tập test: ', num2str(cost_test)]);

