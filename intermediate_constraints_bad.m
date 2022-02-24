clear;
%% 1. Load Data
% General Data

Xtrain = readtable('Xtrain_1000.txt');
[NX, CX, SX] = normalize(Xtrain, 'range');
Xtrain = NX{2:end,:};
Ytrain = readtable('Ytrain_1000.txt');
[NY, CY, SY] = normalize(Ytrain, 'range');
Ytrain = NY{:,:};

Xval = readtable('Xtrain_exdynamic5000.txt');
[NXval, CXval, SXvalv] = normalize(Xval, 'range');
Xval = NXval{2:end,:};
Yval = readtable('Ytrain_exdynamic5000.txt');
[NYval, CYval, SYval] = normalize(Yval, 'range');
Yval = NYval{:,:};
% 
% Data for Initialization
% Data for Net 1

X1 = Ytrain(1:end-1,[2,4,6,8,11]);
Y1 = Ytrain(2:end,[9,12]);
Xval1 = Yval(1:end-1,[2,4,6,8,11]);
Yval1 = Yval(2:end,[9,12]);
% Data for Net 2

X2 = [Xtrain(:,[1,2]), Ytrain(1:end-1,[9,12])];
Y2 = Ytrain(2:end,10);
Xval2 = [Xval(:,[1,2]), Yval(1:end-1,[9,12])];
Yval2 = Yval(2:end,10);
% Data for Net 3

X3 = [Xtrain(:,:), Ytrain(1:end-1,[2,4,6,8,11]), Ytrain(2:end,[9,12,10])];
Y3 = Ytrain(2:end,[1,3,5,7]);
Xval3 = [Xval(:,:), Yval(1:end-1,[2,4,6,8,11]), Yval(1:end-1,[9,10,12])];
Yval3 = Yval(2:end,[1,3,5,7]);
% Data for Net 4

X4 = [Ytrain(2:end,[1,3,5,7,10]), Ytrain(1:end-1,[2,4,6,8,11])];
Y4 = Ytrain(2:end,[2,4,6,8,11]);
Xval4 = [Yval(2:end,[1,3,5,7,10]), Yval(1:end-1,[2,4,6,8,11])];
Yval4 = Yval(2:end,[2,4,6,8,11]);
% 
% Data for Training the Composite Network


x2 = Xtrain(:,[1,2]);
x3 = Xtrain(:,:);
xval2 = Xval(:,[1,2]);
xval3 = Xval(:,:);
bigX = {x2';x3'};
bigY = {Y1';Y2';Y3';Y4'};
bigXval = {xval2';xval3'};
bigYval = {Yval1';Yval2';Yval3';Yval4'};

Xcomp = Xtrain(:,:);
Ycomp = Y4;
Xcomp2 = Xval(:,:);
Ycomp2 = Yval4;

h = 20;
% 

steps = 25;
Perf_Net4 = zeros(steps,1);
testPerf1 = zeros(steps,1);
testPerf2 = zeros(steps,1);
testPerf3 = zeros(steps,1);

fig1 = figure();
set(gcf,'Visible','on');
ax1 = axes('Parent', fig1);
per(1) = animatedline('Color',[0.75 0.5 0.098], 'Parent', ax1, 'DisplayName', 'BigNet');
per(2) = animatedline('Color',[0.85 0.1 0.1], 'Parent', ax1, 'DisplayName', 'test1');
per(3) = animatedline('Color',[0.1 0.85 0.1], 'Parent', ax1, 'DisplayName', 'test2');
per(4) = animatedline('Color',[0.098 0.098 0.85], 'Parent', ax1, 'DisplayName', 'test3');


% ylim([0 .01])
legend(per(1:4))
g = 'no';
tf1 = 'tansig';
tf2 = 'tansig';
tf3 = 'satlin';
tf4 = 'tansig';

for q = 1:steps


%% 2. Create Sub-Networks & initialize parameters
% Network 1
%     in:    T,  V, A, B, C,
%     out:    Rrxn, Qout

net1 = feedforwardnet([h,h],'trainscg');
net1.trainParam.showWindow = false;
net1.layers{1}.transferFcn = tf1;
net1.layers{2}.transferFcn = tf1;
[net1, tr1] = train(net1,X1',Y1','useGPU',g);
y = net1(Xval1');
perf1 = perform(net1,Yval1',y)
% Network 2
%     in:    Qa, Qb, Qout
%     out:    dVdt

net2 = feedforwardnet([h,h],'trainscg');
net2.trainParam.showWindow = false;
net2.layers{1}.transferFcn = tf2;
net2.layers{2}.transferFcn = tf2;
[net2, tr2] = train(net2,X2',Y2','useGPU',g);
y = net2(Xval2');
perf2 = perform(net2,Yval2',y)
% Network 3
%     in:    (Qa, Qb, Ta, Tb, Ca0, Cb0), (A, B, C, T, V), (Rrxn, dVdt, Qout)
%     out:    dCa_dt, dCb_dt, dCc_dt, dTdt

net3 = feedforwardnet([h,h],'traincgf');
net3.trainParam.showWindow = false;
net3.layers{1}.transferFcn = tf3;
net3.layers{2}.transferFcn = tf3;
[net3, tr3] = train(net3,X3',Y3','useGPU',g);
y = net3(Xval3');
perf3 = perform(net3,Yval3',y)
% Network 4
%     in:    dCa_dt, dCb_dt, dCc_dt, dTdt, dVdt, A, B, C, T, V
%     out:    A, B, C, T, V

net4 = feedforwardnet([h,h],'traincgf');
net4.trainParam.showWindow = false;
net4.layers{1}.transferFcn = tf4;
net4.layers{2}.transferFcn = tf4;
[net4, tr4] = train(net4,X4',Y4','useGPU',g);
y = net4(Xval4');
perf4 = perform(net4,Yval4',y)

%% 3. Create Structure for Combined Network
% 

bignet = feedforwardnet([h,h,2,h,h,1,h,h,4,h,h],'trainscg');
bignet.layers{1}.transferFcn = tf1;
bignet.layers{2}.transferFcn = tf1;
bignet.layers{3}.transferFcn = 'purelin';
bignet.layers{4}.transferFcn = tf2;
bignet.layers{5}.transferFcn = tf2;
bignet.layers{6}.transferFcn = 'purelin';
bignet.layers{7}.transferFcn = tf3;
bignet.layers{8}.transferFcn = tf3;
bignet.layers{9}.transferFcn = 'purelin';
bignet.layers{10}.transferFcn = tf4;
bignet.layers{11}.transferFcn = tf4;
bignet.numInputs = 2;
bignet.inputConnect  = [
                        0  0; % input to net1
                        0  0;
                        0  0;
                        1  0; % input to net2
                        0  0;
                        0  0;
                        0  1; % input to net3
                        0  0;
                        0  0;
                        0  0; % input to net4
                        0  0;
                        0  0;
                       ];
% bignet.outputConnect = [0 0 1 0 0 1 0 0 1 0 0 1];

              % from:|1in   1o 2in   2o 3in   3o 4in   4o| % to:
bignet.layerConnect = [0  0  0  0  0  0  0  0  0  0  0  1; % 1 in
                       1  0  0  0  0  0  0  0  0  0  0  0;
                       0  1  0  0  0  0  0  0  0  0  0  0; % 1 out
                       0  0  1  0  0  0  0  0  0  0  0  0; % 2 in
                       0  0  0  1  0  0  0  0  0  0  0  0;
                       0  0  0  0  1  0  0  0  0  0  0  0; % 2 out
                       0  0  1  0  0  1  0  0  0  0  0  1; % 3 in
                       0  0  0  0  0  0  1  0  0  0  0  0;
                       0  0  0  0  0  0  0  1  0  0  0  0; % 3 out
                       0  0  0  0  0  1  0  0  1  0  0  1; % 4 in
                       0  0  0  0  0  0  0  0  0  1  0  0;
                       0  0  0  0  0  0  0  0  0  0  1  0; % 4 out
                      ];

bignet.inputs{1}.size = 2;
bignet.inputs{2}.size = 6;
bignet.layers{12}.size = 5;
% set feedback loop delays
bignet.layerWeights{1,12}.delays = 1;
bignet.layerWeights{7,12}.delays = 1;
bignet.layerWeights{10,12}.delays = 1;
% name stuff so the diagram makes sense
bignet.layers{1}.name = 'Net1 in';
bignet.layers{2}.name = 'Hidden 1';
bignet.layers{4}.name = 'Net2 in';
bignet.layers{5}.name = 'Hidden 2';
bignet.layers{7}.name = 'Net3 in';
bignet.layers{8}.name = 'Hidden 3';
bignet.layers{10}.name = 'Net4 in';
bignet.layers{11}.name = 'Hidden 4';
bignet.layers{3}.name = 'Net1 out';
bignet.layers{6}.name = 'Net2 out';
bignet.layers{9}.name = 'Net3 out';
bignet.layers{12}.name = 'Net4 out';
bignet.outputs{3}.name = "Rrxn Qout";
bignet.outputs{6}.name = "dVdt";
bignet.outputs{9}.name = "dC_dt dTdt";
bignet.outputs{12}.name = "A B C T V";
bignet.inputs{1}.name = "net2 input";
bignet.inputs{2}.name = "net3 input";
T = cell2table(bignet.LW,"RowNames",{'in1','h1','out1','in2','h2','out2','in3','h3','out3','in4','h4','out4'},"VariableNames",{'in1','h1','out1','in2','h2','out2','in3','h3','out3','in4','h4','out4'});
% display diagram

%% 4. Copy Sub-Networks into the Structure
% Copy input weights

% net1


% net2
bignet.IW{4,1} = net2.IW{1,1}(:,1:2);

% net3
bignet.IW{7,2} = net3.IW{1,1}(:,1:6);

% net4


% Copy Layer Weights

% net1 weights
bignet.LW{1,12} = net1.IW{1,1};
bignet.LW{2,1} = net1.LW{2,1}; % in1 to h1
bignet.LW{3,2} = net1.LW{3,2}; %h1 to out1


% net2 weights
bignet.LW{4,3}(:,[1,2]) = net2.IW{1,1}(:,[3,4]); % out1 to in2
bignet.LW{5,4} = net2.LW{2,1}; % in2 to h2
bignet.LW{6,5} = net2.LW{3,2}; % h2 to out2

% net3 weights
bignet.LW{7,6} = net3.IW{1,1}(:,13); % out2 to in3
bignet.LW{7,3} = net3.IW{1,1}(:,[12,14]); % out1 to in3
bignet.LW{8,7} = net3.LW{2,1}; % in3 to h3
bignet.LW{9,8} = net3.LW{3,2}; % h3 to out3

% net4 weights
bignet.LW{10,6} = net4.IW{1,1}(:,5); % out2 to in4
bignet.LW{10,9} = net4.IW{1,1}(:,1:4); % out3 to in4
bignet.LW{10,12} = net4.IW{1,1}(:,1:5); % out4 to in4
bignet.LW{11,10} = net4.LW{2,1}; % in4 to h4
bignet.LW{12,11} = net4.LW{3,2}; % h4 to out4

% Copy Biases


% net1
bignet.b{1} = net1.b{1};
bignet.b{2} = net1.b{2};
bignet.b{3} = net1.b{3};

% net1
bignet.b{4} = net2.b{1};
bignet.b{5} = net2.b{2};
bignet.b{6} = net2.b{3};

% net1
bignet.b{7} = net3.b{1};
bignet.b{8} = net3.b{2};
bignet.b{9} = net3.b{3};

% net1
bignet.b{10} = net4.b{1};
bignet.b{11} = net4.b{2};
bignet.b{12} = net4.b{3};
view(bignet)
% bignet.LayerWeights{2,1}.learn = false; % turn off learning for now
% bignet.LayerWeights{4,3}.learn = false; % turn off learning for now
% bignet.LayerWeights{7,6}.learn = false; % turn off learning for now
% bignet.LayerWeights{7,3}.learn = false; % turn off learning for now
% bignet.LayerWeights{10,9}.learn = false; % turn off learning for now
% bignet.LayerWeights{10,6}.learn = false; % turn off learning for now
%% 5. Train Neural Network

bignet.trainParam.showWindow = false;
BigNet = train(bignet,bigX,Ycomp','useGPU',g);
y = BigNet(bigXval);

testnet1 = feedforwardnet([h,h,h,h,h,h,h,h,h,h],'trainscg');
testnet1.trainParam.showWindow = false;
testnet1 = train(testnet1,Xcomp',Ycomp','useGPU',g);
y1 = testnet1(Xcomp2');
testnet2 = feedforwardnet([h,h,h],'trainscg');
testnet2.trainParam.showWindow = false;
testnet2 = train(testnet2,Xcomp',Ycomp','useGPU',g);
y2 = testnet2(Xcomp2');
testnet3 = feedforwardnet([2*h,2*h,2*h],'trainscg');
testnet3.trainParam.showWindow = false;
testnet3 = train(testnet3,Xcomp',Ycomp','useGPU',g);
y3 = testnet3(Xcomp2');
Perf_Net4(q) = perform(BigNet,Ycomp2',y);
testPerf1(q) = perform(testnet1,Ycomp2',y1);
testPerf2(q) = perform(testnet2,Ycomp2',y2);
testPerf3(q) = perform(testnet3,Ycomp2',y3);
addpoints(per(1),q,mean(Perf_Net4(1:q)))
addpoints(per(2),q,mean(testPerf1(1:q)))
addpoints(per(3),q,mean(testPerf2(1:q)))
addpoints(per(4),q,mean(testPerf3(1:q)))
title("Step: " + (q+1))


drawnow
end
load gong.mat;
sound(y);

data = [Perf_Net4(:),testPerf1(:),testPerf2(:),testPerf3(:)]
[p,tbl,stats] = anova1(data)
[c,m,h,nms] = multcompare(stats,'alpha',0.05,'CType','hsd');

big_pd = fitdist(Perf_Net4,'Normal');
pd1 = fitdist(testPerf1,'Normal');
pd2 = fitdist(testPerf2,'Normal');
pd3 = fitdist(testPerf3,'Normal');

big_ci = paramci(big_pd);
ci1 = paramci(pd1);
ci2 = paramci(pd2);
ci3 = paramci(pd3);

a = max(max(data));
b = min(min(data));
x_values = linspace(0.14,0.175);

y = pdf(big_pd,x_values);
y1 = pdf(pd1,x_values);
y2 = pdf(pd2,x_values);
y3 = pdf(pd3,x_values);

plot(x_values,y,'Color',[0.75 0.5 0.098],'LineWidth',1.2)
hold on
xline(big_ci(1,1),'Color',[0.75 0.5 0.098],'LineWidth',1.2)
xline(big_ci(2,1),'Color',[0.75 0.5 0.098],'LineWidth',1.2)
xline(ci1(1,1),'Color',[0.85 0.1 0.1],'LineWidth',1.2)
xline(ci1(2,1),'Color',[0.85 0.1 0.1],'LineWidth',1.2)
xline(ci2(1,1),'Color',[0.1 0.85 0.1],'LineWidth',1.2)
xline(ci2(2,1),'Color',[0.1 0.85 0.1],'LineWidth',1.2)
xline(ci3(1,1),'Color',[0.098 0.098 0.85],'LineWidth',1.2)
xline(ci3(2,1),'Color',[0.098 0.098 0.85],'LineWidth',1.2)
plot(x_values,y1,'Color',[0.85 0.1 0.1],'LineWidth',1.2)
plot(x_values,y2,'Color',[0.1 0.85 0.1],'LineWidth',1.2)
plot(x_values,y3,'Color',[0.098 0.098 0.85],'LineWidth',1.2)
hold off
% load handel.mat;
% sound(y, 1.45*Fs);