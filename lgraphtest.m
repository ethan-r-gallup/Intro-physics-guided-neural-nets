%% 
% Prepare Data

Xtrain = readtable('Xtrain_10000.txt')
[NX, CX, SX] = normalize(Xtrain)
Xtrain = NX{:,:};
Ytrain = readtable('Ytrain_10000.txt')
[NY, CY, SY] = normalize(Ytrain, 'range');
Ytrain = NY{:,:};


%% *Define Layer*s

layers1 = [
    sequenceInputLayer(10, "Name", "C")
    fullyConnectedLayer(20, "Name", "connected")
    reluLayer("Name", "Rey")
    reluLayer("Name", "Lou")
    fullyConnectedLayer(2, "Name", "ToValues")
    tanhLayer("Name", "Sid")
    ]; % ->Rxn, Qout
%layers1.InputNames = "C"
lgraph1 = layerGraph(layers1);
layers2 = [
    sequenceInputLayer(7, "Name", "input2") 
    fullyConnectedLayer(3, "Name", "connected")
    fullyConnectedLayer(1, "Name", "ToValues")
    tanhLayer("Name", "Sid")
    ]; % -> dVdt
lgraph2 = layerGraph(layers2);
layers3 = [
    sequenceInputLayer(15, "Name", "input3")
    fullyConnectedLayer(30, "Name", "connected")
    reluLayer("Name", "Rey")
    reluLayer("Name", "Lou")
    fullyConnectedLayer(4, "Name", "ToValues")
    tanhLayer("Name", "Sid")
    ]; % -> dCa, dCb, dCc
lgraph3 = layerGraph(layers3);

layers4 = [
    sequenceInputLayer(16, "Name", "input4")
    fullyConnectedLayer(16, "Name", "connected")
    fullyConnectedLayer(5, "Name", "connected2")
    tanhLayer("Name", "Sid")
    ]; % -> A, B, C, Temp, Vol
lgraph4 = layerGraph(layers4);

subplot(1,4,1)
plot(lgraph1)
subplot(1,4,2)
plot(lgraph2)
subplot(1,4,3)
plot(lgraph3)
subplot(1,4,4)
plot(lgraph4)

%% 
% create neural networks

net1 = dlnetwork(lgraph1);
net2 = dlnetwork(lgraph2);
net3 = dlnetwork(lgraph3);
net4 = dlnetwork(lgraph4);
%% 
% Define Parameters

fig1 = figure()
set(gcf,'Visible','on')
ax1 = axes('Parent', fig1);
semilogy(nan, nan);
lineLossTrain(1) = animatedline('Color',[0.75 0.5 0.098], 'Parent', ax1, 'DisplayName', 'net 1');
lineLossTrain(2) = animatedline('Color',[0.85 0.1 0.1], 'Parent', ax1, 'DisplayName', 'net 2');
lineLossTrain(3) = animatedline('Color',[0.1 0.85 0.1], 'Parent', ax1, 'DisplayName', 'net 3');
lineLossTrain(4) = animatedline('Color',[0.098 0.098 0.85], 'Parent', ax1, 'DisplayName', 'net 4');
ylim([0 inf])
yline(.1);yline(.01);

legend(lineLossTrain(1:4))
xlabel("Iteration")
ylabel("Loss")

fig2 = figure()
ax2 = axes('Parent', fig2);
set(gcf,'Visible','on')
%semilogy(nan, nan);
err(1) = animatedline('Color',[0.75 0.5 0.098], 'Parent', ax2, 'DisplayName', 'net 1');
err(2) = animatedline('Color',[0.85 0.1 0.1], 'Parent', ax2, 'DisplayName', 'net 2');
err(3) = animatedline('Color',[0.1 0.85 0.1], 'Parent', ax2, 'DisplayName', 'net 3');
err(4) = animatedline('Color',[0.098 0.098 0.85], 'Parent', ax2, 'DisplayName', 'net 4');
ylim([0 inf])
yline(.05);

legend(err(1:4))
xlabel("Iteration")
ylabel("Diff")
velocity1 = [];velocity2 = [];velocity3 = [];velocity4 = [];
%% 
% Training loop

numEpochs = 10;
initialLearnRate = .0001;
decay = 0.001;
momentum = 0.9;
iteration = 0;
start = tic;
batch = 100;

for epoch = 1:numEpochs

    % Loop over mini-batches.
    for i = 1:batch:10000-batch
        iteration = iteration + 1;
        
        % net1 -> Rxn, Qout
        X1 = [Xtrain(i+1:i+batch,:), Ytrain(i:i+batch-1,[2, 4, 8, 11])];
        Y1 = Ytrain(i+1:i+batch,[9, 12]);
        dlX1 = dlarray(X1,"TC");
        dlY1 = dlarray(Y1,"TC");
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients1,state1,loss1, Pred1, er1] = dlfeval(@modelGradients,net1,dlX1,dlY1);
        net1.State = state1;
        
        % net2 -> dVdt
        X2 = [Xtrain(i+1:i+batch,:), Ytrain(i+1:i+batch,12)];
        Y2 = Ytrain(i+1:i+batch,10);
        dlX2 = dlarray(X2,"TC");
        dlY2 = dlarray(Y2,"TC");
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients2,state2,loss2, Pred2, er2] = dlfeval(@modelGradients,net2,dlX2,dlY2);
        net2.State = state2;
        
        % net3 -> dCa, dCb, dCc, dTemp
        X3 = [Xtrain(i+1:i+batch,:), Ytrain(i+1:i+batch,[9, 10, 12]), Ytrain(i:i+batch-1,[2, 4, 6, 8, 10, 11])];
        Y3 = Ytrain(i+1:i+batch,[1, 3, 5, 10]);
        dlX3 = dlarray(X3,"TC");
        dlY3 = dlarray(Y3,"TC");
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients3,state3,loss3, Pred3, er3] = dlfeval(@modelGradients,net3,dlX3,dlY3);
        net3.State = state3;
        
        % net4 -> A, B, C, Temp, Vol
        X4 = [Xtrain(i+1:i+batch,:), Ytrain(i+1:i+batch,[1, 3, 5, 7, 10]), Ytrain(i:i+batch-1,[2, 4, 6, 8, 11])];
        Y4 = Ytrain(i+1:i+batch,[2, 4, 6, 8, 11]);
        dlX4 = dlarray(X4,"TC");
        dlY4 = dlarray(Y4,"TC");
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients4,state4,loss4, Pred4, er4] = dlfeval(@modelGradients,net4,dlX4,dlY4);
        net4.State = state4;
        
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        % Update the network parameters using the SGDM optimizer.
        [net1,velocity1] = sgdmupdate(net1,gradients1,velocity1,learnRate,momentum);
        [net2,velocity2] = sgdmupdate(net2,gradients2,velocity2,learnRate,momentum);
        [net3,velocity3] = sgdmupdate(net3,gradients3,velocity3,learnRate,momentum);
        [net4,velocity4] = sgdmupdate(net4,gradients4,velocity4,learnRate,momentum);
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain(1),iteration,loss1)
        addpoints(lineLossTrain(2),iteration,loss2)
        addpoints(lineLossTrain(3),iteration,loss3)
        addpoints(lineLossTrain(4),iteration,loss4)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
        addpoints(err(1),iteration,er1)
        addpoints(err(2),iteration,er2)
        addpoints(err(3),iteration,er3)
        addpoints(err(4),iteration,er4)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end
%% Phase 2

numEpochs = 10;
initialLearnRate = .005;
decay = 0.001;
momentum = 0.9;
iteration = 0;
start = tic;
batch = 100;
Pred = zeros(size(Ytrain))
Pred(1,:)=Ytrain(1,:)

for epoch = 1:numEpochs

    % Loop over mini-batches.
    for i = 1:batch:10000-batch
        iteration = iteration + 1;
        
        % net1 -> Rxn, Qout, dVdt
        X1 = [Xtrain(i+1:i+batch,:), Pred(i:i+batch-1,[2, 4, 8, 11])];
        Y1 = Ytrain(i+1:i+batch,[9, 10, 12]);
        dlX1 = dlarray(X1,"TC");
        dlY1 = dlarray(Y1,"TC");
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients1,state1,loss1, Pred(i+1:i+batch,[9, 10, 12]), er1] = dlfeval(@multiModelGradients,net1,dlX1,dlY1);
        net1.State = state1;
        
        % net2 -> dVdt
        X2 = [Xtrain(i+1:i+batch,:), Ytrain(i+1:i+batch,12)];
        Y2 = Ytrain(i+1:i+batch,10);
        dlX2 = dlarray(X2,"TC");
        dlY2 = dlarray(Y2,"TC");
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients2,state2,loss2, Pred2, er2] = dlfeval(@modelGradients,net2,dlX2,dlY2);
        net2.State = state2;
        
        % net3 -> dCa, dCb, dCc, dTemp
        X3 = [Xtrain(i+1:i+batch,:), Ytrain(i+1:i+batch,[9, 10, 12]), Ytrain(i:i+batch-1,[2, 4, 6, 8, 10, 11])];
        Y3 = Ytrain(i+1:i+batch,[1, 3, 5, 10]);
        dlX3 = dlarray(X3,"TC");
        dlY3 = dlarray(Y3,"TC");
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients3,state3,loss3, Pred3, er3] = dlfeval(@modelGradients,net3,dlX3,dlY3);
        net3.State = state3;
        
        % net4 -> A, B, C, Temp, Vol
        X4 = [Xtrain(i+1:i+batch,:), Ytrain(i+1:i+batch,[1, 3, 5, 7, 10]), Ytrain(i:i+batch-1,[2, 4, 6, 8, 11])];
        Y4 = Ytrain(i+1:i+batch,[2, 4, 6, 8, 11]);
        dlX4 = dlarray(X4,"TC");
        dlY4 = dlarray(Y4,"TC");
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients4,state4,loss4, Pred4, er4] = dlfeval(@modelGradients,net4,dlX4,dlY4);
        net4.State = state4;
        
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        % Update the network parameters using the SGDM optimizer.
        [net1,velocity1] = sgdmupdate(net1,gradients1,velocity1,learnRate,momentum);
        [net2,velocity2] = sgdmupdate(net2,gradients2,velocity2,learnRate,momentum);
        [net3,velocity3] = sgdmupdate(net3,gradients3,velocity3,learnRate,momentum);
        [net4,velocity4] = sgdmupdate(net4,gradients4,velocity4,learnRate,momentum);
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain(1),iteration,loss1)
        addpoints(lineLossTrain(2),iteration,loss2)
        addpoints(lineLossTrain(3),iteration,loss3)
        addpoints(lineLossTrain(4),iteration,loss4)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
        addpoints(err(1),iteration,er1)
        addpoints(err(2),iteration,er2)
        addpoints(err(3),iteration,er3)
        addpoints(err(4),iteration,er4)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end

%% 
% gradient and loss function

function [gradients,state,loss, dlYPred, er] = modelGradients(dlnet,dlX,Y)

[dlYPred,state] = forward(dlnet,dlX);
diff = mean(abs(Y-dlYPred));
loss = mse(dlYPred, Y);

gradients = dlgradient(loss,dlnet.Learnables);

loss = double(gather(extractdata(loss)));
er = mean(double(gather(extractdata(diff))));
dlYPred = mean(1-double(gather(extractdata(dlYPred))));

end