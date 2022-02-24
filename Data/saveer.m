Ytrain = table(dCa_dt,Ca,dCb_dt, Cb,dCc_dt, Cc,dTdt,T,Rrxn,dVdt,V,Qout, 'VariableNames',{'dCa_dt','Ca','dCb_dt','Cb','dCc_dt','Cc','dTdt','T','Rrxn','dVdt','V','Qout'});
Ytrain(1,:) = []
writetable(Ytrain, 'Y_pretrain.csv')
Xtrain = table(Qa,Qb,Ta,Tb,Ca0,Cb0,Ca, Cb, Cc,T,V,Qout, 'VariableNames',{'Qa','Qb','Ta','Tb','Ca0','Cb0','Ca','Cb','Cc','T','V','Qout'});
Xtrain(5001,:) = []
writetable(Xtrain, 'X_pretrain.csv')