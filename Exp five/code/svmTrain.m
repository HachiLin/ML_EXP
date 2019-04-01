%----------------训练样本函数---------------
function svm = svmTrain(X,Y,kertype,gamma,C)    
    %二次规划问题，使用quadprog，详细help quadprog
    n = length(Y);  
    H = (Y*Y').*kernel(X,X,kertype,gamma);    
    f = -ones(n,1); 
    A = [];
    b = [];
    Aeq = Y'; 
    beq = 0;
    lb = zeros(n,1); 
    ub = C*ones(n,1);
    a = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    epsilon = 3e-5; %阈值可以根据自身需求选择
    %找出支持向量
    svm_index = find(abs(a)> epsilon);
    svm.sva = a(svm_index);
    svm.Xsv = X(svm_index,:);
    svm.Ysv = Y(svm_index);
    svm.svnum = length(svm_index);
    svm.a = a;
end