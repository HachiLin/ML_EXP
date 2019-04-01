%----------------训练样本函数---------------
function svm = svmTrain(X,Y,C)
    % Options是用来控制算法的选项参数的向量，optimset无参时，
    % 创建一个选项结构所有字段为默认值的选项
    options = optimset;
    options.largeScale = 'off'; %LargeScale指大规模搜索，off表示在规模搜索模式关闭
    options.Display = 'off'; %表示无输出
    
    %二次规划用来求解问题，使用quadprog
    n = length(Y);  
    %使用线性核   
    H = (Y.*X)*(Y.*X)';
    H=(H+H')/2;
    f = -ones(n,1); %f'为1*n个-1
    A = [];
    b = [];
    Aeq = Y'; 
    beq = 0;
    lb = zeros(n,1); 
    ub = C*ones(n,1);
    a0 = zeros(n,1);  % a0是解的初始近似值
    a = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    epsilon = 1e-9;
    sv_label = find(abs(a)> epsilon);
    svm.sva = a(sv_label);
    svm.Xsv = X(sv_label,:);
    svm.Ysv = Y(sv_label);
    svm.svnum = length(sv_label);
    svm.a = a;
end