function [test_miss,train_miss] = predict2(train_data_name,test_data_name,kertype,gamma,C)
    %(1)-------------------training data ready-------------------
    train_data = train_data_name;
    n = size(train_data,2);
    train_x = train_data(:,1:n-1);
    train_y = train_data(:,n);

    %(2)-----------------training model-------------------
    %二次规划用来求解问题，使用quadprog
    n = length(train_y);  
    H = (train_y*train_y').*kernel(train_x,train_x,kertype,gamma);   
    f = -ones(n,1); %f'为1*n个-1
    A = [];
    b = [];
    Aeq = train_y'; 
    beq = 0;
    lb = zeros(n,1);
    if C == 0  %无正则项
        ub = [];
    else       %有正则项
        ub = C.*ones(n,1);
    end
    train_a = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    epsilon = 2e-7; 
    %找出支持向量
    sv_index = find(abs(train_a)> epsilon);
    Xsv = train_x(sv_index,:);
    Ysv = train_y(sv_index);
    svnum = length(sv_index);
    train_w(1:784,1) = sum(train_a.*train_y.*train_x(:,1:784));
    train_b = sum(Ysv-Xsv*train_w)/svnum;
    train_label = sign(train_x*train_w+train_b);
    train_miss = find(train_label~=train_y);

    %(3)-------------------testing data ready----------------------
    test_data = test_data_name;
    m = size(test_data,2);
    test_x = test_data(:,1:m-1);
    test_y = test_data(:,m);
    test_label = sign(test_x*train_w+train_b);
    test_miss = find(test_label~=test_y);

    %(4)------------------detail -----------------------;
    %print the detail
    fprintf('--------------------------------------------\n');
    fprintf('C = %d\n',C);
    fprintf('number of test data label: %d\n',size(test_data,1));
    fprintf('number of train data label: %d\n',size(train_data,1));
    fprintf('predict corret number of test data label: %d\n',length(find(test_label==test_y)));
    fprintf('predict corret number of train data label: %d\n',length(find(train_label==train_y)));
    fprintf('Success rate of test data: %.4f\n',length(find(test_label==test_y))/size(test_data,1));
    fprintf('Success rate of train data: %.4f\n',length(find(train_label==train_y))/size(train_data,1));
    fprintf('--------------------------------------------\n');
end