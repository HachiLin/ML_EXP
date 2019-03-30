function [theta,J] = func(alpha)
    x = load('ex1_2x.dat');
    y = load('ex1_2y.dat');
    m = length(y); 
    x = [ones(m,1),x]; 
    
    %数据标准化，减少迭代次数，加快梯度下降速度
    sigma = std(x);
    mu = mean(x);
    x(:,2) = (x(:,2) - mu(2))./sigma(2);
    x(:,3) = (x(:,3) - mu(3))./sigma(3);

    theta = zeros(size(x(1,:)))'; %初始化theta 
    J = zeros(50,1); %初始代价矩阵
    for i = 1:50
        h = x * theta; %拟合函数
        E = h - y;
        J(i,1) = (0.5/m)*(E'*E);
        theta = theta - (alpha/m)*x'*E;
    end
end