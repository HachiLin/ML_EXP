x = load('ex2x.dat');
y = load('ex2y.dat');
m = length(y);
xx = x;
mu = mean(x);
sigma = std(x);
x = (x - mean(x))./std(x); %数据标准化
x = [ones(m,1),x] ; xx = [ones(m,1),xx] ;
% find返回满足指定条件的行的索引
pos = find ( y == 1 ) ; neg = find ( y == 0 );
plot (xx( pos , 2 ),xx( pos , 3 ),'+'); 
hold on;
plot (xx( neg , 2 ),xx( neg , 3 ),'o');
xlabel('exam1 value');
ylabel('exam2 value');

MaxIter = 1500;
theta = zeros(size(x(1,:)))'; %初始化theta
e = 1e-6;
alpha = 0.08;
g = @(z) 1./(1+exp(-z)); %定义sigmoid函数
for i = 1:MaxIter
    z = x * theta;
    h = g(z); %逻辑回归模型
    L_theta(i,1) = -(1/m)*sum(y.*log(h)+(1-y).*log(1-h)); %极大对数似然函数
    delta_L = (1/m)*x'*(h-y); %计算梯度   
    %更新 L 和 theta
    if (i > 1) && (abs(L_theta(i,1) - L_theta(i-1,1)) <= e)
        break;
    end
    theta = theta - alpha*delta_L;
    store(i,:) = [theta',L_theta(i,1)];
end

%画出决策边界，因为数据是标准化后的，因此需要还原回去
x_axis = x(:,2)*sigma(1) + mu(1);
y_axis = (-theta(1,1).*x(:,1) - theta(2,1).*x(:,2))/theta(3,1);
y_axis = y_axis*sigma(2) + mu(2);
plot(x_axis, y_axis,'-');
figure;
plot(1:i-1,store,'-');
legend('\theta_0','\theta_1','\theta_2','L{(\theta)}');
xlabel('iter value');
ylabel('value');

