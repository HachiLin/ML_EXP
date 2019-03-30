x = load('ex2x.dat');
y = load('ex2y.dat');
m = length(y);
mu = mean(x);
xx = x;
sigma = std(x);
x = (x - mean(x))./std(x);
x = [ones(m,1),x] ;
n = size(x,2);
% find返回满足指定条件的行的索引
pos = find ( y == 1 ) ; neg = find ( y == 0 ) ;
plot ( xx ( pos , 1 ) , xx ( pos , 2 ) , '+' ) ; 
hold on
plot ( xx ( neg , 1 ) , xx ( neg , 2 ) , ' o ' )
xlabel('exam1 value');
ylabel('exam2 value');
MaxIter = 1500;
theta = zeros(size(x(1,:)))';
e = 1e-6;
H = zeros(n,n);
g = @(x) 1./(1+exp(-x));
for i = 1:MaxIter
    z = x * theta;
    h = g(z); %logistic model
    L_theta(i,1) = -(1/m)*sum(y.*log(h)+(1-y).*log(1-h)); %log likelihood function
    delta_L = (1/m)*x'*(h-y); %calculate gradient   
    %calculate Hessian matrix
    H = (1/m).*x' * diag(h) * diag(1-h) * x;
    %update L and theta
    if (i > 1) && (abs(L_theta(i,1) - L_theta(i-1,1)) <= e )
        break;
    end
    theta = theta - H^(-1)*delta_L;
    store(i,:) = [theta',L_theta(i,1)];
end
x_axis = x(:,2)*sigma(1) + mu(1);
y_axis = (-theta(1,1).*x(:,1) - theta(2,1).*x(:,2))/theta(3,1);
y_axis = y_axis*sigma(2) + mu(2);
plot(x_axis, y_axis,'-');
figure
plot(1:i-1,store,'-');
legend('\theta_0','\theta_1','\theta_2','L{(\theta)}');
xlabel('iter value');
ylabel('value');

