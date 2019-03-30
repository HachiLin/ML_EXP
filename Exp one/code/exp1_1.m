x = load('ex1_1x.dat');
y = load('ex1_1y.dat');

figure % open a new figure window
plot (x , y , ' o ' );
ylabel ( ' Height in meters ' );
xlabel ( 'Age in years ' );

m = length(y) ; % store the number of training examples
x = [ones(m,1),x] ; % Add a column of ones to x
alpha = 0.07; %learning rate

theta0(1,1) = 0;
theta1(1,1) = 0;
maxIter = 1500;  %max iteration
tol = 1e-8;

for i = 1:maxIter
    theta0(i+1,1) = theta0(i,1) - alpha*(1/m)*sum((theta0(i,1).*x(:,1)+theta1(i,1).*x(:,2)- y).*x(:,1));
    theta1(i+1,1) = theta1(i,1) - alpha*(1/m)*sum((theta0(i,1).*x(:,1)+theta1(i,1).*x(:,2)- y).*x(:,2));
    theta_before = [theta0(i,1);theta1(i,1)];
    theta_now = [theta0(i+1,1);theta1(i+1,1)];
    J_before = (0.5/m)*sum(x*theta_before - y);
    J_now = (0.5/m)*sum(x*theta_now - y);
    if abs(J_now - J_before) < tol
        break;
    end
end
hold on;
plot( x(:,2),theta0(i+1,1) + x(:,2)*theta1(i+1,1),'-');
legend( ' Training data ' , ' Linear regression ' );
