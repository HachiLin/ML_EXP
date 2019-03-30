x = load('ex3Logx.dat');
y = load('ex3Logy.dat');
% Find the indices for the 2 classes
pos = find ( y==1 ) ; neg = find ( y == 0 );
xx = x;
degree = 6;
x = map_feature(x(:,1),x(:,2),degree);
[m, n] = size(x);
lambda = [0;1;10];
L = eye(n,n);
L(1,1) = 0;
g=@(z) 1.0 ./ (1+exp(-z)); 
for k=1:length(lambda)
    theta = zeros(n,1);
    for i = 1:15
       %Calculate the hypothesis function
       z = x*theta;
       h = g(z);
       %Calculate the cost function
       J = -(1/m)*sum(y.*log(h)+(1-y).*log(1-h))+(lambda(k,1)/(2*m))*sum(theta(2:end).^2);
       %Calculate Hession matrix
       H = (1/m).*x'*diag(h)*diag(1-h)*x + (lambda(k,1)/m)*L;
       %Calculate gradient
       G = (lambda(k,1)/m).*theta; 
       G(1,1) = 0;
       J_delta = (1/m).*x'*(h-y) + G;
       %Update the value of theta
       theta = theta - H^(-1)*J_delta;
       store(i,k)=J;
       fprintf('J=%f\n',J);
    end
    % Define the ranges of the grid
    u = linspace(-1 , 1.5, 200);
    v = linspace(-1 , 1.5 ,200);
    %Initialize space for the values to be plotted
    b = zeros(length(u),length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            b(i,j) = map_feature(u(i), v(j),degree)*theta;
        end
    end
    plot ( xx ( pos , 1 ) , xx ( pos , 2 ) , '+' );
    hold on;
    plot ( xx ( neg , 1 ) , xx ( neg , 2 ) , ' o ' );
    b = b';
    contour(u, v, b, [0, 0], 'LineWidth', 2);
    legend('y = 1', 'y = 0', 'Decision boundary');
    title(sprintf('\\lambda = %g', lambda(k,1)), 'FontSize', 10);
    figure;
end
 for i=1:length(lambda)
     plot(1:15,store(:,i),'o--','MarkerFaceColor', 'r');
     title(sprintf('\\lambda = %g', lambda(i,1)), 'FontSize', 10);
     legend('J(\theta)');
     if i == length(lambda)
         break;
     end
     figure;
 end