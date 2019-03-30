x = load('ex3Linx.dat');
y = load('ex3Liny.dat');
plot(x,y,'ro','MarkerFaceColor','r');
hold on;
m = length(y);
x = [ones(m,1),x ,x.^2 ,x.^3 ,x.^4 ,x.^5 ];
L = eye(size(x,2),size(x,2));
L(1,1)=0;
lambda = [0;1;10];
x_val = linspace(-1,1,100)';
n = length(x_val);
xx = [ones(n,1),x_val ,x_val.^2 ,x_val.^3 ,x_val.^4 ,x_val.^5 ];
for i = 1:length(lambda)
    theta = (x'*x+lambda(i,1)*L)^-1*x'*y;
    plot(x_val,xx*theta,'-');
    hold on;
end
legend('Training data','\lambda=0','\lambda=1','\lambda=10');