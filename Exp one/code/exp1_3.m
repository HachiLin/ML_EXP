alpha = [0.12,0.15,0.18];
data = load('ex1_2x.dat');
store_theta = zeros(size(alpha,2),size(data,2)+1);
for i = 1:length(alpha)
    [theta,J] = func(alpha(i));
    store_theta(i,:) = theta;
    plot(1:50,J);
    hold on;
end
xlabel('Number of iteration');
ylabel('Cost J');
legend( '\alpha_0 = 0.12' , '\alpha_1 = 0.15' , '\alpha_2 = 0.18');
    