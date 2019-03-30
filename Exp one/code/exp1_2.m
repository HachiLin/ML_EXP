J_vals = zeros (100 , 100) ; % initialize Jvals to
							 % 100*100 matrix of 0's
theta0_vals = linspace (-3 , 3 , 100) ;
theta1_vals = linspace (-1 , 1 , 100) ;
% ����linespace(x1,x2,N)������x1��x2��N�ֱ�Ϊ��ʼֵ����ֵֹ��Ԫ�ظ�����
for i = 1 : length (theta0_vals)
	for j = 1 : length (theta1_vals )
		t = [theta0_vals(i); theta1_vals(j)] ;
		J_vals(i,j) = (0.5/m)*(x*t-y)'*(x*t-y);
	end
end
J_vals = J_vals'; %ת��
figure ;
surf(theta0_vals,theta1_vals,J_vals);
xlabel ('\theta_0 ');ylabel('\theta_1');