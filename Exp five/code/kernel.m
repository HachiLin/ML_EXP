%---------------�˺���---------------
function K = kernel(X,Y,type,gamma)
    switch type
    case 'linear'   %���Ժ�
        K = X*Y';
    case 'rbf'      %��˹��
        m = size(X,1);
        K = zeros(m,m);
        for i = 1:m
            for j = 1:m
                K(i,j) = exp(-gamma*norm(X(i,:)-Y(j,:))^2);
            end
        end
    end
end