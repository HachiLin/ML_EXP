function test = predict3(train_name,type,gamma,C)
 %-----------------------training data ready------------------------
    train = train_name;
    [m,n] = size(train);
    train_x = train(:,1:n-1);
    train_y = train(:,n);
    pos = find(train_y == 1);
    neg = find(train_y == -1);
    plot(train_x(pos,1),train_x(pos,2),'k+');
    hold on;
    plot(train_x(neg,1),train_x(neg,2),'bs');
    hold on;
    
    %-----------------------training model--------------------------
    %二次规划用来求解问题，使用quadprog
    K = kernel(train_x,train_x,type,gamma);
    H = (train_y*train_y').*K;    
    f = -ones(m,1); 
    A = [];
    b = [];
    Aeq = train_y'; 
    beq = 0;
    lb = zeros(m,1); 
    if C == 0
        ub = [];
    else
        ub = C*ones(m,1);
    end
    a = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    epsilon = 1e-5;
    %查找支持向量
    sv_index = find(abs(a)> epsilon);
    Xsv = train_x(sv_index,:);
    Ysv = train_y(sv_index);
    svnum = length(sv_index);

    %make classfication predictions over a grid of values
    xplot = linspace(min(train_x(:,1)),max(train_x(:,1)),100)';
    yplot = linspace(min(train_x(:,2)),max(train_x(:,2)),100)';
    [X,Y] = meshgrid(xplot,yplot);
    vals = zeros(size(X));

    %calculate decision value
    train_a = a;
    sum_b = 0;
    for k = 1:svnum
        sum = 0;
        for i = 1:m
            sum = sum + train_a(i,1)*train_y(i,1)*K(i,k);
        end 
        sum_b = sum_b + Ysv(k) - sum;
    end
    train_b = sum_b/svnum;
    for i = 1:100
        for j = 1:100
            x_y = [X(i,j),Y(i,j)];
            sum = 0;
            for k = 1:m
                sum = sum + train_a(k,1)*train_y(k,1)*exp(-gamma*norm(train_x(k,:)-x_y)^2);
            end
            vals(i,j) = sum + train_b;
        end 
    end

    %plot the SVM boundary
    colormap bone;
    contour(X,Y,vals,[0 0],'LineWidth',2);
    title(['\gamma = ',num2str(gamma)]);
end