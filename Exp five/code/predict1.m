function test = predict1(train_data_name,test_data_name,kertype,gamma,C)
    %(1)-------------------training data ready-------------------
    train_data = load(train_data_name);
    n = size(train_data,2); %data column
    train_x = train_data(:,1:n-1);
    train_y = train_data(:,n);
    %find the position of positive label and negtive label
    pos = find ( train_y == 1 ); 
    neg = find ( train_y == -1 );
    figure('Position',[400 400 1000 400]);
    subplot(1,2,1);
    plot(train_x(pos,1),train_x(pos,2),'k+');
    hold on;
    plot(train_x(neg,1),train_x(neg,2),'bs');
    hold on;

    %(2)-----------------decision boundary-------------------
    train_svm = svmTrain(train_x,train_y,kertype,gamma,C);
    %plot the support vector
    plot(train_svm.Xsv(:,1),train_svm.Xsv(:,2),'ro');
    train_a = train_svm.a;
    train_w = [sum(train_a.*train_y.*train_x(:,1));sum(train_a.*train_y.*train_x(:,2))];
    train_b = sum(train_svm.Ysv-train_svm.Xsv*train_w)/size(train_svm.Xsv,1);
    train_x_axis = 0:1:200;
    plot(train_x_axis,-train_b-train_w(1,1)*train_x_axis/train_w(2,1),'-');
    legend('1','-1','suport vector','decision boundary');
    title('training data')
    hold on;

    %(3)-------------------testing data ready----------------------
    test_data = load(test_data_name);
    m = size(test_data,2); %data column
    test_x = test_data(:,1:m-1);
    test_y = test_data(:,m);
    %find the test data positive label and negtive label
    test_label = sign(test_x*train_w + train_b);
    subplot(1,2,2);
    test_pos = find ( test_y == 1 ); 
    test_neg = find ( test_y == -1 );
    plot(test_x(test_pos,1),test_x(test_pos,2),'k+');
    hold on;
    plot(test_x(test_neg,1),test_x(test_neg,2),'bs');
    hold on;

    %(4)------------------decision boundary -----------------------
    test_x_axis = 0:1:200;
    plot(test_x_axis,-train_b-train_w(1,1)*test_x_axis/train_w(2,1),'-');
    legend('1','-1','decision boundary');
    title('testing data');
    %print the detail
    fprintf('--------------------------------------------\n');
    fprintf('training_data: %s\n',train_data_name);
    fprintf('testing_data: %s\n',test_data_name);
    fprintf('C = %d\n',C);
    fprintf('number of test data label: %d\n',size(test_data,1));
    fprintf('predict corret number of test data label: %d\n',length(find(test_label==test_y)));
    fprintf('Success rate: %.4f\n',length(find(test_label==test_y))/size(test_data,1));
    fprintf('--------------------------------------------\n');
end