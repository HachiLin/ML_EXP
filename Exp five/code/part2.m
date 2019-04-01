train_data = load('hand_digits_train.dat');
test_data = load('hand_digits_test.dat');
train_len = size(train_data,1); test_len = size(test_data,1); 
train_index = randperm(train_len,3000);
test_index = randperm(test_len,2115);
train_select = zeros(3000,785);
test_select = zeros(2115,785);
for i = 1:3000
    train_select(i,:) = train_data(train_index(i),:);
end
for i = 1:2115
    test_select(i,:) = test_data(test_index(i),:);
end


C = [0.01,0.1,1,10,100];
%无正则项 
%[train_miss_index,test_miss_index] = predict2(train_select,test_select,'linear',0,0);
%有正则项
for i=1:size(C,2)
    predict2(train_select,test_select,'linear',0,C(i));
end
%查看被错误分类的手写字体
%训练错误手写字体
% for i = 1:length(train_miss_index)
%    strimage('train-01-images.svm',i);
%    figure;
% end
% %测试错误手写字体
% for i = 1:length(test_miss_index)
%    strimage('test-01-images.svm',i);
%    if i~=length(test_miss_index)
%        figure;
%    end
% end

