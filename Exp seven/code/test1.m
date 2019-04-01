%从人脸照片中随选择7张作为训练集
train_data = zeros(40*7,112*92);
test_data = zeros(40*3,112*92);
for i = 1:40
   file_name = ['s',num2str(i)];
   %随机选取同一文件夹下的7个图片
   index = randperm(10);
   for j = 1:7
       %文件名拼接
       name = ['att_faces\',file_name,'\',num2str(index(j)),'.pgm'];
       image = double(imread(name));
       %将矩阵展开为一行
       train_data(7*(i-1)+j,:) = reshape(image,1,112*92);
   end
   for j = 1:3
       %文件名拼接
       name = ['att_faces\',file_name,'\',num2str(index(7+j)),'.pgm'];
       image = double(imread(name));
       test_data(3*(i-1)+j,:) = reshape(image,1,112*92);
   end
end
%================PCA算法开始================
%(一)Step one: 中心化数据
numData = [train_data;test_data];
X = numData - repmat(mean(numData),40*10,1);

%(二)Step two: 求中心化数据的协方差矩阵
C = X'*X/size(X,1);

%(三)Step three: 计算特征向量和特征值
[V,D] = eig(C);

%(四)Step four: 将特征值和特征向量按降序排序
[dummy,order] = sort(diag(-D));
d=diag(D);      %将特征值取出，构成一个列向量
E=V(:,order);   %将特征向量按照特征值大小进行降序排列
newd=d(order);  %将特征值构成的列向量按降序排列

%(五)Step five: 取前k行组成矩阵P,构成变换矩阵
%选取5个特征贡献率值来当作测试对象
Crate = [0.75 0.80 0.85 0.90 0.95];
sumd = sum(newd);
for ii = 1:length(Crate)
    k = 0;
    for j = 1:length(newd)
        i = sum(newd(1:j,1),1)/sumd; %计算贡献率=前k个特征值之和/总特征值之和 
        if i > Crate(ii) %当贡献率大于95%时循环结束,并记下取多少个特征值
            k = j;
            break;
        end
    end
    P = E(:,1:k);
    newX = X*P;
    %================SVM 实现多分类================
    %================一对多SVM分类器===============
    %(一)Step one: 重新分配数据
    train_X = newX(1:280,:);
    test_X = newX(281:400,:);
    num = 40;
    data1 = cell(num,1);
    data2 = cell(num,1);
    for i = 1:num
       data1{i,1} = divdata(i,train_X,7);
       data2{i,1} = divdata(i,test_X,3);
    end

    %(二)Step two: 训练数据
    a = zeros(num,size(train_X,1));
    w = zeros(num,size(train_X,2));
    b = zeros(num,1);
    for i = 1:num
       svm = svmTrain(data1{i,1}(:,1:k),data1{i,1}(:,k+1),1);
       a(i,:) =  svm.a';
       for j = 1:size(train_X,2)
          w(i,j) = sum(a(i,:)'.*data1{i,1}(:,k+1).*data1{i,1}(:,j));
       end
       b(i,1) = sum(svm.Ysv-svm.Xsv*w(i,:)')/svm.svnum;
    end

    %(三)Step three: 测试数据
    labels_num = zeros(size(test_X,1),num);
    for i = 1:num
       labels_num(:,i) = (test_X*w(i,:)')'+b(i,1); 
    end

    %(四)Step four: 得出分类结果
    labels = zeros(size(test_X,1),1);
    for i = 1:size(test_X,1)
        [~,index] = max(labels_num(i,:));
        labels(i,1) = index;
    end

    %(五)Step five： 分析分类结果
    compare = zeros(size(test_X,1),1);
    for i = 1:num
       compare((i-1)*3+1:(i-1)*3+3,1) = i; 
    end
    success_num = length(find(compare==labels));
    success_rate = success_num/size(test_X,1);
    %打印分类正确率情况
    fprintf('==========one-to-all==========\n');
    fprintf(' k = %d\n',k);
    fprintf(' Crate:%f\n test_data_num: %d\n',Crate(ii),size(test_X,1));
    fprintf(' success_num: %f\n success_rate: %f\n',success_num,success_rate);
end
