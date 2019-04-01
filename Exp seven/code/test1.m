%��������Ƭ����ѡ��7����Ϊѵ����
train_data = zeros(40*7,112*92);
test_data = zeros(40*3,112*92);
for i = 1:40
   file_name = ['s',num2str(i)];
   %���ѡȡͬһ�ļ����µ�7��ͼƬ
   index = randperm(10);
   for j = 1:7
       %�ļ���ƴ��
       name = ['att_faces\',file_name,'\',num2str(index(j)),'.pgm'];
       image = double(imread(name));
       %������չ��Ϊһ��
       train_data(7*(i-1)+j,:) = reshape(image,1,112*92);
   end
   for j = 1:3
       %�ļ���ƴ��
       name = ['att_faces\',file_name,'\',num2str(index(7+j)),'.pgm'];
       image = double(imread(name));
       test_data(3*(i-1)+j,:) = reshape(image,1,112*92);
   end
end
%================PCA�㷨��ʼ================
%(һ)Step one: ���Ļ�����
numData = [train_data;test_data];
X = numData - repmat(mean(numData),40*10,1);

%(��)Step two: �����Ļ����ݵ�Э�������
C = X'*X/size(X,1);

%(��)Step three: ������������������ֵ
[V,D] = eig(C);

%(��)Step four: ������ֵ��������������������
[dummy,order] = sort(diag(-D));
d=diag(D);      %������ֵȡ��������һ��������
E=V(:,order);   %������������������ֵ��С���н�������
newd=d(order);  %������ֵ���ɵ�����������������

%(��)Step five: ȡǰk����ɾ���P,���ɱ任����
%ѡȡ5������������ֵ���������Զ���
Crate = [0.75 0.80 0.85 0.90 0.95];
sumd = sum(newd);
for ii = 1:length(Crate)
    k = 0;
    for j = 1:length(newd)
        i = sum(newd(1:j,1),1)/sumd; %���㹱����=ǰk������ֵ֮��/������ֵ֮�� 
        if i > Crate(ii) %�������ʴ���95%ʱѭ������,������ȡ���ٸ�����ֵ
            k = j;
            break;
        end
    end
    P = E(:,1:k);
    newX = X*P;
    %================SVM ʵ�ֶ����================
    %================һ�Զ�SVM������===============
    %(һ)Step one: ���·�������
    train_X = newX(1:280,:);
    test_X = newX(281:400,:);
    num = 40;
    data1 = cell(num,1);
    data2 = cell(num,1);
    for i = 1:num
       data1{i,1} = divdata(i,train_X,7);
       data2{i,1} = divdata(i,test_X,3);
    end

    %(��)Step two: ѵ������
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

    %(��)Step three: ��������
    labels_num = zeros(size(test_X,1),num);
    for i = 1:num
       labels_num(:,i) = (test_X*w(i,:)')'+b(i,1); 
    end

    %(��)Step four: �ó�������
    labels = zeros(size(test_X,1),1);
    for i = 1:size(test_X,1)
        [~,index] = max(labels_num(i,:));
        labels(i,1) = index;
    end

    %(��)Step five�� ����������
    compare = zeros(size(test_X,1),1);
    for i = 1:num
       compare((i-1)*3+1:(i-1)*3+3,1) = i; 
    end
    success_num = length(find(compare==labels));
    success_rate = success_num/size(test_X,1);
    %��ӡ������ȷ�����
    fprintf('==========one-to-all==========\n');
    fprintf(' k = %d\n',k);
    fprintf(' Crate:%f\n test_data_num: %d\n',Crate(ii),size(test_X,1));
    fprintf(' success_num: %f\n success_rate: %f\n',success_num,success_rate);
end
