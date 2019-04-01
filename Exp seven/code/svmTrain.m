%----------------ѵ����������---------------
function svm = svmTrain(X,Y,C)
    % Options�����������㷨��ѡ�������������optimset�޲�ʱ��
    % ����һ��ѡ��ṹ�����ֶ�ΪĬ��ֵ��ѡ��
    options = optimset;
    options.largeScale = 'off'; %LargeScaleָ���ģ������off��ʾ�ڹ�ģ����ģʽ�ر�
    options.Display = 'off'; %��ʾ�����
    
    %���ι滮����������⣬ʹ��quadprog
    n = length(Y);  
    %ʹ�����Ժ�   
    H = (Y.*X)*(Y.*X)';
    H=(H+H')/2;
    f = -ones(n,1); %f'Ϊ1*n��-1
    A = [];
    b = [];
    Aeq = Y'; 
    beq = 0;
    lb = zeros(n,1); 
    ub = C*ones(n,1);
    a0 = zeros(n,1);  % a0�ǽ�ĳ�ʼ����ֵ
    a = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    epsilon = 1e-9;
    sv_label = find(abs(a)> epsilon);
    svm.sva = a(sv_label);
    svm.Xsv = X(sv_label,:);
    svm.Ysv = Y(sv_label);
    svm.svnum = length(sv_label);
    svm.a = a;
end