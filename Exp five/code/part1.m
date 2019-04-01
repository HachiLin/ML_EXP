kertype = 'linear';
gamma = 0;  %C = 1;
C = [0.01,0.1,1,10,100];
%predict1('training_1.txt','test_1.txt',kertype,gamma,C);
%predict1('training_2.txt','test_2.txt',kertype,gamma,C);
for i=1:size(C,2)
    predict1('training_1.txt','test_1.txt',kertype,gamma,C(i));
end
for i=1:size(C,2)
    predict1('training_2.txt','test_2.txt',kertype,gamma,C(i));
end