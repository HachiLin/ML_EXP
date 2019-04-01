svm1 = re_hand_digits('train-01-images.svm',12665);
svm2 = re_hand_digits('test-01-images.svm',2115);
train_x = svm1.grid; train_y = svm1.label;
test_x = svm2.grid; test_y = svm2.label;
train = [train_x,train_y];
test = [test_x,test_y];

[row,col] = size(test);
fid=fopen('hand_digits_test.dat','wt');                                                        
for i=1:1:row
    for j=1:1:col
        if(j==col)
            fprintf(fid,'%g\n',test(i,j));
        else
            fprintf(fid,'%g\t',test(i,j));
        end
    end
end
fclose(fid);

[row,col] = size(train);
fid=fopen('hand_digits_train.dat','wt');                                                         
for i=1:1:row
    for j=1:1:col
        if(j==col)
            fprintf(fid,'%g\n',train(i,j));
        else
            fprintf(fid,'%g\t',train(i,j));
        end
    end
end
fclose(fid);
