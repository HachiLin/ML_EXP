function [label_pre,sum,sucess_rate] = LogMLE(test,train)
    [train_row,train_col] = size(train);
    test_num = size(test,1);
    %记录每个类对应的对数似然函数值，最大值对应的下标减1即最终我们预测的类
    label_for_MLE = zeros(test_num,5);
    label_pre = zeros(test_num,1); %存储所有我们预测的类
    label_num = max(train(:,end)); %类别总数
    for i = 1:test_num
        for y = 0:label_num
            count_y = length(find(train(:,train_col)==y));
            p_y = count_y/(train_row+label_num);
            log_count_xy = 0;
            for j = 1:train_col-1
                count_xy = length(find(train(:,train_col)==y & train(:,j)==test(i,j)));
                %拉普拉斯平滑
                if j==1
                    p_xy = (count_xy+1)/(count_y+3);
                elseif j==2
                    p_xy = (count_xy+1)/(count_y+5);
                elseif j==3
                    p_xy = (count_xy+1)/(count_y+4);
                elseif j==4
                    p_xy = (count_xy+1)/(count_y+4);
                elseif j==5
                    p_xy = (count_xy+1)/(count_y+3);
                elseif j==6
                    p_xy = (count_xy+1)/(count_y+2);
                elseif j==7
                    p_xy = (count_xy+1)/(count_y+3);
                elseif j==8
                    p_xy = (count_xy+1)/(count_y+3);
                end
                log_count_xy = log_count_xy + log(p_xy);
            end
            label_for_MLE(i,y+1) = log(p_y) + log_count_xy;
        end
        [~,b2] = find(label_for_MLE(i,:)==max(max(label_for_MLE(i,:))));
        label_pre(i,1) = b2-1;
        sum = length(find(label_pre(:,1)==test(:,end)));
        sucess_rate = sum/test_num;
    end
end