function data = divdata(index,X,n)
    len = size(X,1);
    if index == 1
        data = [[X(1:n,:),ones(n,1)];[X(n+1:len,:),-ones(len-n,1)]];
    elseif index == 80
        data = [[X(1:len-n,:),-ones(len-n,1)];[X(len-n+1:len,:),ones(n,1)]];
    else
        data = [[X(1:n*(index-1),:),-ones(n*(index-1),1)];[X(n*index-n+1:n*index,:),ones(n,1)];[X(n*index+1:len,:),-ones(len-n*index,1)]];
    end
end