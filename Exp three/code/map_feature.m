function s  = map_feature(x,y,n)
for k=1:size(x,1)
    for i =1:n
        if(i == 1)
            s(k,1) = 1;
            s(k,2) = x(k,1);
            s(k,3) = y(k,1);
            m = 3;
        else
            for j=0:i
                m = m+1;
                s(k,m)=x(k,1)^(i-j)*y(k,1)^j;
            end
        end
    end
end

