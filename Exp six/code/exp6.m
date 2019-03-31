data = double(imread('bird_small.tiff'));
[x,y,z] = size(data);
%聚类质心个数，也就是代替图片的颜色个数
k = 16;
maxiter = 100; %最大的迭代次数

% 初始化样本中心，随机选取16个样本作为聚类中心
center = zeros(k, 3);
for i = 1:k
    center(i,:) = data(floor(rand*x),floor(rand*y),:);
end

%算法过程
for iter = 1:maxiter
    new_center = zeros(size(center));
    num = zeros(1,k);
    
    %确定每个样本的类别
    for i = 1:x
       for j = 1:x
           r = data(i,j,1);
           g = data(i,j,2);
           b = data(i,j,3);
           %计算样本到每个聚类中心的距离
           diff = ones(k,1)*[r, g, b] - center;
           distance = sum(diff.^2, 2); 
           %求距离最小值的下标
           [~,temp] = min(distance);
           new_center(temp,1) = new_center(temp,1) + r;
           new_center(temp,2) = new_center(temp,2) + g;
           new_center(temp,3) = new_center(temp,3) + b;
           num(temp) = num(temp) + 1;
       end
    end

    %更新样本中心
    for i = 1:k
        if (num(i) > 0)
          new_center(i,:) = new_center(i,:)./num(i);
        end
    end
    
    %判断是否收敛
    d = sum(sqrt(sum((new_center - center).^2, 2)));
    if d < 1e-5
        break
    end
    center = new_center;
end
%RGB值取整
center = round(center);

%将上面小图片的聚类中心应用到大图片上
large_image = double(imread('bird_large.tiff'));
large_size = size(large_image,1);
distance = zeros(1,k);
for i = 1:large_size
   for j = 1:large_size
       r = large_image(i,j,1); 
       g = large_image(i,j,2); 
       b = large_image(i,j,3);
       %计算样本到每个聚类中心的距离
       diff = ones(k,1)*[r, g, b] - center;
       distance = sum(diff.^2, 2); 
       %求距离最小值的下标
       [~,temp] = min(distance);
       large_image(i,j,:) = center(temp,:);
   end
end
%Display
imshow(uint8(round(large_image))); 
% Save image
imwrite(uint8(round(large_image)),'bird_kmeans.jpg');
