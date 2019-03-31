data = double(imread('bird_small.tiff'));
[x,y,z] = size(data);
%�������ĸ�����Ҳ���Ǵ���ͼƬ����ɫ����
k = 16;
maxiter = 100; %���ĵ�������

% ��ʼ���������ģ����ѡȡ16��������Ϊ��������
center = zeros(k, 3);
for i = 1:k
    center(i,:) = data(floor(rand*x),floor(rand*y),:);
end

%�㷨����
for iter = 1:maxiter
    new_center = zeros(size(center));
    num = zeros(1,k);
    
    %ȷ��ÿ�����������
    for i = 1:x
       for j = 1:x
           r = data(i,j,1);
           g = data(i,j,2);
           b = data(i,j,3);
           %����������ÿ���������ĵľ���
           diff = ones(k,1)*[r, g, b] - center;
           distance = sum(diff.^2, 2); 
           %�������Сֵ���±�
           [~,temp] = min(distance);
           new_center(temp,1) = new_center(temp,1) + r;
           new_center(temp,2) = new_center(temp,2) + g;
           new_center(temp,3) = new_center(temp,3) + b;
           num(temp) = num(temp) + 1;
       end
    end

    %������������
    for i = 1:k
        if (num(i) > 0)
          new_center(i,:) = new_center(i,:)./num(i);
        end
    end
    
    %�ж��Ƿ�����
    d = sum(sqrt(sum((new_center - center).^2, 2)));
    if d < 1e-5
        break
    end
    center = new_center;
end
%RGBֵȡ��
center = round(center);

%������СͼƬ�ľ�������Ӧ�õ���ͼƬ��
large_image = double(imread('bird_large.tiff'));
large_size = size(large_image,1);
distance = zeros(1,k);
for i = 1:large_size
   for j = 1:large_size
       r = large_image(i,j,1); 
       g = large_image(i,j,2); 
       b = large_image(i,j,3);
       %����������ÿ���������ĵľ���
       diff = ones(k,1)*[r, g, b] - center;
       distance = sum(diff.^2, 2); 
       %�������Сֵ���±�
       [~,temp] = min(distance);
       large_image(i,j,:) = center(temp,:);
   end
end
%Display
imshow(uint8(round(large_image))); 
% Save image
imwrite(uint8(round(large_image)),'bird_kmeans.jpg');
