function strimage(filename,n)
  fidin = fopen(filename); 
  i = 1;
  apres = [];

while ~feof(fidin)
  tline = fgetl(fidin); % 从文件读行 
  apres{i} = tline;
  i = i+1;
end
  %选中我们选定的第n张图片
  a = char(apres(n));
  
  lena = size(a);
  lena = lena(2);
  %xy存储像素的索引和相应的灰度值
  xy = sscanf(a(4:lena), '%d:%d');

  lenxy = size(xy);
  lenxy = lenxy(1);
  
  
  grid = [];
  grid(784) = 0;  %28*28网格，0代表黑色背景
  for i=2:2:lenxy  %% 隔一个数
      if(xy(i)<=0)
          break
      end
    grid(xy(i-1)) = xy(i) * 100/255; %转为有颜色的图像
  end
  
  %显示手写数字图像
  grid1 = reshape(grid,28,28);
  grid1 = fliplr(diag(ones(28,1)))*grid1;
  grid1 = rot90(grid1,3);
  image(grid1);
  hold on;
end