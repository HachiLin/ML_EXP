function strimage(filename,n)
  fidin = fopen(filename); 
  i = 1;
  apres = [];

while ~feof(fidin)
  tline = fgetl(fidin); % ���ļ����� 
  apres{i} = tline;
  i = i+1;
end
  %ѡ������ѡ���ĵ�n��ͼƬ
  a = char(apres(n));
  
  lena = size(a);
  lena = lena(2);
  %xy�洢���ص���������Ӧ�ĻҶ�ֵ
  xy = sscanf(a(4:lena), '%d:%d');

  lenxy = size(xy);
  lenxy = lenxy(1);
  
  
  grid = [];
  grid(784) = 0;  %28*28����0�����ɫ����
  for i=2:2:lenxy  %% ��һ����
      if(xy(i)<=0)
          break
      end
    grid(xy(i-1)) = xy(i) * 100/255; %תΪ����ɫ��ͼ��
  end
  
  %��ʾ��д����ͼ��
  grid1 = reshape(grid,28,28);
  grid1 = fliplr(diag(ones(28,1)))*grid1;
  grid1 = rot90(grid1,3);
  image(grid1);
  hold on;
end