function svm = re_hand_digits(filename,n)
  fidin = fopen(filename); 
  i = 1;
  apres = [];

while ~feof(fidin)
  tline = fgetl(fidin); % ���ļ����� 
  apres{i} = tline;
  i = i+1;
end

 grid = zeros(n,784);
 label = zeros(n,1);
 for k = 1:n
    a = char(apres(k));
    lena = size(a,2);
    xy = sscanf(a(4:lena), '%d:%d');
    label(k,1) = sscanf(a(1:3),'%d');
    lenxy = size(xy,1);
    for i=2:2:lenxy  %% ��һ����
      if(xy(i)<=0)
          break
      end
      grid(k,xy(i-1)) = xy(i) * 100/255; %תΪ����ɫ��ͼ��
    end  
 end 
  svm.grid = grid;
  svm.label = label;
end
