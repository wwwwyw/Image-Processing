im=imread('1.jpg');   %��ȡͼƬ
 
%im=rgb2gray(im);   %�����rgbͼƬ��תΪ�Ҷ�ͼ
 
[h,w]=size(im);         %��ȡͼƬ��(h)����(w)
 
%ɨ��ÿһ������,����¼�׵㣨ֵΪ1�����꼰����
n=0;
for y0=1:1:h
    for x0=1:1:w
        if(im(y0,x0)==255)
            n=n+1;
            y(n)=y0;
            x(n)=x0;
        end   
    end  
end
 
%��С���˷����ֱ��
A = 0.0;
B = 0.0;
C = 0.0;
D = 0.0;
for i=1:1:n
    A=A+x(i)*x(i);
    B=B+x(i);
    C=C+x(i)*y(i);
    D=D+y(i);
end
 
a = (C*n - B*D) / (A*n - B*B);
b = (A*D - C*B) / (A*n - B*B);
 
%�Ҷ�ͼתrgb��ɫͼƬ
imrgb=repmat(im,[1,1,3]);
 
%���ߣ��߿�3����ɫ
for i=1:1:w    
    y=a*i+b;
    y2=int32(y);        %��yת��������
    %�����ϵ�ͨ��1����ɫ����Ϊ255
    imrgb(y2-1,i,1)=255;
    imrgb(y2,i,1)=255;
    imrgb(y2+1,i,1)=255;
    
    %�����ϵ�����ͨ������ɫ����ɫ����Ϊ0
    imrgb(y2-1,i,2)=0;
    imrgb(y2,i,2)=0;
    imrgb(y2+1,i,2)=0;
    imrgb(y2-1,i,3)=0;
    imrgb(y2,i,3)=0;
    imrgb(y2+1,i,3)=0;
end
 
%��ʾͼƬ
imshow(imrgb);
