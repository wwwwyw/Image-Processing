im=imread('1.jpg');   %读取图片
 
%im=rgb2gray(im);   %如果是rgb图片则转为灰度图
 
[h,w]=size(im);         %获取图片高(h)、宽(w)
 
%扫描每一个像素,并记录白点（值为1）坐标及个数
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
 
%最小二乘法拟合直线
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
 
%灰度图转rgb彩色图片
imrgb=repmat(im,[1,1,3]);
 
%画线，线宽3，红色
for i=1:1:w    
    y=a*i+b;
    y2=int32(y);        %把y转换成整数
    %把线上的通道1（红色）置为255
    imrgb(y2-1,i,1)=255;
    imrgb(y2,i,1)=255;
    imrgb(y2+1,i,1)=255;
    
    %把线上的其他通道（绿色和蓝色）置为0
    imrgb(y2-1,i,2)=0;
    imrgb(y2,i,2)=0;
    imrgb(y2+1,i,2)=0;
    imrgb(y2-1,i,3)=0;
    imrgb(y2,i,3)=0;
    imrgb(y2+1,i,3)=0;
end
 
%显示图片
imshow(imrgb);
