function [] = houghman()
clc  
clear  
%����������׼ֱ������ֱ�߼��%  
% x1=sort(100.*rand(1,100));  
% y1=-4*x1+2;  
% data1=[x1;y1];  
% x2=sort(10+100.*rand(1,100));  
% y2=x2+4;  
% data2=[x2;y2];  
% x3=sort(100.*rand(1,100));  
% y3=14*x3+6;  
% data3=[x3;y3];  
% x4=sort(100.*rand(1,100));  
% y4=-7*x4+8;  
% data4=[x4;y4];  
% x5=sort(100.*rand(1,100));  
% y5=6*x5+10;  
% data5=[x5;y5];  
% data=[data1,data2,data3,data4,data5];%�����㼯  
  
%��������������ֱ������ֱ�����%  
%�������ݼ���ѡһʹ��  
x1=sort(100.*rand(1,100));  
y1=x1+2+2.*rand(1,100);  
data1=[x1;y1];  
x2=sort(100.*rand(1,100));  
y2=-4*x2+4+4.*randn(1,100);  
data2=[x2;y2];  
x3=sort(100.*rand(1,100));  
y3=16*x3+6+6.*rand(1,100);  
data3=[x3;y3];  
x4=sort(100.*rand(1,100));  
y4=-7*x4+8+8.*rand(1,100);  
data4=[x4;y4];  
x5=sort(100.*rand(1,100));  
y5=6*x5+10+10.*randn(1,100);  
data5=[x5;y5];  
data=[data1,data2,data3,data4,data5];%�����㼯  
[m,n]=size(data);%ͳ�Ƶ���  
%��������ռ�  
n_max=300;%����ռ���������ֵ  
h=zeros(315,2*n_max);  
theta_i=1;  
sigma=70;%���������ֵ  
i=0;  
%ֱ�߹�ʽ�Ƶ�  
%y=sin(theta)/cos(theta)*x+b  
%->p=b*cos(theta)=-sin(theta)*x+cos(theta)*y  
for theta=0:0.01:3.14  
    p=[-sin(theta),cos(theta)];  
    d=p*data;  
    for i=1:n  
   %���ڻ���ռ���d�Ƚϴ󣬶�dֵ����������  
    h(theta_i,round(d(i)/10+n_max))=h(theta_i,round(d(i)/10+n_max))+1;  
    end  
    theta_i=theta_i+1;  
end  
[theta_x,p]=find(h>sigma);%����ͶƱ������sigma��λ��  
l_number=size(theta_x);%����ֱ������  
r=(p-n_max)*10;%����ԭ�ؾ���R  
theta_x=0.01*theta_x;%��theta��ԭ  
figure('color','w');  
plot(data(1,:),data(2,:),'*');  
hold on  
x_line=0:20:100;  
  
for i=1:l_number  
    if(abs(cos(theta_x(i)))<0.01)%б�ʲ����ڵ����  
        x=r(i);y=1:100;  
        plot(x,y,'r');  
    else  
        y=tan(theta_x(i))*x_line+r(i)/cos(theta_x(i));%�����������  
        plot(x_line,y,'r');  
    end  
end  
hold off  
figure('color','w');  
imshow(uint8(10*h));%չʾ����ռ��� 
