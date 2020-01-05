function [] = houghman()
clc  
clear  
%生成五条标准直线用于直线检测%  
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
% data=[data1,data2,data3,data4,data5];%构建点集  
  
%生成五条带噪声直线用于直线拟合%  
%两个数据集二选一使用  
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
data=[data1,data2,data3,data4,data5];%构建点集  
[m,n]=size(data);%统计点数  
%构建霍夫空间  
n_max=300;%霍夫空间的纵轴最大值  
h=zeros(315,2*n_max);  
theta_i=1;  
sigma=70;%设置拟合阈值  
i=0;  
%直线公式推导  
%y=sin(theta)/cos(theta)*x+b  
%->p=b*cos(theta)=-sin(theta)*x+cos(theta)*y  
for theta=0:0.01:3.14  
    p=[-sin(theta),cos(theta)];  
    d=p*data;  
    for i=1:n  
   %由于霍夫空间中d比较大，对d值进行了缩放  
    h(theta_i,round(d(i)/10+n_max))=h(theta_i,round(d(i)/10+n_max))+1;  
    end  
    theta_i=theta_i+1;  
end  
[theta_x,p]=find(h>sigma);%查找投票数大于sigma的位置  
l_number=size(theta_x);%符合直线条数  
r=(p-n_max)*10;%将还原回距离R  
theta_x=0.01*theta_x;%将theta还原  
figure('color','w');  
plot(data(1,:),data(2,:),'*');  
hold on  
x_line=0:20:100;  
  
for i=1:l_number  
    if(abs(cos(theta_x(i)))<0.01)%斜率不存在的情况  
        x=r(i);y=1:100;  
        plot(x,y,'r');  
    else  
        y=tan(theta_x(i))*x_line+r(i)/cos(theta_x(i));%画出拟合曲线  
        plot(x_line,y,'r');  
    end  
end  
hold off  
figure('color','w');  
imshow(uint8(10*h));%展示霍夫空间结果 
