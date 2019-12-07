function [img]=generateFigure(imgW,imgH)

   my_img=255*ones(imgH,imgW,3);
   my_img=uint8(my_img);
   my_img(:,1,:)=0;

   my_img((round(imgH/2);),:,:)=0;
   x=0:2*pi/imgW:2*pi;
   y1=sin(x);
   y2=cos(x);
   y3=x.^2;

   x=int32(x*imgW/(2*pi));
   y1=int32(imgH/2-y1*imgH/4);
   y2=int32(imgH/2-y2*imgH/4);
   y3=int32(imgH/2-y3*imgH/4);

i=1;
   while i<=imgW
       if x(i)==0
       end;
       
       if  y1(i)<=imgH
           A(y1(i),x(i),2)=0;
           A(y1(i),x(i),3)=0;
       end;
       if  y2(i)<=imgH
           my_img(y2(i),x(i),1)=0;
           my_img(y2(i),x(i),3)=0;
       end;
       if  y3(i)>0 && y3(i)<=imgH
           my_img(y3(i),x(i),1)=0;
           my_img(y3(i),x(i),2)=0;
       end;
   i=i+1;
   end;
   imshow(my_img);
end
