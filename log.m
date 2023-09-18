x=0:255;
for i=0:128 
   for j=0:255
       if (2^(8*i/128)-1)<=j
           if (2^(8*(i+1)/128)-1)>j
               y(j+1)=10^(5*(128-i)/128-5);
           end
       end
   end
end

plot(x(1:10),y(1:10))
xlabel('X_{low}')
ylabel('W_{low}')
semilogy(x,y)
xlabel('X_{low}')
ylabel('W_{low}')