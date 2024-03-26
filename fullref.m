out="./LOLTEST/baselinealltest/mit/gen/1000"
high="./LOLTEST/baselinealltest/mit/high"
outlist=dir(out+"/*.png")
highlist=dir(high+"/*.png")
l=[]
for i=1:length(outlist)
    a=imread(out+"/"+outlist(i).name)
    b=imread(high+"/"+highlist(i).name)
    l(1,i)=psnr(a,b)
    l(2,i)=ssim(a,b)
    l(3,i)=niqe(a)
    l(4,i)=brisque(a)
end
l=l'
csvwrite(out+"/results.csv",l)