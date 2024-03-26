out="./LOLTEST/baselinealltest/DICM/low"
outlist=dir(out+"/*.png")
l=[]
for i=1:length(outlist)
    a=imread(out+"/"+outlist(i).name)
    l(3,i)=niqe(a)
    l(4,i)=brisque(a)
end
l=l'
csvwrite(out+"/results.csv",l)