d="../LOLTESTOUT/SCI/difficult"
list=dir(d+"/*.png")
l=[]
for i=1:length(list)
    a=imread(d+"/"+list(i).name)
    l(end+1)=niqe(a)
end
l=l'
csvwrite(d+"/results.csv",l)