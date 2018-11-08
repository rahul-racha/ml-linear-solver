from builtins import float
f = open('mat.out','r')

result = [[0]*4224 for i in range(4224)]

i=0
for line in f:
    if i>=2:
        #print(i)
        line = line[:-1]
        lineSplit = line.split(":")
        indiceInfo = lineSplit[1].strip().split(")  (")
        #print(indiceInfo)
        for index,ele in enumerate(indiceInfo):
            if index==0:
                ele = ele[1:]
            if index == len(indiceInfo)-1:
                ele = ele[:-1]
            someInfo = ele.split(',')
            infoIndex = int(someInfo[0].strip())
            value = float(someInfo[1].strip()) 
            #print(infoIndex, value)
            result[i-2][infoIndex] = float(str(value))
    i+=1
            
#2d matrix is in result
