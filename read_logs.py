# readlines.py
f = open("./lra-news1.out", 'r')
lines = f.readlines()
qk,soft,av = [], [], []
for line in lines:
    if line.startswith("SDDMM"):
        qk.append(float(line.split(": ")[1]))
        #print(float(line.split(": ")[1]))
    elif line.startswith("softmax"):
        soft.append(float(line.split(": ")[1]))
    elif line.startswith("SPMM"):
        av.append(float(line.split(": ")[1]))
f.close()

print(sum(qk)/len(qk))
print(sum(soft)/len(soft))
print(sum(av)/len(av))