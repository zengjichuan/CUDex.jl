using CUDex

inputsize = 16
seqlength = 10
batchsize = 4
hiddensize = 32
numlayers = 1
dropout = 0.5

x = DexArray(rand(seqlength,batchsize,inputsize)) #

y,hy,cy,w,rd,rs = rnn_ft(x,hiddensize=16)
display(y)
