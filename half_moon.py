import matplotlib.pyplot as plt
import numpy as np
import math
def dbmoon(N=100, d=1, r=10, w=6):
    N1 = 10*N
    w2 = w/2
    done = True
    data = np.empty(0)
    while done:
        #generate Rectangular data
        tmp_x = 2*(r+w2)*(np.random.random([N1, 1])-0.5)
        tmp_y = (r+w2)*np.random.random([N1, 1])
        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
        tmp_ds = np.sqrt(tmp_x*tmp_x + tmp_y*tmp_y)
        #generate double moon data ---upper
        idx = np.logical_and(tmp_ds > (r-w2), tmp_ds < (r+w2))
        idx = (idx.nonzero())[0]
        if data.shape[0] == 0:
            data = tmp.take(idx, axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
        if data.shape[0] >= N:
           done = False
    print (data)
    db_moon = data[0:N, :]
    print(db_moon)
    #generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    return db_moon

N = 1000
d = 1
r = 10
w = 5
a = 0.5
num_MSE = []
num_step = []
data = dbmoon(N, d, r, w)

x0 = [1 for x in range(1,2001)]
x = np.array([np.reshape(x0, len(x0)), np.reshape(data[0:2*N, 0], len(data)), np.reshape(data[0:2*N, 1], len(data))]).transpose()
m = np.array([10, 10, 0])
b_pre = [1 for y in range(1, 1001)]
b_pos = [-1 for y in range(1, 1001)]
b=b_pre+b_pos
print(b)
def sgn(v):
    if v >= 0:
        return 1
    else:
        return -1
#compute y(n)
def compute_yn(myw, myx):
    return sgn(np.dot(myw.T, myx))
#Update the weights
def new_w(old_w, myd, myx, a):
    return old_w+a*(myd-compute_yn(old_w,myx))*myx

for ii in range(500):
    i = 0
    sum=0
    for xn in x:
        m = new_w(m, b[i], xn, a)
        sum += (b[i]-compute_yn(m, xn))**2+0.0
    i += 1
    #compute MSE
    mean_square =np.sqrt(sum/N/2)
    num_MSE.append(mean_square)
    num_step.append(ii+1)
    #plt.plot(num_step, num_MSE)
print(m)
#draw The decision boundary
testx = np.array(range(-15, 25))
testy = -testx*m[1]/m[2]-m[0]/m[2]
for i in range(len(testy)):
    testy[i] = 0
plt.plot(data[0:N, 0], data[0:N, 1], 'r*', data[N:2*N, 0], data[N:2*N, 1], 'b*')
print(testx,testy)
plt.plot(testx, testy, 'g--')
plt.show()