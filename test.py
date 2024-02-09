import matplotlib.pyplot as plt

y = [4,13,5,6,1,2,3]
y1 = [1,3,4,6,3,6,7]

x = range(len(y))


plt.plot(x,y)
plt.plot(x,y1)

plt.legend()
plt.savefig("testing.png")