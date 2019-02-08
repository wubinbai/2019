import matplotlib.pyplot as plt
plt.ion()    # 打开交互模式
    # 同时打开两个窗口显示图片
plt.figure()
plt.plot(range(10))
#plt.imshow(range(10))
plt.figure()
plt.plot(range(10,20))
#plt.imshow(range(10,20))
# 显示前关掉交互模式
plt.ioff()
#plt.show()
