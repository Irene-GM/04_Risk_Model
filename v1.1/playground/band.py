import matplotlib.pyplot as plt
import numpy as np; np.random.seed(42)

fig, axes = plt.subplots(ncols=3, sharey=True)
for ax in axes:
    ax.plot(np.arange(30), np.cumsum(np.random.randn(30)))

# set width and height in physical units (inches)
width = 20 # inch
height= 5 # inch
def resize(evt=None):
    w,h = fig.get_size_inches()
    l = ((w-width)/2.)/w
    b = ((h-height)/2.)/h
    fig.subplots_adjust(left=l, right=1.-l, bottom=b, top=1.-b)
    fig.canvas.draw_idle()


resize()
fig.canvas.mpl_connect("resize_event", resize)

plt.show()