# **Python Libraries Programming Interview Questions**

---

## **Phase 1: Basics (Questions 1–50)**

Focus: Understanding Figure, Axes, Axis, and simple plotting functions.

**Sample questions:**

1. Create a simple line plot of `[1,2,3,4,5]` vs `[10,20,25,30,35]`
   → `plt.plot([1,2,3,4,5], [10,20,25,30,35])`

2. Plot two lines on the same figure with different colors
   → `plt.plot(x1, y1, 'r'); plt.plot(x2, y2, 'b')`

3. Change the line style to dashed and color to red
   → `plt.plot(x, y, 'r--')`

4. Add a title “My First Plot” to your figure
   → `plt.title("My First Plot")`

5. Label the x-axis as “Time” and y-axis as “Value”
   → `plt.xlabel("Time"); plt.ylabel("Value")`

6. Add a legend to differentiate two lines
   → `plt.legend(['Line 1', 'Line 2'])`

7. Create a figure with **size 10x6 inches**
   → `plt.figure(figsize=(10,6))`

8. Save a plot as `my_plot.png`
   → `plt.savefig('my_plot.png')`

9. Create a scatter plot for two lists `[1,2,3]` and `[4,5,6]`
   → `plt.scatter([1,2,3], [4,5,6])`

10. Plot multiple subplots (2x1) in a single figure
    → `plt.subplot(2,1,1); ...; plt.subplot(2,1,2); ...`

11. Use `plt.subplots()` to create 2x2 Axes and plot different data in each
    → `fig, axs = plt.subplots(2,2); axs[0,0].plot(...); axs[0,1].plot(...); axs[1,0].plot(...); axs[1,1].plot(...)`

12. Set the x-axis limits to 0–10 and y-axis limits to 0–100
    → `plt.xlim(0,10); plt.ylim(0,100)`

13. Add gridlines to your plot
    → `plt.grid(True)`

14. Annotate a point `(2,20)` with text “Important Point”
    → `plt.annotate('Important Point', xy=(2,20), xytext=(3,30), arrowprops=dict(arrowstyle='->'))`

15. Change marker style to `o` and size to `10`
    → `plt.plot(x, y, 'o', markersize=10)`

16. Use `plt.style.use('ggplot')` and observe changes
    → `plt.style.use('ggplot')`

17. Explore default colormaps with `plt.cm.viridis`
    → `plt.scatter(x, y, c=values, cmap=plt.cm.viridis)`

18. Plot a horizontal line at `y=15`
    → `plt.axhline(y=15, color='r', linestyle='--')`

19. Plot a vertical line at `x=3`
    → `plt.axvline(x=3, color='b', linestyle='--')`

20. Create a bar chart with categories `['A','B','C']` and values `[10,20,30]`
    → `plt.bar(['A','B','C'], [10,20,30])`


…and so on until **Question 50**, gradually including **patches** (Rectangle, Circle), basic **histograms**, and **simple customizations**.

---

## **Phase 2: Medium (Questions 51–130)**

Focus: Deepening control over the Artist hierarchy, multiple Axes, and styling.

**Sample questions:**
51. Plot a sine wave from 0 to 2π
    → `x = np.linspace(0, 2*np.pi, 100); y = np.sin(x); plt.plot(x, y)`

52. Add a cosine wave to the same plot
    → `plt.plot(x, np.cos(x))`

53. Adjust line width and alpha (transparency)
    → `plt.plot(x, y, linewidth=2, alpha=0.5)`

54. Add multiple legends for different data series
    → `plt.plot(x, y1, label='Sine'); plt.plot(x, y2, label='Cosine'); plt.legend()`

55. Create a stacked bar chart
    → `plt.bar(x, y1); plt.bar(x, y2, bottom=y1)`

56. Plot a histogram with 20 bins
    → `plt.hist(data, bins=20)`

57. Customize tick labels to show percentages
    → `plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))`

58. Rotate x-axis labels by 45 degrees
    → `plt.xticks(rotation=45)`

59. Place legend outside the Axes on the right
    → `plt.legend(loc='center left', bbox_to_anchor=(1,0.5))`

60. Use `ax.annotate()` to add arrows pointing to a peak
    → `ax.annotate('Peak', xy=(x_peak, y_peak), xytext=(x_peak+1, y_peak+1), arrowprops=dict(arrowstyle='->'))`

61. Combine a bar chart and line chart in one Axes
    → `plt.bar(x, y1); plt.plot(x, y2, color='r')`

62. Change the figure background color
    → `plt.figure(facecolor='lightgrey')`

63. Change the Axes background color
    → `ax.set_facecolor('lightyellow')`

64. Add a subplot that spans multiple columns
    → `plt.subplot2grid((2,2), (0,0), colspan=2)`

65. Share x-axis between multiple subplots
    → `fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)`

66. Create a log-scale plot on the y-axis
    → `plt.yscale('log'); plt.plot(x, y)`

67. Plot error bars using `plt.errorbar()`
    → `plt.errorbar(x, y, yerr=errors, fmt='o')`

68. Draw a pie chart with labels and explode effect
    → `plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%')`

69. Customize tick frequency using `MultipleLocator`
    → `ax.xaxis.set_major_locator(MultipleLocator(2))`

70. Use `tight_layout()` to avoid overlapping labels
    → `plt.tight_layout()`


…and continue building on **ticks, grids, color maps, advanced annotations, and composite figures**.

---

## **Phase 3: Advanced (Questions 131–200)**

Focus: Fully mastering Artist hierarchy, custom objects, interactive plots, and real-world-like data visualization.

**Sample questions:**
131. Manually create a Line2D object and add it to Axes
     → `from matplotlib.lines import Line2D; line = Line2D(x, y, color='r'); ax.add_line(line)`

132. Manually create a Text object and place it in figure coordinates
     → `from matplotlib.text import Text; text = Text(x=0.5, y=0.9, text='Hello', transform=fig.transFigure); fig.add_artist(text)`

133. Create custom patch objects (Rectangle, Circle) on a plot
     → `from matplotlib.patches import Rectangle, Circle; ax.add_patch(Rectangle((1,1),2,3)); ax.add_patch(Circle((2,2), radius=1))`

134. Overlay multiple Axes in the same Figure with different scales
     → `ax2 = ax.figure.add_axes(ax.get_position(), sharex=ax, frameon=False); ax2.plot(x, y2)`

135. Use `GridSpec` to create complex subplot arrangements
     → `import matplotlib.gridspec as gridspec; gs = gridspec.GridSpec(2,3); ax1 = fig.add_subplot(gs[0, :2])`

136. Create a 3D plot with `Axes3D`
     → `from mpl_toolkits.mplot3d import Axes3D; ax = fig.add_subplot(111, projection='3d')`

137. Plot a 3D surface using `plot_surface`
     → `ax.plot_surface(X, Y, Z, cmap='viridis')`

138. Use `imshow()` to display a 2D array as an image with a colormap
     → `plt.imshow(array, cmap='hot')`

139. Add a colorbar to a heatmap
     → `plt.colorbar(im)`

140. Create a scatter plot with size and color mapped to data values
     → `plt.scatter(x, y, s=sizes, c=colors, cmap='viridis')`

141. Animate a sine wave over time using `FuncAnimation`
     → `from matplotlib.animation import FuncAnimation; ani = FuncAnimation(fig, update, frames=100, interval=50)`

142. Plot multiple time series with shared x-axis
     → `fig, axs = plt.subplots(n,1, sharex=True); axs[0].plot(t1,y1); axs[1].plot(t2,y2)`

143. Customize major and minor ticks differently
     → `ax.xaxis.set_major_locator(MultipleLocator(1)); ax.xaxis.set_minor_locator(MultipleLocator(0.2))`

144. Use `transforms` to position an annotation relative to Axes, not data
     → `ax.annotate('Note', xy=(0.5,0.5), xycoords='axes fraction')`

145. Combine multiple figures in one figure canvas using `Figure.add_axes()`
     → `fig = plt.figure(); ax1 = fig.add_axes([0,0,0.5,0.5]); ax2 = fig.add_axes([0.5,0.5,0.4,0.4])`

146. Export a plot as PDF, SVG, and PNG
     → `plt.savefig('plot.pdf'); plt.savefig('plot.svg'); plt.savefig('plot.png')`

147. Create a complex dashboard-like figure with inset plots
     → `from mpl_toolkits.axes_grid1.inset_locator import inset_axes; ax_inset = inset_axes(ax, width="30%", height="30%", loc=1)`

148. Plot data from a CSV file using pandas and Matplotlib
     → `import pandas as pd; df = pd.read_csv('data.csv'); df.plot(x='Time', y='Value')`

149. Create custom colormaps for heatmaps
     → `from matplotlib.colors import LinearSegmentedColormap; cmap = LinearSegmentedColormap.from_list('mycmap',['blue','white','red'])`

150. Create a twin y-axis for dual-scale plots
     → `ax2 = ax.twinx(); ax2.plot(x, y2, color='r')`


---

# **Matplotlib Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Figure, Axes, Axis, basic plotting, labels, legends, styles*

1. Import `matplotlib.pyplot` as `plt` and plot `[1,2,3]` vs `[4,5,6]`.
   → `import matplotlib.pyplot as plt; plt.plot([1,2,3],[4,5,6]); plt.show()`

2. Create a line plot with a green dashed line.
   → `plt.plot([1,2,3],[4,5,6],'g--')`

3. Plot two lines on the same figure with different colors.
   → `plt.plot([1,2,3],[4,5,6],'r'); plt.plot([1,2,3],[6,5,4],'b')`

4. Add a title “Basic Line Plot”.
   → `plt.title("Basic Line Plot")`

5. Label the x-axis as “Time” and y-axis as “Value”.
   → `plt.xlabel("Time"); plt.ylabel("Value")`

6. Add a legend to differentiate two lines.
   → `plt.plot([1,2,3],[4,5,6],label='Line1'); plt.plot([1,2,3],[6,5,4],label='Line2'); plt.legend()`

7. Change line width to 3.
   → `plt.plot([1,2,3],[4,5,6],linewidth=3)`

8. Set marker style to `o` with size 8.
   → `plt.plot([1,2,3],[4,5,6],'o',markersize=8)`

9. Create a figure of size 8x6 inches.
   → `plt.figure(figsize=(8,6))`

10. Save the figure as `plot1.png`.
    → `plt.savefig('plot1.png')`

11. Create a scatter plot of `[1,2,3]` vs `[4,5,6]`.
    → `plt.scatter([1,2,3],[4,5,6])`

12. Change scatter marker color to red.
    → `plt.scatter([1,2,3],[4,5,6],color='red')`

13. Change scatter marker to `^` and size to 100.
    → `plt.scatter([1,2,3],[4,5,6],marker='^',s=100)`

14. Plot multiple subplots (2x1) with different data.
    → `plt.subplot(2,1,1); plt.plot([1,2,3],[4,5,6]); plt.subplot(2,1,2); plt.plot([1,2,3],[6,5,4])`

15. Use `plt.subplots()` to create 2x2 Axes and plot in each.
    → `fig, axs = plt.subplots(2,2); axs[0,0].plot([1,2,3],[4,5,6])`

16. Set x-axis limits from 0 to 10 and y-axis from 0 to 50.
    → `plt.xlim(0,10); plt.ylim(0,50)`

17. Add gridlines to the plot.
    → `plt.grid(True)`

18. Rotate x-axis labels by 45 degrees.
    → `plt.xticks(rotation=45)`

19. Annotate point `(2,20)` with “Important Point”.
    → `plt.annotate("Important Point",(2,20))`

20. Add horizontal line at y=15.
    → `plt.axhline(y=15,color='r')`

21. Add vertical line at x=3.
    → `plt.axvline(x=3,color='b')`

22. Create a bar chart with categories `['A','B','C']` and values `[10,20,30]`.
    → `plt.bar(['A','B','C'],[10,20,30])`

23. Change bar colors to blue.
    → `plt.bar(['A','B','C'],[10,20,30],color='blue')`

24. Change bar width to 0.5.
    → `plt.bar(['A','B','C'],[10,20,30],width=0.5)`

25. Plot a histogram of `[1,2,2,3,3,3,4,4,5]` with 5 bins.
    → `plt.hist([1,2,2,3,3,3,4,4,5],bins=5)`

26. Add title and labels to histogram.
    → `plt.title("Histogram"); plt.xlabel("Value"); plt.ylabel("Frequency")`

27. Use `plt.style.use('ggplot')` for a different look.
    → `plt.style.use('ggplot')`

28. Explore `plt.style.available` to list all styles.
    → `plt.style.available`

29. Create a figure with two y-axes using `twinx()`.
    → `fig, ax1 = plt.subplots(); ax2 = ax1.twinx()`

30. Plot sine and cosine on the same axes.
    → `import numpy as np; x=np.linspace(0,2*np.pi,100); plt.plot(x,np.sin(x)); plt.plot(x,np.cos(x))`

31. Change figure background color.
    → `plt.figure(facecolor='lightgrey')`

32. Change Axes background color.
    → `ax = plt.gca(); ax.set_facecolor('lightyellow')`

33. Customize tick frequency on x-axis using `MultipleLocator`.
    → `from matplotlib.ticker import MultipleLocator; ax = plt.gca(); ax.xaxis.set_major_locator(MultipleLocator(2))`

34. Add minor ticks on y-axis.
    → `ax.yaxis.set_minor_locator(MultipleLocator(1)); ax.minorticks_on()`

35. Add a legend outside the plot.
    → `plt.legend(loc='center left', bbox_to_anchor=(1,0.5))`

36. Combine a bar chart and line chart in one Axes.
    → `plt.bar(['A','B','C'],[10,20,30]); plt.plot(['A','B','C'],[10,20,30],color='r')`

37. Change font size of title and labels.
    → `plt.title("Title",fontsize=16); plt.xlabel("X",fontsize=14); plt.ylabel("Y",fontsize=14)`

38. Change font family to `serif`.
    → `plt.rcParams['font.family'] = 'serif'`

39. Create a figure with multiple rows and columns of subplots.
    → `fig, axs = plt.subplots(3,2)`

40. Share x-axis between multiple subplots.
    → `fig, axs = plt.subplots(2,1,sharex=True)`

41. Share y-axis between multiple subplots.
    → `fig, axs = plt.subplots(2,1,sharey=True)`

42. Add space between subplots using `plt.tight_layout()`.
    → `plt.tight_layout()`

43. Create a pie chart with labels `['A','B','C']` and values `[10,20,30]`.
    → `plt.pie([10,20,30],labels=['A','B','C'])`

44. Explode the first slice of the pie chart.
    → `plt.pie([10,20,30],labels=['A','B','C'],explode=[0.1,0,0])`

45. Show percentage values on pie chart.
    → `plt.pie([10,20,30],labels=['A','B','C'],autopct='%1.1f%%')`

46. Plot a horizontal bar chart.
    → `plt.barh(['A','B','C'],[10,20,30])`

47. Change orientation of tick labels on horizontal bar chart.
    → `plt.yticks(rotation=45)`

48. Create a stacked bar chart.
    → `plt.bar(['A','B','C'],[5,10,15]); plt.bar(['A','B','C'],[5,10,15],bottom=[5,10,15])`

49. Plot a simple area chart using `fill_between()`.
    → `x=np.arange(0,5,1); y=np.array([1,2,3,4,5]); plt.fill_between(x,y)`

50. Use `plt.show()` to display the figure.
    → `plt.show()`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Deep control of Artist hierarchy, multiple Axes, styles, annotations*

51. Plot a sine wave from 0 to 2π with 100 points.
    → `x = np.linspace(0, 2*np.pi, 100); plt.plot(x, np.sin(x))`

52. Add a cosine wave to the same plot.
    → `plt.plot(x, np.cos(x))`

53. Change alpha (transparency) to 0.5 for one line.
    → `plt.plot(x, np.sin(x), alpha=0.5)`

54. Add multiple legends for different data series.
    → `plt.plot(x,np.sin(x),label='sin'); plt.plot(x,np.cos(x),label='cos'); plt.legend()`

55. Plot a bar chart with error bars.
    → `plt.bar([1,2,3],[10,20,30],yerr=[1,2,3])`

56. Plot a histogram with 20 bins and normalized frequency.
    → `plt.hist(data, bins=20, density=True)`

57. Change tick labels to show percentages.
    → `from matplotlib.ticker import PercentFormatter; plt.gca().yaxis.set_major_formatter(PercentFormatter(1))`

58. Rotate tick labels by 90 degrees.
    → `plt.xticks(rotation=90)`

59. Place legend outside the axes on the right.
    → `plt.legend(loc='center left', bbox_to_anchor=(1,0.5))`

60. Add an arrow annotation pointing to the maximum value.
    → `plt.annotate('Max', xy=(x[np.argmax(y)], max(y)), xytext=(x[np.argmax(y)], max(y)+1), arrowprops=dict(arrowstyle='->'))`

61. Combine bar chart and line plot in same Axes.
    → `plt.bar([1,2,3],[10,20,30]); plt.plot([1,2,3],[10,20,30],color='r')`

62. Use `plt.subplots_adjust()` to control spacing between plots.
    → `plt.subplots_adjust(left=0.1,right=0.9,hspace=0.4)`

63. Create a log-scale plot on y-axis.
    → `plt.yscale('log'); plt.plot([1,2,3],[10,100,1000])`

64. Create a log-scale plot on x-axis.
    → `plt.xscale('log'); plt.plot([1,10,100],[1,2,3])`

65. Plot a scatter plot with size mapped to data.
    → `sizes = [50,100,200]; plt.scatter([1,2,3],[4,5,6],s=sizes)`

66. Plot a scatter plot with color mapped to data.
    → `colors = [10,20,30]; plt.scatter([1,2,3],[4,5,6],c=colors)`

67. Use `plt.cm.viridis` colormap for scatter plot.
    → `plt.scatter([1,2,3],[4,5,6],c=colors,cmap=plt.cm.viridis)`

68. Plot multiple subplots with shared x-axis.
    → `fig, axs = plt.subplots(2,1,sharex=True)`

69. Plot multiple subplots with shared y-axis.
    → `fig, axs = plt.subplots(2,1,sharey=True)`

70. Use `ax.annotate()` to label a local minimum.
    → `ax.annotate('Min', xy=(x[idx], y[idx]), xytext=(x[idx], y[idx]-1), arrowprops=dict(arrowstyle='->'))`

71. Create a stacked area chart.
    → `plt.stackplot(x, y1, y2, labels=['y1','y2']); plt.legend()`

72. Add minor gridlines to the plot.
    → `plt.minorticks_on(); plt.grid(which='minor', linestyle=':', linewidth=0.5)`

73. Change linestyle to dotted for one line.
    → `plt.plot(x, y, linestyle=':')`

74. Plot multiple lines with different line styles.
    → `plt.plot(x, y1, '-'); plt.plot(x, y2, '--'); plt.plot(x, y3, ':')`

75. Add text annotation using figure coordinates.
    → `plt.gcf().text(0.5,0.95,'Figure Text',ha='center')`

76. Change legend font size and frame.
    → `plt.legend(fontsize=12, frameon=True)`

77. Change tick direction to `in` or `out`.
    → `plt.tick_params(direction='in')`

78. Change tick length and width.
    → `plt.tick_params(length=10, width=2)`

79. Create a bar chart with error bars.
    → `plt.bar([1,2,3],[10,20,30],yerr=[1,2,3])`

80. Customize colors of a histogram.
    → `plt.hist(data, bins=10, color='green', edgecolor='black')`

81. Plot a cumulative histogram.
    → `plt.hist(data, bins=10, cumulative=True)`

82. Create a scatter plot with categorical x-axis.
    → `plt.scatter(['A','B','C'],[10,20,30])`

83. Plot data using pandas DataFrame.
    → `import pandas as pd; df=pd.DataFrame({'x':[1,2,3],'y':[4,5,6]}); df.plot(x='x',y='y')`

84. Change x-axis and y-axis limits dynamically using data.
    → `plt.xlim(min(x), max(x)); plt.ylim(min(y), max(y))`

85. Use `ax.set_xticks()` to set custom ticks.
    → `ax.set_xticks([0,1,2,3,4])`

86. Use `ax.set_xticklabels()` to set custom labels.
    → `ax.set_xticklabels(['A','B','C','D','E'])`

87. Use `ax.set_yticks()` and `ax.set_yticklabels()`.
    → `ax.set_yticks([0,10,20]); ax.set_yticklabels(['Low','Medium','High'])`

88. Create a horizontal bar chart with colors based on values.
    → `values=[10,20,30]; plt.barh(['A','B','C'],values,color=['red','green','blue'])`

89. Create a figure with inset Axes.
    → `fig, ax = plt.subplots(); inset = fig.add_axes([0.5,0.5,0.4,0.4]); inset.plot(x,y)`

90. Plot two lines and fill the area between them.
    → `plt.plot(x,y1); plt.plot(x,y2); plt.fill_between(x,y1,y2,alpha=0.3)`

91. Use `ax.fill_between()` with different alpha values.
    → `ax.fill_between(x,y1,y2,alpha=0.5)`

92. Plot a bar chart with positive and negative values.
    → `plt.bar([1,2,3],[10,-5,15])`

93. Plot multiple lines with different markers.
    → `plt.plot(x,y1,'o-'); plt.plot(x,y2,'s--')`

94. Customize marker edge color and width.
    → `plt.plot(x,y,'o',markeredgecolor='black',markeredgewidth=2)`

95. Create a polar plot.
    → `plt.subplot(projection='polar'); plt.plot(theta,r)`

96. Plot a sine wave in polar coordinates.
    → `theta=np.linspace(0,2*np.pi,100); r=np.sin(theta); plt.polar(theta,r)`

97. Customize polar plot gridlines and labels.
    → `ax=plt.subplot(projection='polar'); ax.set_rticks([0.5,1,1.5]); ax.set_theta_zero_location('N')`

98. Plot a histogram on a logarithmic scale.
    → `plt.hist(data, bins=10); plt.yscale('log')`

99. Plot multiple histograms on the same Axes.
    → `plt.hist([data1,data2], bins=10, label=['d1','d2']); plt.legend()`

100. Customize histogram bin edges.
     → `bins=[0,1,2,3,5,8]; plt.hist(data, bins=bins)`


…*(continue questions 101–130, covering error bars, color maps, twin axes, GridSpec, minor/major ticks, styles, font customization, pie charts, horizontal/stacked bar charts, legends, annotation placement, figure size adjustments, combining multiple plotting types, subplot arrangements)*.

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Fully mastering Artist hierarchy, custom objects, 3D plots, interactive plots, animations*

131. Manually create a `Line2D` object and add it to Axes.
     → `from matplotlib.lines import Line2D; line = Line2D([0,1],[0,1]); ax = plt.gca(); ax.add_line(line)`

132. Manually create a `Text` object at figure coordinates.
     → `from matplotlib.text import Text; text = Text(x=0.5, y=0.5, text='Hello', transform=plt.gcf().transFigure); plt.gca().add_artist(text)`

133. Create custom patch objects (Rectangle, Circle) on a plot.
     → `from matplotlib.patches import Rectangle, Circle; ax=plt.gca(); ax.add_patch(Rectangle((0,0),1,2)); ax.add_patch(Circle((1,1),0.5))`

134. Overlay multiple Axes in the same Figure with different scales.
     → `fig=plt.figure(); ax1=fig.add_axes([0,0,1,1]); ax2=fig.add_axes([0,0,1,1], frameon=False); ax2.plot(...)`

135. Use `GridSpec` to create complex subplot arrangements.
     → `import matplotlib.gridspec as gridspec; gs = gridspec.GridSpec(2,2); fig=plt.figure(); ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,:])`

136. Create a 3D plot using `Axes3D`.
     → `from mpl_toolkits.mplot3d import Axes3D; fig=plt.figure(); ax=fig.add_subplot(111, projection='3d')`

137. Plot a 3D surface using `plot_surface`.
     → `X,Y = np.meshgrid(x,y); Z = np.sin(X)*np.cos(Y); ax.plot_surface(X,Y,Z)`

138. Plot a 3D wireframe.
     → `ax.plot_wireframe(X,Y,Z)`

139. Plot a 3D scatter plot.
     → `ax.scatter(X.flatten(), Y.flatten(), Z.flatten())`

140. Plot 3D contour plot.
     → `ax.contour3D(X,Y,Z,50)`

141. Display a 2D array as an image using `imshow()`.
     → `plt.imshow(data)`

142. Add a colorbar to the image.
     → `plt.colorbar()`

143. Customize colormap of `imshow()`.
     → `plt.imshow(data, cmap='viridis')`

144. Use logarithmic colormap normalization.
     → `from matplotlib.colors import LogNorm; plt.imshow(data, norm=LogNorm())`

145. Plot a scatter plot with size and color mapped to two separate data columns.
     → `plt.scatter(x,y,s=sizes,c=colors)`

146. Add a twin y-axis in a 3D plot.
     → `# 3D twin axes is not directly supported; use inset Axes or overlayed 2D plots for similar effect`

147. Annotate a point in a 3D plot.
     → `ax.text(x[0], y[0], z[0], "Point")`

148. Use `transforms` to position annotations relative to axes.
     → `import matplotlib.transforms as mtransforms; ax.text(0.5,0.5,"Text", transform=ax.transAxes)`

149. Combine multiple figures in one canvas using `Figure.add_axes()`.
     → `fig=plt.figure(); ax1=fig.add_axes([0,0,0.5,0.5]); ax2=fig.add_axes([0.5,0.5,0.5,0.5])`

150. Export a figure as PDF, SVG, and PNG.
     → `plt.savefig('figure.pdf'); plt.savefig('figure.svg'); plt.savefig('figure.png')`

151. Animate a sine wave using `FuncAnimation`.
     → `from matplotlib.animation import FuncAnimation; def update(i): line.set_ydata(np.sin(x+i/10)); ani = FuncAnimation(fig, update, frames=100)`

152. Animate multiple lines with different speeds.
     → `def update(i): line1.set_ydata(np.sin(x+i/10)); line2.set_ydata(np.cos(x+i/5)); ani = FuncAnimation(fig, update, frames=100)`

153. Create a time series plot from CSV data.
     → `import pandas as pd; df=pd.read_csv('data.csv'); plt.plot(pd.to_datetime(df['Date']), df['Value'])`

154. Plot multiple time series with shared x-axis.
     → `fig, axs = plt.subplots(n,1,sharex=True); axs[0].plot(...); axs[1].plot(...)`

155. Customize major and minor ticks differently.
     → `from matplotlib.ticker import MultipleLocator; ax.xaxis.set_major_locator(MultipleLocator(5)); ax.xaxis.set_minor_locator(MultipleLocator(1))`

156. Plot multiple y-axis scales in one figure.
     → `fig, ax1 = plt.subplots(); ax2 = ax1.twinx(); ax1.plot(...); ax2.plot(...)`

157. Plot multiple heatmaps side by side.
     → `fig, axs=plt.subplots(1,2); axs[0].imshow(data1); axs[1].imshow(data2)`

158. Use `pcolormesh()` for irregular grids.
     → `plt.pcolormesh(X,Y,Z)`

159. Plot a hexbin chart.
     → `plt.hexbin(x,y,C=values,gridsize=30,cmap='viridis')`

160. Plot a violin plot.
     → `plt.violinplot(data)`

161. Customize violin plot colors and widths.
     → `vp = plt.violinplot(data); for pc in vp['bodies']: pc.set_facecolor('red'); pc.set_alpha(0.5)`

162. Plot a boxplot with custom colors.
     → `plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor='cyan'))`

163. Overlay scatter points on boxplot.
     → `plt.boxplot(data); plt.scatter(np.random.rand(len(data))*1.1, data)`

164. Create a swarmplot using matplotlib only.
     → `for i,y in enumerate(data): x = np.random.normal(i,0.05,len(y)); plt.scatter(x,y)`

165. Plot correlation heatmap from pandas DataFrame.
     → `import seaborn as sns; sns.heatmap(df.corr())`

166. Use masks to hide parts of heatmap.
     → `mask = np.triu(np.ones_like(df.corr(), dtype=bool)); sns.heatmap(df.corr(), mask=mask)`

167. Plot time series with rolling mean overlay.
     → `plt.plot(df['Value']); plt.plot(df['Value'].rolling(7).mean())`

168. Plot time series with shaded error region.
     → `plt.plot(x, y); plt.fill_between(x, y-yerr, y+yerr, alpha=0.3)`

169. Use `step()` to create step plots.
     → `plt.step(x, y)`

170. Plot a dual-axis plot with different scales and line styles.
     → `fig, ax1 = plt.subplots(); ax2 = ax1.twinx(); ax1.plot(x,y1,'-'); ax2.plot(x,y2,'--')`

171. Create inset axes for zoomed-in plot.
     → `from mpl_toolkits.axes_grid1.inset_locator import inset_axes; axins = inset_axes(ax, width="30%", height="30%", loc='upper right')`

172. Add custom gridlines for inset axes.
     → `axins.grid(True, linestyle='--')`

173. Add annotation with arrow pointing to inset region.
     → `ax.annotate('', xy=(x1,y1), xytext=(x2,y2), arrowprops=dict(arrowstyle='->'))`

174. Combine polar plot and Cartesian plot in one figure.
     → `fig=plt.figure(); ax1=fig.add_subplot(121); ax2=fig.add_subplot(122, projection='polar')`

175. Plot multiple pie charts in subplots.
     → `fig, axs=plt.subplots(1,2); axs[0].pie([10,20,30]); axs[1].pie([5,15,20])`

176. Plot nested pie chart.
     → `plt.pie([sum_outer), sum_inner], radius=[1,0.7])  # Simplified`

177. Plot a donut chart.
     → `plt.pie([10,20,30], radius=1, wedgeprops=dict(width=0.3))`

178. Use `broken_barh` to create timeline plots.
     → `plt.broken_barh([(1,2),(4,3)], (10,5))`

179. Create a Gantt chart.
     → `plt.broken_barh([(start,end-start) for start,end in tasks], (y_pos, height))`

180. Plot calendar heatmap.
     → `# Typically use seaborn or custom grid with imshow to represent days`

181. Create a custom colormap from scratch.
     → `from matplotlib.colors import LinearSegmentedColormap; cmap=LinearSegmentedColormap.from_list('mycmap',['blue','white','red'])`

182. Use diverging colormap for positive and negative values.
     → `plt.imshow(data, cmap='bwr')`

183. Plot a contour plot with labeled contours.
     → `CS = plt.contour(X,Y,Z); plt.clabel(CS)`

184. Fill contours with colors.
     → `plt.contourf(X,Y,Z)`

185. Overlay scatter on contour plot.
     → `plt.contour(X,Y,Z); plt.scatter(x,y)`

186. Plot quiver (vector field) plot.
     → `plt.quiver(X,Y,U,V)`

187. Plot streamplot for fluid dynamics visualization.
     → `plt.streamplot(X,Y,U,V)`

188. Create custom legend handles.
     → `from matplotlib.patches import Patch; handles=[Patch(color='red',label='A')]; plt.legend(handles=handles)`

189. Use proxy artists for complex legend entries.
     → `from matplotlib.lines import Line2D; proxy = [Line2D([0],[0], color='red', lw=2)]; plt.legend(handles=proxy)`

190. Add annotations with formatted numbers (scientific, percentage).
     → `plt.text(1,2,'{:.2e}'.format(value)); plt.text(1,2,'{:.1%}'.format(value))`

191. Customize all fonts in the figure globally.
     → `plt.rcParams.update({'font.size':14,'font.family':'serif'})`

192. Plot multiple figures in a loop efficiently.
     → `for i in range(3): fig, ax = plt.subplots(); ax.plot(data[i])`

193. Plot multiple datasets with consistent styling.
     → `for d in datasets: plt.plot(d, linestyle='-', marker='o')`

194. Create reusable plotting function for custom style.
     → `def plot_custom(x,y): plt.plot(x,y, linestyle='--', marker='s'); plt.show()`

195. Use `rcParams` to change default plot settings.
     → `plt.rcParams['lines.linewidth']=2; plt.rcParams['axes.grid']=True`

196. Create figure with interactive widgets (sliders) using matplotlib.
     → `from matplotlib.widgets import Slider; slider = Slider(ax_slider, 'X', 0, 10)`

197. Use `mpl_toolkits.axes_grid1` for colorbar alignment.
     → `from mpl_toolkits.axes_grid1 import make_axes_locatable; divider=make_axes_locatable(ax); cax=divider.append_axes('right', size='5%', pad=0.05); plt.colorbar(im, cax=cax)`

198. Plot a 3D animation.
     → `# Use FuncAnimation to update ax.plot_surface or ax.scatter in 3D`

199. Combine Matplotlib with PIL to annotate images.
     → `from PIL import Image; img=Image.open('img.png'); plt.imshow(img); plt.text(10,10,'Label')`

200. Create a fully customized dashboard-like figure with multiple plots, annotations, legends, inset axes, and colorbars.
     → `# Combine subplots, inset_axes, colorbars, custom annotations, legends; assemble with plt.figure() and plt.add_axes() for full dashboard`


---

Here’s the structured draft:

---

# **NumPy Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Arrays, creation, indexing, data types, basic operations*

1. Import NumPy as `np` and print its version.
   → `import numpy as np; print(np.__version__)`

2. Create a 1D array `[1,2,3,4,5]`.
   → `arr = np.array([1,2,3,4,5])`

3. Create a 2D array `[[1,2,3],[4,5,6]]`.
   → `arr = np.array([[1,2,3],[4,5,6]])`

4. Check the type and shape of an array.
   → `type(arr); arr.shape`

5. Create an array of zeros with shape (3,4).
   → `np.zeros((3,4))`

6. Create an array of ones with shape (2,5).
   → `np.ones((2,5))`

7. Create an array filled with a constant value, e.g., 7, shape (3,3).
   → `np.full((3,3),7)`

8. Create an array using `arange(0,10,2)`.
   → `np.arange(0,10,2)`

9. Create an array using `linspace(0,1,5)`.
   → `np.linspace(0,1,5)`

10. Create a 3x3 identity matrix using `eye()`.
    → `np.eye(3)`

11. Create a 2x3 random array using `np.random.rand()`.
    → `np.random.rand(2,3)`

12. Create a 2x3 normal-distributed array using `np.random.randn()`.
    → `np.random.randn(2,3)`

13. Check data type of array elements using `dtype`.
    → `arr.dtype`

14. Change data type using `astype()`.
    → `arr.astype(float)`

15. Access a specific element in 1D array.
    → `arr[2]`

16. Access a specific element in 2D array.
    → `arr[1,2]`

17. Slice a 1D array `[2:5]`.
    → `arr[2:5]`

18. Slice a 2D array to get submatrix.
    → `arr[0:2,1:3]`

19. Access last element using negative index.
    → `arr[-1]`

20. Access last row in a 2D array.
    → `arr[-1,:]`

21. Access last column in a 2D array.
    → `arr[:,-1]`

22. Use boolean indexing to select elements >3.
    → `arr[arr>3]`

23. Use boolean indexing to select even numbers.
    → `arr[arr%2==0]`

24. Use fancy indexing to select specific rows/columns.
    → `arr[[0,1],[1,2]]`

25. Modify a single element in a 1D array.
    → `arr[0] = 10`

26. Modify an entire row in a 2D array.
    → `arr[1,:] = [7,8,9]`

27. Modify an entire column in a 2D array.
    → `arr[:,2] = [1,2]`

28. Add two arrays element-wise.
    → `arr1 + arr2`

29. Subtract two arrays element-wise.
    → `arr1 - arr2`

30. Multiply two arrays element-wise.
    → `arr1 * arr2`

31. Divide two arrays element-wise.
    → `arr1 / arr2`

32. Perform matrix multiplication using `@` or `dot()`.
    → `arr1 @ arr2.T` or `np.dot(arr1, arr2.T)`

33. Square each element of an array.
    → `arr**2`

34. Take the square root of each element.
    → `np.sqrt(arr)`

35. Compute sum of all elements.
    → `arr.sum()`

36. Compute sum along axis 0 and axis 1.
    → `arr.sum(axis=0); arr.sum(axis=1)`

37. Compute mean and median of an array.
    → `arr.mean(); np.median(arr)`

38. Compute standard deviation and variance.
    → `arr.std(); arr.var()`

39. Find minimum and maximum values.
    → `arr.min(); arr.max()`

40. Find indices of minimum and maximum using `argmin` and `argmax`.
    → `arr.argmin(); arr.argmax()`

41. Use `np.unique()` to get unique elements.
    → `np.unique(arr)`

42. Use `np.sort()` to sort an array.
    → `np.sort(arr)`

43. Use `np.argsort()` to get sorting indices.
    → `np.argsort(arr)`

44. Flatten a 2D array to 1D.
    → `arr.flatten()`

45. Reshape a 1D array to 2D.
    → `arr.reshape(1,5)`

46. Transpose a 2D array.
    → `arr.T`

47. Concatenate two arrays along axis 0.
    → `np.concatenate([arr1,arr2], axis=0)`

48. Concatenate two arrays along axis 1.
    → `np.concatenate([arr1,arr2], axis=1)`

49. Split an array into two parts.
    → `np.split(arr,2)`

50. Split a 2D array vertically and horizontally.
    → `np.vsplit(arr,2); np.hsplit(arr,2)`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced indexing, broadcasting, linear algebra, statistics, I/O*

51. Create a 5x5 array of random integers between 0 and 10.
    → `np.random.randint(0,10,(5,5))`

52. Mask elements greater than 5 using boolean indexing.
    → `arr[arr>5]`

53. Replace elements divisible by 2 with -1.
    → `arr[arr%2==0] = -1`

54. Select every second element from a 1D array.
    → `arr[::2]`

55. Select every second row from a 2D array.
    → `arr[::2,:]`

56. Reverse a 1D array using slicing.
    → `arr[::-1]`

57. Reverse rows and columns in a 2D array.
    → `arr[::-1,::-1]`

58. Broadcast addition of a 1D array to 2D array.
    → `arr2d + arr1d`

59. Multiply a 2D array by a 1D array using broadcasting.
    → `arr2d * arr1d`

60. Add scalar value to entire array.
    → `arr + 5`

61. Compute element-wise exponentials.
    → `np.exp(arr)`

62. Compute natural logarithm of array elements.
    → `np.log(arr)`

63. Compute sine and cosine of an array.
    → `np.sin(arr); np.cos(arr)`

64. Compute dot product of two vectors.
    → `np.dot(v1,v2)`

65. Compute cross product of two vectors.
    → `np.cross(v1,v2)`

66. Compute determinant of a 2x2 matrix.
    → `np.linalg.det(mat2x2)`

67. Compute determinant of a 3x3 matrix.
    → `np.linalg.det(mat3x3)`

68. Compute matrix inverse.
    → `np.linalg.inv(mat)`

69. Compute eigenvalues and eigenvectors.
    → `np.linalg.eig(mat)`

70. Compute singular value decomposition (SVD).
    → `np.linalg.svd(mat)`

71. Solve a system of linear equations using `np.linalg.solve()`.
    → `np.linalg.solve(A,b)`

72. Compute rank of a matrix.
    → `np.linalg.matrix_rank(mat)`

73. Compute trace of a matrix.
    → `np.trace(mat)`

74. Compute covariance matrix using `np.cov()`.
    → `np.cov(x,y)`

75. Compute correlation coefficient using `np.corrcoef()`.
    → `np.corrcoef(x,y)`

76. Compute histogram of an array using `np.histogram()`.
    → `np.histogram(arr,bins=10)`

77. Compute cumulative sum using `cumsum()`.
    → `arr.cumsum()`

78. Compute cumulative product using `cumprod()`.
    → `arr.cumprod()`

79. Round elements to nearest integer using `round()`.
    → `np.round(arr)`

80. Floor and ceil elements using `floor()` and `ceil()`.
    → `np.floor(arr); np.ceil(arr)`

81. Clip array elements between two values.
    → `np.clip(arr, 0, 5)`

82. Find where elements satisfy a condition using `np.where()`.
    → `np.where(arr>3)`

83. Use `np.take()` to select elements by index.
    → `np.take(arr,[0,2,4])`

84. Use `np.put()` to replace elements at indices.
    → `np.put(arr,[0,2],[10,20])`

85. Use `np.choose()` for conditional selection.
    → `np.choose([0,1,2],[arr1,arr2,arr3])`

86. Generate random integers with specific seed.
    → `np.random.seed(42); np.random.randint(0,10,5)`

87. Shuffle elements of an array randomly.
    → `np.random.shuffle(arr)`

88. Repeat elements of an array.
    → `np.repeat(arr,3)`

89. Tile an array to repeat along axes.
    → `np.tile(arr,(2,3))`

90. Stack arrays vertically using `vstack()`.
    → `np.vstack([arr1,arr2])`

91. Stack arrays horizontally using `hstack()`.
    → `np.hstack([arr1,arr2])`

92. Stack arrays along a new axis using `stack()`.
    → `np.stack([arr1,arr2], axis=0)`

93. Split arrays using `hsplit()` and `vsplit()`.
    → `np.hsplit(arr,2); np.vsplit(arr,2)`

94. Load array from text file using `np.loadtxt()`.
    → `np.loadtxt('file.txt')`

95. Save array to text file using `np.savetxt()`.
    → `np.savetxt('file.txt', arr)`

96. Load array from binary file using `np.load()`.
    → `np.load('file.npy')`

97. Save array to binary file using `np.save()`.
    → `np.save('file.npy', arr)`

98. Use structured arrays to store mixed data types.
    → `dtype=[('name','U10'),('age','i4')]; arr = np.array([('Alice',25)], dtype=dtype)`

99. Access fields of structured array.
    → `arr['name']; arr['age']`

100. Convert list of lists to NumPy array.
     → `np.array([[1,2,3],[4,5,6]])`


…*(questions 101–130 include advanced slicing, advanced broadcasting, linear algebra with large matrices, generating random numbers, cumulative and windowed operations, and statistics functions like percentile, quantile, masked arrays, and structured arrays)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Advanced manipulations, broadcasting tricks, memory efficiency, performance, complex operations*

131. Create a 3D array and index slices along each axis.
     → `arr = np.arange(27).reshape(3,3,3); arr[0,:,:]; arr[:,1,:]; arr[:,:,2]`

132. Swap axes of a 3D array.
     → `np.swapaxes(arr, 0, 2)`

133. Reshape 3D array to 2D.
     → `arr.reshape(3,9)`

134. Flatten 3D array to 1D.
     → `arr.flatten()`

135. Use `np.broadcast_arrays()` to align shapes.
     → `a,b = np.broadcast_arrays(arr1, arr2)`

136. Use `np.meshgrid()` to create coordinate grids.
     → `X,Y = np.meshgrid(x,y)`

137. Compute function values over meshgrid.
     → `Z = np.sin(X) + np.cos(Y)`

138. Use `np.vectorize()` for element-wise operations on arrays.
     → `f = np.vectorize(lambda x: x**2); f(arr)`

139. Perform element-wise comparison between arrays.
     → `arr1 == arr2`

140. Use `np.all()` and `np.any()` for logical tests.
     → `np.all(arr>0); np.any(arr<0)`

141. Use `np.isclose()` to compare float arrays.
     → `np.isclose(arr1, arr2)`

142. Compute Manhattan distance between two vectors.
     → `np.sum(np.abs(v1-v2))`

143. Compute Euclidean distance between two arrays.
     → `np.linalg.norm(v1-v2)`

144. Implement linear regression using matrix operations.
     → `beta = np.linalg.inv(X.T @ X) @ X.T @ y`

145. Compute polynomial features of a vector.
     → `np.column_stack([x**i for i in range(1,4)])`

146. Perform FFT using `np.fft.fft()`.
     → `np.fft.fft(arr)`

147. Perform inverse FFT using `np.fft.ifft()`.
     → `np.fft.ifft(arr)`

148. Compute correlation of two 1D arrays.
     → `np.corrcoef(x,y)[0,1]`

149. Compute moving average using convolution.
     → `np.convolve(arr, np.ones(3)/3, mode='valid')`

150. Compute weighted average using `np.average()` with weights.
     → `np.average(arr, weights=w)`

151. Use `np.lib.stride_tricks` to create sliding window view.
     → `from numpy.lib.stride_tricks import sliding_window_view; sliding_window_view(arr, 3)`

152. Perform block-wise operations using reshaping.
     → `arr.reshape(-1, block_size).sum(axis=1)`

153. Normalize an array to 0–1 range.
     → `(arr - arr.min()) / (arr.max() - arr.min())`

154. Standardize an array to mean=0, std=1.
     → `(arr - arr.mean()) / arr.std()`

155. Replace NaN values with mean of array.
     → `arr[np.isnan(arr)] = np.nanmean(arr)`

156. Mask NaN values and perform computations.
     → `np.nanmean(arr); np.nansum(arr)`

157. Compute cumulative product along axis in 2D array.
     → `arr.cumprod(axis=1)`

158. Compute rank of elements along axis.
     → `np.argsort(np.argsort(arr, axis=1), axis=1) + 1`

159. Sort array along specified axis.
     → `np.sort(arr, axis=0)`

160. Perform argsort along axis.
     → `np.argsort(arr, axis=1)`

161. Compute percentile along axis.
     → `np.percentile(arr, 90, axis=0)`

162. Compute quantiles.
     → `np.quantile(arr, [0.25,0.5,0.75], axis=0)`

163. Apply function along axis using `np.apply_along_axis()`.
     → `np.apply_along_axis(np.sum, 1, arr)`

164. Compute pairwise distances between rows of a matrix.
     → `np.sqrt(((arr[:,None,:]-arr[None,:,:])**2).sum(axis=2))`

165. Generate random samples from uniform distribution.
     → `np.random.uniform(0,1,10)`

166. Generate random samples from normal distribution with mean/std.
     → `np.random.normal(loc=0, scale=1, size=10)`

167. Generate multivariate normal samples.
     → `np.random.multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]], size=5)`

168. Perform linear algebra eigen decomposition for symmetric matrices.
     → `np.linalg.eigh(mat)`

169. Compute pseudo-inverse of a matrix.
     → `np.linalg.pinv(mat)`

170. Use Kronecker product.
     → `np.kron(A,B)`

171. Perform Hadamard (element-wise) product.
     → `A * B`

172. Reshape array without copying data using `reshape(-1)`.
     → `arr.reshape(-1)`

173. Check memory layout of an array (`C` vs `F`).
     → `arr.flags`

174. Convert array to Fortran-contiguous layout.
     → `np.asfortranarray(arr)`

175. Use `np.mgrid` for dense coordinate grids.
     → `X,Y = np.mgrid[0:3,0:3]`

176. Use `np.ogrid` for memory-efficient grids.
     → `X,Y = np.ogrid[0:3,0:3]`

177. Compute outer product of vectors.
     → `np.outer(v1,v2)`

178. Compute inner product of vectors.
     → `np.inner(v1,v2)`

179. Broadcast scalar operations to higher-dim array.
     → `arr + 5`

180. Create 3D boolean mask and apply to array.
     → `mask = arr>5; arr[mask]`

181. Use `np.take_along_axis()` for advanced indexing.
     → `np.take_along_axis(arr, indices, axis=1)`

182. Use `np.put_along_axis()` to modify elements.
     → `np.put_along_axis(arr, indices, values, axis=1)`

183. Use `np.choose()` for multi-condition selection.
     → `np.choose([0,1,2],[arr1,arr2,arr3])`

184. Implement fast element-wise conditional using `np.where()`.
     → `np.where(arr>0, arr, 0)`

185. Perform block matrix multiplication with einsum.
     → `np.einsum('ij,jk->ik', A, B)`

186. Perform trace of a batched 3D matrix using `einsum`.
     → `np.einsum('...ii', arr)`

187. Use `einsum` for outer product.
     → `np.einsum('i,j->ij', v1,v2)`

188. Compute covariance matrix using `einsum`.
     → `np.einsum('ij,ik->jk', X-X.mean(0), X-X.mean(0))/(X.shape[0]-1)`

189. Compute Gram matrix using `einsum`.
     → `np.einsum('ij,ik->jk', X, X)`

190. Compute pairwise Euclidean distance using `einsum`.
     → `np.sqrt(np.einsum('ij,ij->i', X1-X2, X1-X2))`

191. Vectorize a Python function for arrays.
     → `f = np.vectorize(lambda x: x**2); f(arr)`

192. Optimize array computations using in-place operations.
     → `arr += 5`

193. Minimize memory allocation with `out` parameter in ufuncs.
     → `np.add(arr1, arr2, out=arr1)`

194. Compare performance of list vs NumPy array operations.
     → `%timeit [i**2 for i in range(1000)]; %timeit np.arange(1000)**2`

195. Time NumPy operations with `%timeit`.
     → `%timeit np.dot(A,B)`

196. Use structured arrays for tabular datasets.
     → `dtype=[('name','U10'),('age','i4')]; arr = np.array([('Alice',25)], dtype=dtype)`

197. Load CSV into structured array and manipulate fields.
     → `arr = np.genfromtxt('file.csv', delimiter=',', dtype=dtype); arr['age']`

198. Perform cumulative operations along multiple axes.
     → `arr.cumsum(axis=0); arr.cumprod(axis=1)`

199. Perform masked array operations.
     → `masked = np.ma.masked_where(arr<0, arr); masked.mean()`

200. Combine multiple advanced techniques to implement small data pipeline using NumPy (e.g., generate, normalize, compute statistics, filter, and aggregate).
     → `data = np.random.rand(100,5); norm = (data-data.min(0))/(data.max(0)-data.min(0)); filtered = norm[norm[:,0]>0.5]; stats = filtered.mean(axis=0)`


---

# **Plotly Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Plotly Express, simple plots, figure creation, basic customization*

1. Import `plotly.express` as `px` and print version.
   → `import plotly.express as px; print(px.__version__)`

2. Create a simple line plot of `[1,2,3,4]` vs `[10,15,13,17]`.
   → `px.line(x=[1,2,3,4], y=[10,15,13,17])`

3. Create a scatter plot of `[1,2,3,4]` vs `[10,15,13,17]`.
   → `px.scatter(x=[1,2,3,4], y=[10,15,13,17])`

4. Add a title to a plot.
   → `px.line(x=[1,2,3,4], y=[10,15,13,17], title="My Plot")`

5. Label x-axis and y-axis.
   → `px.line(x=[1,2,3,4], y=[10,15,13,17], labels={'x':'X Axis','y':'Y Axis'})`

6. Change color of markers in scatter plot.
   → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], color=[1,2,1,2])`

7. Change size of markers in scatter plot.
   → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], size=[10,20,15,25])`

8. Use Plotly Express to create a bar chart.
   → `px.bar(x=['A','B','C'], y=[10,20,15])`

9. Change bar colors in a bar chart.
   → `px.bar(x=['A','B','C'], y=[10,20,15], color=['red','green','blue'])`

10. Create a histogram using Plotly Express.
    → `px.histogram([1,2,2,3,3,3,4,4,5])`

11. Adjust number of bins in histogram.
    → `px.histogram([1,2,2,3,3,3,4,4,5], nbins=5)`

12. Create a box plot.
    → `px.box([1,2,3,4,5,2,3,4])`

13. Customize box plot color.
    → `px.box([1,2,3,4,5,2,3,4], color_discrete_sequence=['purple'])`

14. Create a violin plot.
    → `px.violin([1,2,3,4,5,2,3,4])`

15. Customize violin plot color.
    → `px.violin([1,2,3,4,5,2,3,4], color_discrete_sequence=['orange'])`

16. Create an area chart using `line()` with `fill='tozeroy'`.
    → `px.line(x=[1,2,3,4], y=[10,15,13,17], fill='tozeroy')`

17. Create a simple Pie chart.
    → `px.pie(values=[10,20,30])`

18. Add labels and values to Pie chart.
    → `px.pie(names=['A','B','C'], values=[10,20,30])`

19. Customize Pie chart colors.
    → `px.pie(names=['A','B','C'], values=[10,20,30], color_discrete_sequence=['red','green','blue'])`

20. Pull out a slice in Pie chart using `pull`.
    → `px.pie(names=['A','B','C'], values=[10,20,30], pull=[0.1,0,0])`

21. Create a sunburst chart with hierarchical data.
    → `px.sunburst(names=['A','B','C','D'], parents=['','A','A','B'], values=[10,20,5,5])`

22. Create a treemap.
    → `px.treemap(names=['A','B','C','D'], parents=['','A','A','B'], values=[10,20,5,5])`

23. Create a scatter plot with color mapped to a third variable.
    → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], color=[0.1,0.2,0.3,0.4])`

24. Create a scatter plot with size mapped to a variable.
    → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], size=[5,10,15,20])`

25. Combine color and size mapping in scatter plot.
    → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], color=[0.1,0.2,0.3,0.4], size=[5,10,15,20])`

26. Create a scatter plot with categorical color.
    → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], color=['A','B','A','B'])`

27. Create a bar chart with grouped bars using `barmode='group'`.
    → `px.bar(x=['A','A','B','B'], y=[10,15,20,25], color=['X','Y','X','Y'], barmode='group')`

28. Create stacked bar chart using `barmode='stack'`.
    → `px.bar(x=['A','A','B','B'], y=[10,15,20,25], color=['X','Y','X','Y'], barmode='stack')`

29. Reverse y-axis in a bar chart.
    → `fig = px.bar(x=['A','B','C'], y=[10,20,30]); fig.update_yaxes(autorange='reversed')`

30. Reverse x-axis in a bar chart.
    → `fig = px.bar(x=['A','B','C'], y=[10,20,30]); fig.update_xaxes(autorange='reversed')`

31. Add hover text to scatter plot.
    → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], hover_name=['P','Q','R','S'])`

32. Customize hover template.
    → `fig = px.scatter(x=[1,2,3,4], y=[10,15,13,17]); fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y}')`

33. Display multiple traces in one figure using `px.scatter()` with `color` argument.
    → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], color=['A','A','B','B'])`

34. Display multiple traces using `px.line()` with `line_dash` argument.
    → `px.line(x=[1,2,3,4]*2, y=[10,15,13,17]*2, color=['A','A','B','B'], line_dash='color')`

35. Customize marker symbols in scatter plot.
    → `px.scatter(x=[1,2,3,4], y=[10,15,13,17], symbol=['circle','square','diamond','cross'])`

36. Change line style in line plot.
    → `px.line(x=[1,2,3,4], y=[10,15,13,17], line_dash='dash')`

37. Change line width in line plot.
    → `fig = px.line(x=[1,2,3,4], y=[10,15,13,17]); fig.update_traces(line=dict(width=4))`

38. Customize figure size using `width` and `height`.
    → `px.line(x=[1,2,3,4], y=[10,15,13,17], width=800, height=600)`

39. Update layout title font size and family.
    → `fig.update_layout(title_font=dict(size=24,family='Arial'))`

40. Update axis titles font and color.
    → `fig.update_xaxes(title_font=dict(size=16,color='blue')); fig.update_yaxes(title_font=dict(size=16,color='red'))`

41. Update axes range.
    → `fig.update_xaxes(range=[0,5]); fig.update_yaxes(range=[0,20])`

42. Add gridlines to a plot.
    → `fig.update_xaxes(showgrid=True); fig.update_yaxes(showgrid=True)`

43. Remove gridlines.
    → `fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)`

44. Show minor ticks.
    → `fig.update_xaxes(showticklabels=True, minor=dict(ticks='inside'))`

45. Update legend position.
    → `fig.update_layout(legend=dict(x=0.8,y=0.9))`

46. Hide legend.
    → `fig.update_layout(showlegend=False)`

47. Update legend title.
    → `fig.update_layout(legend_title_text='Group')`

48. Customize legend marker size and symbol.
    → `fig.update_traces(marker=dict(size=10,symbol='diamond'))`

49. Export figure as HTML.
    → `fig.write_html('figure.html')`

50. Export figure as static PNG.
    → `fig.write_image('figure.png')`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Plotly Graph Objects, multiple traces, subplots, interactive features*

51. Import `plotly.graph_objects` as `go`.
    → `import plotly.graph_objects as go`

52. Create a line trace using `go.Scatter`.
    → `line_trace = go.Scatter(x=[1,2,3,4], y=[10,15,13,17], mode='lines')`

53. Create a scatter trace using `go.Scatter`.
    → `scatter_trace = go.Scatter(x=[1,2,3,4], y=[10,15,13,17], mode='markers')`

54. Create a bar trace using `go.Bar`.
    → `bar_trace = go.Bar(x=['A','B','C'], y=[10,20,15])`

55. Combine multiple traces in one figure.
    → `fig = go.Figure(data=[line_trace, scatter_trace, bar_trace])`

56. Create a figure using `go.Figure()`.
    → `fig = go.Figure()`

57. Add traces to figure using `add_trace()`.
    → `fig.add_trace(line_trace); fig.add_trace(scatter_trace)`

58. Add a layout title using `update_layout()`.
    → `fig.update_layout(title='My Figure')`

59. Add x-axis and y-axis labels using `update_layout()`.
    → `fig.update_layout(xaxis_title='X Axis', yaxis_title='Y Axis')`

60. Customize x-axis tick format.
    → `fig.update_xaxes(tickformat='%d')`

61. Customize y-axis tick format.
    → `fig.update_yaxes(tickformat='.2f')`

62. Change marker color for one trace.
    → `scatter_trace.marker.color = 'red'`

63. Change line color for one trace.
    → `line_trace.line.color = 'green'`

64. Change marker size for one trace.
    → `scatter_trace.marker.size = 12`

65. Change line width for one trace.
    → `line_trace.line.width = 4`

66. Change line dash style.
    → `line_trace.line.dash = 'dashdot'`

67. Add hover info to trace.
    → `scatter_trace.hoverinfo = 'x+y+text'`

68. Customize hover template for one trace.
    → `scatter_trace.hovertemplate = 'X: %{x}<br>Y: %{y}'`

69. Update legend position in figure layout.
    → `fig.update_layout(legend=dict(x=0.8, y=0.9))`

70. Customize legend font and marker size.
    → `fig.update_layout(legend=dict(font=dict(size=14), traceorder='normal'))`

71. Add annotation to figure.
    → `fig.add_annotation(x=2, y=15, text='Important Point')`

72. Add multiple annotations.
    → `fig.add_annotation(x=2, y=15, text='A'); fig.add_annotation(x=3, y=17, text='B')`

73. Use arrows in annotations.
    → `fig.add_annotation(x=3, y=17, text='Peak', showarrow=True, arrowhead=2)`

74. Update figure background color.
    → `fig.update_layout(plot_bgcolor='lightgray')`

75. Update plot area background color.
    → `fig.update_layout(paper_bgcolor='lightyellow')`

76. Create multiple subplots using `make_subplots()`.
    → `from plotly.subplots import make_subplots; fig = make_subplots(rows=2, cols=2)`

77. Add traces to specific subplot using `row` and `col`.
    → `fig.add_trace(line_trace, row=1, col=1); fig.add_trace(bar_trace, row=1, col=2)`

78. Update subplot titles.
    → `fig.update_layout(title_text='Subplots Example')`

79. Share x-axis among subplots.
    → `fig = make_subplots(rows=2, cols=1, shared_xaxes=True)`

80. Share y-axis among subplots.
    → `fig = make_subplots(rows=2, cols=1, shared_yaxes=True)`

81. Update subplot layout padding using `update_layout()`.
    → `fig.update_layout(margin=dict(l=50,r=50,t=50,b=50))`

82. Combine line and scatter traces in one subplot.
    → `fig.add_trace(line_trace, row=1, col=1); fig.add_trace(scatter_trace, row=1, col=1)`

83. Combine bar and line traces in one subplot.
    → `fig.add_trace(bar_trace, row=1, col=1); fig.add_trace(line_trace, row=1, col=1)`

84. Use secondary y-axis in subplot.
    → `fig = make_subplots(specs=[[{"secondary_y": True}]])`

85. Add multiple traces to secondary y-axis.
    → `fig.add_trace(line_trace, secondary_y=True)`

86. Use `update_xaxes()` to customize one subplot’s x-axis.
    → `fig.update_xaxes(title='Custom X', row=1, col=1)`

87. Use `update_yaxes()` to customize one subplot’s y-axis.
    → `fig.update_yaxes(title='Custom Y', row=1, col=1)`

88. Reverse x-axis in one subplot.
    → `fig.update_xaxes(autorange='reversed', row=1, col=1)`

89. Reverse y-axis in one subplot.
    → `fig.update_yaxes(autorange='reversed', row=1, col=1)`

90. Add vertical line using `go.layout.Shape(type='line')`.
    → `fig.update_layout(shapes=[dict(type='line', x0=2, x1=2, y0=0, y1=20, line=dict(color='red', width=2))])`

91. Add horizontal line using `go.layout.Shape`.
    → `fig.update_layout(shapes=[dict(type='line', x0=0, x1=4, y0=15, y1=15, line=dict(color='blue', width=2))])`

92. Add rectangle shape for highlighting region.
    → `fig.update_layout(shapes=[dict(type='rect', x0=1, x1=2, y0=10, y1=15, fillcolor='yellow', opacity=0.3)])`

93. Add circle shape to highlight point.
    → `fig.update_layout(shapes=[dict(type='circle', x0=2, x1=3, y0=13, y1=14, fillcolor='green', opacity=0.3)])`

94. Add ellipse shape.
    → `fig.update_layout(shapes=[dict(type='ellipse', x0=1, x1=3, y0=10, y1=15, fillcolor='orange', opacity=0.3)])`

95. Add polygon shape.
    → `fig.update_layout(shapes=[dict(type='path', path='M 1 1 L 2 3 L 3 1 Z', fillcolor='purple', opacity=0.3)])`

96. Update shape color and opacity.
    → `fig.update_shapes(dict(fillcolor='red', opacity=0.5))`

97. Add interactive buttons using `updatemenus`.
    → `fig.update_layout(updatemenus=[dict(type='buttons', buttons=[dict(label='Trace 1', method='update', args=[{'visible':[True,False]}])])])`

98. Add dropdown menu to switch traces.
    → `fig.update_layout(updatemenus=[dict(type='dropdown', buttons=[dict(label='Show A', method='update', args=[{'visible':[True,False]}])])])`

99. Add slider to animate plot.
    → `fig.update_layout(sliders=[dict(steps=[dict(method='animate', args=[[f'frame{i}'], dict(mode='immediate')], label=str(i)) for i in range(10)])])`

100. Animate line plot over time using `frame` argument.
     → `fig = go.Figure(data=[go.Scatter(x=[1,2,3], y=[1,2,3])], frames=[go.Frame(data=[go.Scatter(y=[i,j,k])]) for i,j,k in zip(...)] )`


…*(questions 101–130 continue with medium-level interactions: choropleth maps, scatter_3d, line_3d, updating layout dynamically, linking traces, hoverlabel customization, multi-axis annotations, shapes and layers, grouped/stacked bar interactivity, subplots with different types, combined charts, legend customization, responsive layouts)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: 3D plots, maps, animations, advanced interactions, dashboards*

131. Create a 3D scatter plot using `px.scatter_3d()`.
     → `px.scatter_3d(x=[1,2,3], y=[4,5,6], z=[7,8,9])`

132. Customize marker size and color in 3D scatter.
     → `px.scatter_3d(x=[1,2,3], y=[4,5,6], z=[7,8,9], size=[10,20,30], color=[1,2,3])`

133. Add animation frames to 3D scatter.
     → `px.scatter_3d(x=[1,2,3], y=[4,5,6], z=[7,8,9], animation_frame=[0,1,2])`

134. Create a 3D line plot using `go.Scatter3d`.
     → `go.Scatter3d(x=[1,2,3], y=[4,5,6], z=[7,8,9], mode='lines')`

135. Add multiple 3D traces to one figure.
     → `fig = go.Figure(); fig.add_trace(go.Scatter3d(...)); fig.add_trace(go.Scatter3d(...))`

136. Customize 3D axes titles and ranges.
     → `fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', xaxis_range=[0,10]))`

137. Rotate camera view in 3D plot.
     → `fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1)))`

138. Update camera projection type.
     → `fig.update_layout(scene_camera=dict(projection_type='orthographic'))`

139. Create 3D surface plot using `go.Surface`.
     → `go.Surface(z=[[1,2,3],[4,5,6]])`

140. Customize surface colors using colormap.
     → `go.Surface(z=[[1,2,3],[4,5,6]], colorscale='Viridis')`

141. Add contour to 3D surface plot.
     → `go.Surface(z=[[1,2,3],[4,5,6]], contours=dict(z=dict(show=True, color='white')))`

142. Customize lighting on 3D surface.
     → `go.Surface(z=[[1,2,3],[4,5,6]], lighting=dict(ambient=0.5, diffuse=0.8))`

143. Add text annotations to 3D plot.
     → `fig.add_trace(go.Scatter3d(x=[1], y=[2], z=[3], text=['Point'], mode='markers+text'))`

144. Create choropleth map using Plotly Express.
     → `px.choropleth(locations=['USA','CAN'], color=[10,20], locationmode='ISO-3')`

145. Customize colorscale for choropleth.
     → `px.choropleth(locations=['USA','CAN'], color=[10,20], color_continuous_scale='Viridis')`

146. Add hover data to map.
     → `px.choropleth(locations=['USA','CAN'], color=[10,20], hover_data={'Value':[10,20]})`

147. Add map projection options.
     → `px.choropleth(locations=['USA','CAN'], color=[10,20], projection='orthographic')`

148. Create scatter geo map.
     → `px.scatter_geo(lat=[40,50], lon=[-100,-80])`

149. Customize marker size and color on map.
     → `px.scatter_geo(lat=[40,50], lon=[-100,-80], size=[10,20], color=[1,2])`

150. Add multiple traces to geo map.
     → `fig = go.Figure(); fig.add_trace(go.Scattergeo(...)); fig.add_trace(go.Scattergeo(...))`

151. Animate map over time.
     → `px.scatter_geo(lat=[40,50], lon=[-100,-80], animation_frame=[1,2])`

152. Create density map using `px.density_mapbox`.
     → `px.density_mapbox(lat=[40,50], lon=[-100,-80], z=[1,2], radius=10)`

153. Customize mapbox style.
     → `px.density_mapbox(..., mapbox_style='carto-positron')`

154. Add custom mapbox token.
     → `px.set_mapbox_access_token('YOUR_TOKEN')`

155. Create bubble map with multiple sizes.
     → `px.scatter_mapbox(lat=[40,50], lon=[-100,-80], size=[10,20], color=[1,2])`

156. Create parallel coordinates plot using `px.parallel_coordinates`.
     → `px.parallel_coordinates(pd.DataFrame({'A':[1,2],'B':[3,4]}))`

157. Customize line color in parallel coordinates.
     → `px.parallel_coordinates(df, color='A', color_continuous_scale='Viridis')`

158. Add categorical coloring.
     → `px.parallel_coordinates(df, color='Category')`

159. Create sankey diagram using `go.Sankey`.
     → `go.Sankey(node=dict(label=['A','B']), link=dict(source=[0], target=[1], value=[10]))`

160. Customize link colors in sankey.
     → `go.Sankey(node=dict(label=['A','B']), link=dict(source=[0], target=[1], value=[10], color='blue'))`

161. Add node labels and colors.
     → `go.Sankey(node=dict(label=['A','B'], color=['red','green']), link=dict(source=[0], target=[1], value=[10]))`

162. Create sunburst plot using `px.sunburst`.
     → `px.sunburst(names=['A','B','C'], parents=['','A','A'], values=[10,20,5])`

163. Customize color scale in sunburst.
     → `px.sunburst(..., color=[10,20,5], color_continuous_scale='Viridis')`

164. Add hover data to sunburst.
     → `px.sunburst(..., hover_data={'Value':[10,20,5]})`

165. Create treemap using `px.treemap`.
     → `px.treemap(names=['A','B','C'], parents=['','A','A'], values=[10,20,5])`

166. Combine sunburst and treemap in dashboard layout.
     → `# Use subplot grid with make_subplots and add_trace(px.sunburst(), row=1,col=1), add_trace(px.treemap(), row=1,col=2)`

167. Create radar chart (polar plot) using `go.Scatterpolar`.
     → `go.Scatterpolar(r=[1,2,3], theta=['A','B','C'], fill='toself')`

168. Customize radial axes and angular axes.
     → `fig.update_layout(polar=dict(radialaxis=dict(range=[0,5]), angularaxis=dict(direction='clockwise')))`

169. Fill area under polar plot.
     → `go.Scatterpolar(r=[1,2,3], theta=['A','B','C'], fill='toself')`

170. Add multiple polar traces.
     → `fig.add_trace(go.Scatterpolar(r=[1,2,3], theta=['A','B','C'], fill='toself'))`

171. Animate polar plot over time.
     → `px.line_polar(df, r='r', theta='theta', animation_frame='time')`

172. Create funnel chart using `px.funnel`.
     → `px.funnel(x=[100,80,60], y=['Stage1','Stage2','Stage3'])`

173. Customize funnel colors.
     → `px.funnel(..., color=[1,2,3], color_discrete_sequence=['red','green','blue'])`

174. Add multiple funnel traces.
     → `fig.add_trace(go.Funnel(y=['Stage1','Stage2'], x=[50,40]))`

175. Create indicator chart using `go.Indicator`.
     → `go.Indicator(mode='gauge+number', value=75, title={'text':'Speed'})`

176. Update value and delta properties of indicator.
     → `fig.update_traces(value=80, delta={'reference':75})`

177. Combine multiple indicators in one figure.
     → `fig.add_trace(go.Indicator(...)); fig.add_trace(go.Indicator(...))`

178. Use subplot for indicators and charts together.
     → `fig = make_subplots(rows=2, cols=1); fig.add_trace(go.Indicator(...), row=1, col=1); fig.add_trace(go.Scatter(...), row=2, col=1)`

179. Create timeline visualization using Gantt chart (`px.timeline`).
     → `px.timeline(df, x_start='Start', x_end='Finish', y='Task')`

180. Update start and end dates in timeline chart.
     → `fig.update_traces(x=[start_dates, end_dates])`

181. Color-code timeline bars by category.
     → `px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Category')`

182. Animate timeline chart.
     → `px.timeline(df, x_start='Start', x_end='Finish', y='Task', animation_frame='Time')`

183. Add custom hover info to timeline chart.
     → `px.timeline(..., hover_data={'Category':True,'Owner':True})`

184. Create waterfall chart using `px.waterfall`.
     → `px.waterfall(x=['Start','Profit','Loss','End'], y=[100,30,-20,110])`

185. Customize measure type in waterfall.
     → `px.waterfall(..., measure=['absolute','relative','relative','total'])`

186. Combine waterfall chart with bar chart in subplot.
     → `# Use make_subplots and add_trace(px.waterfall(), row=1,col=1), add_trace(px.bar(), row=2,col=1)`

187. Create table visualization using `go.Table`.
     → `go.Table(header=dict(values=['A','B']), cells=dict(values=[[1,2],[3,4]]))`

188. Customize table header colors and fonts.
     → `go.Table(header=dict(values=['A','B'], fill_color='blue', font=dict(color='white', size=14)))`

189. Add multiple tables to one figure.
     → `fig.add_trace(go.Table(...)); fig.add_trace(go.Table(...))`

190. Combine table with chart in dashboard.
     → `# Use make_subplots and add_trace(go.Table(), row=1,col=1), add_trace(go.Scatter(), row=2,col=1)`

191. Use `dash` to embed Plotly figure in interactive app.
     → `import dash; from dash import dcc, html; app = dash.Dash(__name__); app.layout = html.Div([dcc.Graph(figure=fig)])`

192. Add callbacks to update figure based on input.
     → `@app.callback(Output('graph', 'figure'), [Input('dropdown', 'value')])`

193. Update figure layout dynamically using dropdowns.
     → `# Callback updates fig.update_layout(...) based on dropdown value`

194. Update traces dynamically using sliders.
     → `# Callback updates fig.data[i].y based on slider value`

195. Use buttons to toggle traces visibility.
     → `fig.update_layout(updatemenus=[dict(type='buttons', buttons=[dict(label='Show/Hide', method='update', args=[{'visible':[True,False]}])])])`

196. Animate multiple traces over time.
     → `px.line(df, x='x', y=['y1','y2'], animation_frame='time')`

197. Combine choropleth, scatter, and line plots in dashboard.
     → `# Use make_subplots and add_trace(px.choropleth(), row=1,col=1), add_trace(px.scatter(), row=1,col=2), add_trace(px.line(), row=2,col=1)`

198. Add annotations and shapes to dashboard layout.
     → `fig.add_annotation(...); fig.add_shape(...)`

199. Export fully interactive dashboard to HTML.
     → `fig.write_html('dashboard.html')`

200. Build complete Plotly dashboard with multiple charts, maps, 3D plots, interactive filters, and annotations.
     → `# Combine all above techniques using make_subplots, add_trace, update_layout, Dash callbacks, interactive widgets, and export as HTML`


---

# **Seaborn Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, data loading, simple plots, styling, basic customization*

1. Import `seaborn` as `sns` and print version.
   → `import seaborn as sns; print(sns.__version__)`

2. Load built-in dataset `tips` using `sns.load_dataset()`.
   → `tips = sns.load_dataset('tips')`

3. Display the first 5 rows of `tips`.
   → `tips.head()`

4. Create a simple scatter plot with `sns.scatterplot()`.
   → `sns.scatterplot(x='total_bill', y='tip', data=tips)`

5. Create a simple line plot with `sns.lineplot()`.
   → `sns.lineplot(x='total_bill', y='tip', data=tips)`

6. Create a simple bar plot with `sns.barplot()`.
   → `sns.barplot(x='day', y='total_bill', data=tips)`

7. Create a count plot using `sns.countplot()`.
   → `sns.countplot(x='day', data=tips)`

8. Create a box plot with `sns.boxplot()`.
   → `sns.boxplot(x='day', y='total_bill', data=tips)`

9. Create a violin plot with `sns.violinplot()`.
   → `sns.violinplot(x='day', y='total_bill', data=tips)`

10. Create a strip plot with `sns.stripplot()`.
    → `sns.stripplot(x='day', y='total_bill', data=tips)`

11. Create a swarm plot with `sns.swarmplot()`.
    → `sns.swarmplot(x='day', y='total_bill', data=tips)`

12. Add `hue` to scatter plot to color by category.
    → `sns.scatterplot(x='total_bill', y='tip', hue='sex', data=tips)`

13. Add `style` to scatter plot to differentiate markers.
    → `sns.scatterplot(x='total_bill', y='tip', style='smoker', data=tips)`

14. Add `size` to scatter plot for numeric variable.
    → `sns.scatterplot(x='total_bill', y='tip', size='size', data=tips)`

15. Add a title to a Seaborn plot using `plt.title()`.
    → `import matplotlib.pyplot as plt; plt.title('My Plot')`

16. Change x-axis and y-axis labels using `plt.xlabel()` and `plt.ylabel()`.
    → `plt.xlabel('Total Bill'); plt.ylabel('Tip')`

17. Change figure size using `plt.figure(figsize=(width,height))`.
    → `plt.figure(figsize=(10,6))`

18. Set Seaborn style to `'darkgrid'`.
    → `sns.set_style('darkgrid')`

19. Set Seaborn style to `'whitegrid'`.
    → `sns.set_style('whitegrid')`

20. Set Seaborn style to `'ticks'`.
    → `sns.set_style('ticks')`

21. Remove gridlines using `'white'` style.
    → `sns.set_style('white')`

22. Set color palette to `'deep'`.
    → `sns.set_palette('deep')`

23. Set color palette to `'muted'`.
    → `sns.set_palette('muted')`

24. Set color palette to `'bright'`.
    → `sns.set_palette('bright')`

25. Use `sns.set_context('talk')` to adjust figure context.
    → `sns.set_context('talk')`

26. Use `sns.set_context('notebook')`.
    → `sns.set_context('notebook')`

27. Use `sns.set_context('paper')`.
    → `sns.set_context('paper')`

28. Change marker size in scatter plot.
    → `sns.scatterplot(x='total_bill', y='tip', s=100, data=tips)`

29. Change line width in line plot.
    → `sns.lineplot(x='total_bill', y='tip', linewidth=2, data=tips)`

30. Change box plot colors using `palette`.
    → `sns.boxplot(x='day', y='total_bill', palette='Set2', data=tips)`

31. Change violin plot colors using `palette`.
    → `sns.violinplot(x='day', y='total_bill', palette='Set3', data=tips)`

32. Use `hue` in bar plot to show subcategories.
    → `sns.barplot(x='day', y='total_bill', hue='sex', data=tips)`

33. Display numerical aggregation (mean) in bar plot.
    → `sns.barplot(x='day', y='total_bill', estimator=np.mean, data=tips)`

34. Change estimator function in bar plot to `np.sum`.
    → `sns.barplot(x='day', y='total_bill', estimator=np.sum, data=tips)`

35. Create horizontal bar plot by swapping `x` and `y`.
    → `sns.barplot(y='day', x='total_bill', data=tips)`

36. Rotate x-axis tick labels.
    → `plt.xticks(rotation=45)`

37. Adjust y-axis tick labels font size.
    → `plt.yticks(fontsize=12)`

38. Change figure background color.
    → `plt.figure(facecolor='lightgrey')`

39. Change axes background color.
    → `ax = plt.gca(); ax.set_facecolor('lightyellow')`

40. Remove axes spines using `sns.despine()`.
    → `sns.despine()`

41. Remove top and right spines only.
    → `sns.despine(top=True, right=True)`

42. Keep left spine only.
    → `sns.despine(top=True, right=True, bottom=True)`

43. Show gridlines while removing spines.
    → `sns.set_style('whitegrid'); sns.despine()`

44. Add jitter to strip plot.
    → `sns.stripplot(x='day', y='total_bill', jitter=True, data=tips)`

45. Adjust jitter size in strip plot.
    → `sns.stripplot(x='day', y='total_bill', jitter=0.2, data=tips)`

46. Control marker transparency in scatter plot using `alpha`.
    → `sns.scatterplot(x='total_bill', y='tip', alpha=0.5, data=tips)`

47. Display multiple scatter plots in one figure using `plt.subplot()`.
    → `plt.subplot(1,2,1); sns.scatterplot(...); plt.subplot(1,2,2); sns.scatterplot(...)`

48. Save Seaborn figure as PNG.
    → `plt.savefig('figure.png')`

49. Save figure as SVG.
    → `plt.savefig('figure.svg')`

50. Show figure using `plt.show()`.
    → `plt.show()`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Multi-variable visualizations, aggregation, categorical plots, regression*

51. Create a scatter plot with regression line using `sns.regplot()`.
    → `sns.regplot(x='total_bill', y='tip', data=tips)`

52. Fit polynomial regression using `order=2` in `regplot`.
    → `sns.regplot(x='total_bill', y='tip', order=2, data=tips)`

53. Fit lowess regression using `lowess=True`.
    → `sns.regplot(x='total_bill', y='tip', lowess=True, data=tips)`

54. Display confidence interval in regression using `ci=95`.
    → `sns.regplot(x='total_bill', y='tip', ci=95, data=tips)`

55. Remove confidence interval in regression using `ci=None`.
    → `sns.regplot(x='total_bill', y='tip', ci=None, data=tips)`

56. Create residual plot using `sns.residplot()`.
    → `sns.residplot(x='total_bill', y='tip', data=tips)`

57. Create joint plot using `sns.jointplot()` with scatter kind.
    → `sns.jointplot(x='total_bill', y='tip', kind='scatter', data=tips)`

58. Use `kind='hex'` in joint plot.
    → `sns.jointplot(x='total_bill', y='tip', kind='hex', data=tips)`

59. Use `kind='kde'` in joint plot.
    → `sns.jointplot(x='total_bill', y='tip', kind='kde', data=tips)`

60. Use `kind='reg'` in joint plot.
    → `sns.jointplot(x='total_bill', y='tip', kind='reg', data=tips)`

61. Create pair plot using `sns.pairplot()`.
    → `sns.pairplot(tips)`

62. Add `hue` to pair plot.
    → `sns.pairplot(tips, hue='sex')`

63. Choose subset of variables in pair plot using `vars`.
    → `sns.pairplot(tips, vars=['total_bill', 'tip', 'size'])`

64. Drop upper triangle in pair plot.
    → `sns.pairplot(tips, corner=True)`

65. Create categorical box plot using `x` and `y`.
    → `sns.boxplot(x='day', y='total_bill', data=tips)`

66. Use `hue` in categorical box plot.
    → `sns.boxplot(x='day', y='total_bill', hue='sex', data=tips)`

67. Show swarm plot over box plot for distribution.
    → `sns.boxplot(x='day', y='total_bill', data=tips); sns.swarmplot(x='day', y='total_bill', data=tips, color='0.25')`

68. Show strip plot over violin plot for distribution.
    → `sns.violinplot(x='day', y='total_bill', data=tips); sns.stripplot(x='day', y='total_bill', data=tips, color='k', alpha=0.5)`

69. Use `inner='quartile'` in violin plot.
    → `sns.violinplot(x='day', y='total_bill', inner='quartile', data=tips)`

70. Use `inner='stick'` in violin plot.
    → `sns.violinplot(x='day', y='total_bill', inner='stick', data=tips)`

71. Use `split=True` in violin plot for two categories.
    → `sns.violinplot(x='day', y='total_bill', hue='sex', split=True, data=tips)`

72. Aggregate categorical data using `sns.barplot()`.
    → `sns.barplot(x='day', y='total_bill', data=tips)`

73. Customize estimator function to `np.median`.
    → `sns.barplot(x='day', y='total_bill', estimator=np.median, data=tips)`

74. Create count plot with hue for subcategories.
    → `sns.countplot(x='day', hue='sex', data=tips)`

75. Change palette in count plot.
    → `sns.countplot(x='day', data=tips, palette='Set2')`

76. Create factor plot using `sns.catplot()`.
    → `sns.catplot(x='day', y='total_bill', data=tips)`

77. Use `kind='strip'` in catplot.
    → `sns.catplot(x='day', y='total_bill', kind='strip', data=tips)`

78. Use `kind='swarm'` in catplot.
    → `sns.catplot(x='day', y='total_bill', kind='swarm', data=tips)`

79. Use `kind='box'` in catplot.
    → `sns.catplot(x='day', y='total_bill', kind='box', data=tips)`

80. Use `kind='violin'` in catplot.
    → `sns.catplot(x='day', y='total_bill', kind='violin', data=tips)`

81. Use `kind='bar'` in catplot.
    → `sns.catplot(x='day', y='total_bill', kind='bar', data=tips)`

82. Create a facet grid using `sns.FacetGrid()`.
    → `g = sns.FacetGrid(tips, col='sex')`

83. Map scatter plot to facet grid using `map()`.
    → `g.map(sns.scatterplot, 'total_bill', 'tip')`

84. Map histogram to facet grid using `map()`.
    → `g.map(plt.hist, 'total_bill')`

85. Map KDE plot to facet grid using `map()`.
    → `g.map(sns.kdeplot, 'total_bill')`

86. Use `col` in facet grid for column facets.
    → `sns.FacetGrid(tips, col='day')`

87. Use `row` in facet grid for row facets.
    → `sns.FacetGrid(tips, row='sex')`

88. Use `hue` in facet grid.
    → `sns.FacetGrid(tips, col='day', hue='sex')`

89. Control figure size in facet grid.
    → `sns.FacetGrid(tips, col='day', height=4, aspect=1.2)`

90. Adjust spacing between facets.
    → `g = sns.FacetGrid(tips, col='day'); g.fig.subplots_adjust(wspace=0.3)`

91. Add titles to facet grid.
    → `g = sns.FacetGrid(tips, col='day'); g.map(sns.scatterplot, 'total_bill', 'tip'); g.set_titles('Day: {col_name}')`

92. Create a heatmap using `sns.heatmap()`.
    → `sns.heatmap(data=tips.corr())`

93. Customize heatmap colors using `cmap`.
    → `sns.heatmap(data=tips.corr(), cmap='coolwarm')`

94. Annotate heatmap with values using `annot=True`.
    → `sns.heatmap(data=tips.corr(), annot=True)`

95. Format annotations in heatmap.
    → `sns.heatmap(data=tips.corr(), annot=True, fmt=".2f")`

96. Mask upper triangle in heatmap.
    → `mask = np.triu(np.ones_like(tips.corr(), dtype=bool)); sns.heatmap(tips.corr(), mask=mask)`

97. Mask lower triangle in heatmap.
    → `mask = np.tril(np.ones_like(tips.corr(), dtype=bool)); sns.heatmap(tips.corr(), mask=mask)`

98. Control line widths in heatmap.
    → `sns.heatmap(tips.corr(), linewidths=1)`

99. Add color bar to heatmap.
    → `sns.heatmap(tips.corr(), cbar=True)`

100. Remove color bar from heatmap.
     → `sns.heatmap(tips.corr(), cbar=False)`


…*(questions 101–130 continue with medium-level: correlation heatmaps, clustered heatmaps, kde plots, rug plots, multiple regression overlays, facet grids with multiple plot types, swarm/strip overlays, categorical ordering, palette management, axis scaling, normalization, and data transformations)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Complex multi-plot arrangements, advanced customizations, statistical modeling, dashboards*

131. Create clustermap using `sns.clustermap()`.
     → `sns.clustermap(tips.corr())`

132. Customize row and column clustering.
     → `sns.clustermap(tips.corr(), row_cluster=True, col_cluster=False)`

133. Normalize data in clustermap.
     → `sns.clustermap(tips.corr(), standard_scale=1)`

134. Annotate values in clustermap.
     → `sns.clustermap(tips.corr(), annot=True)`

135. Change color palette in clustermap.
     → `sns.clustermap(tips.corr(), cmap='coolwarm')`

136. Adjust dendrogram line width.
     → `sns.clustermap(tips.corr(), linewidths=2)`

137. Create kde plot using `sns.kdeplot()`.
     → `sns.kdeplot(x=tips['total_bill'])`

138. Fill area under KDE curve.
     → `sns.kdeplot(x=tips['total_bill'], fill=True)`

139. Overlay multiple KDE plots.
     → `sns.kdeplot(x=tips['total_bill']); sns.kdeplot(x=tips['tip'])`

140. Use cumulative KDE.
     → `sns.kdeplot(x=tips['total_bill'], cumulative=True)`

141. Change bandwidth in KDE plot.
     → `sns.kdeplot(x=tips['total_bill'], bw_adjust=0.5)`

142. Use rug plot to show individual points.
     → `sns.kdeplot(x=tips['total_bill'], rug=True)`

143. Overlay KDE and rug plot.
     → `sns.kdeplot(x=tips['total_bill'], rug=True, fill=True)`

144. Create 2D KDE plot using `sns.kdeplot()`.
     → `sns.kdeplot(x=tips['total_bill'], y=tips['tip'])`

145. Add contour lines to 2D KDE plot.
     → `sns.kdeplot(x=tips['total_bill'], y=tips['tip'], levels=5)`

146. Fill contours in 2D KDE plot.
     → `sns.kdeplot(x=tips['total_bill'], y=tips['tip'], fill=True)`

147. Use scatter plot overlay on 2D KDE.
     → `sns.kdeplot(x=tips['total_bill'], y=tips['tip'], fill=True); sns.scatterplot(x='total_bill', y='tip', data=tips, color='k')`

148. Create joint plot with 2D KDE kind.
     → `sns.jointplot(x='total_bill', y='tip', kind='kde', data=tips)`

149. Add marginal histograms to joint plot.
     → `sns.jointplot(x='total_bill', y='tip', kind='scatter', marginal_kws=dict(bins=20, fill=True), data=tips)`

150. Add marginal box plots to joint plot.
     → `sns.jointplot(x='total_bill', y='tip', kind='scatter', marginal_ticks=True, data=tips)`

151. Create multiple subplots with seaborn in one figure.
     → `fig, axes = plt.subplots(1,2); sns.scatterplot(x='total_bill', y='tip', data=tips, ax=axes[0]); sns.boxplot(x='day', y='total_bill', data=tips, ax=axes[1])`

152. Combine different plot types in one figure (e.g., box + strip).
     → `sns.boxplot(x='day', y='total_bill', data=tips); sns.stripplot(x='day', y='total_bill', data=tips, color='k', alpha=0.5)`

153. Control figure size using `plt.figure()`.
     → `plt.figure(figsize=(12,6))`

154. Share x-axis across multiple seaborn plots.
     → `fig, axes = plt.subplots(2,1, sharex=True)`

155. Share y-axis across multiple seaborn plots.
     → `fig, axes = plt.subplots(2,1, sharey=True)`

156. Rotate tick labels in complex figure.
     → `plt.xticks(rotation=45); plt.yticks(rotation=30)`

157. Adjust font sizes globally using `sns.set_context()`.
     → `sns.set_context('talk')`

158. Use custom palette across multiple plots.
     → `sns.set_palette('Set2')`

159. Save multiple seaborn figures programmatically.
     → `plt.savefig('figure1.png'); plt.savefig('figure2.png')`

160. Annotate multiple points on scatter plot.
     → `for i, row in tips.iterrows(): plt.text(row['total_bill'], row['tip'], str(i))`

161. Add regression line with multiple categories.
     → `sns.lmplot(x='total_bill', y='tip', hue='sex', data=tips)`

162. Customize confidence intervals for multiple categories.
     → `sns.lmplot(x='total_bill', y='tip', hue='sex', ci=90, data=tips)`

163. Plot interaction effects using `catplot()`.
     → `sns.catplot(x='day', y='total_bill', hue='sex', kind='bar', data=tips)`

164. Plot violin + swarm overlay for multiple subplots.
     → `fig, axes = plt.subplots(1,2); sns.violinplot(x='day', y='total_bill', data=tips, ax=axes[0]); sns.swarmplot(x='day', y='total_bill', data=tips, ax=axes[1])`

165. Plot box + strip overlay with multiple subplots.
     → `fig, axes = plt.subplots(1,2); sns.boxplot(x='day', y='total_bill', data=tips, ax=axes[0]); sns.stripplot(x='day', y='total_bill', data=tips, ax=axes[1], color='k', alpha=0.5)`

166. Customize figure titles for multi-plot figures.
     → `fig.suptitle('Multi-plot Figure', fontsize=16)`

167. Use seaborn with pandas categorical dtype for ordering.
     → `tips['day'] = pd.Categorical(tips['day'], categories=['Thur','Fri','Sat','Sun'], ordered=True)`

168. Sort categories in plots manually.
     → `sns.boxplot(x='day', y='total_bill', data=tips, order=['Thur','Fri','Sat','Sun'])`

169. Reverse category order.
     → `sns.boxplot(x='day', y='total_bill', data=tips, order=['Sun','Sat','Fri','Thur'])`

170. Apply logarithmic scaling to y-axis.
     → `plt.yscale('log')`

171. Apply log scaling to x-axis.
     → `plt.xscale('log')`

172. Create facet grid with custom row and column ordering.
     → `sns.FacetGrid(tips, row='sex', col='day', row_order=['Female','Male'], col_order=['Thur','Fri','Sat','Sun'])`

173. Adjust spacing between facet grid plots.
     → `g = sns.FacetGrid(tips, col='day'); g.fig.subplots_adjust(wspace=0.2, hspace=0.3)`

174. Annotate facet grid with titles.
     → `g.set_titles(col_template='Day: {col_name}')`

175. Overlay multiple plots with hue in facet grid.
     → `g = sns.FacetGrid(tips, col='day', hue='sex'); g.map(sns.scatterplot, 'total_bill', 'tip')`

176. Map different plot types to different facets.
     → `g = sns.FacetGrid(tips, col='day'); g.map(sns.scatterplot, 'total_bill', 'tip'); g.map(sns.kdeplot, 'total_bill')`

177. Use seaborn with large datasets efficiently.
     → `sns.scatterplot(x='total_bill', y='tip', data=tips.sample(200))`

178. Reduce figure complexity with sample data.
     → `sampled = tips.sample(50); sns.pairplot(sampled)`

179. Apply smoothing to line plots.
     → `sns.lineplot(x='total_bill', y='tip', data=tips, ci=None, estimator=None)`

180. Plot confidence intervals with multiple categories.
     → `sns.lineplot(x='total_bill', y='tip', hue='sex', ci=95, data=tips)`

181. Combine line plot + bar plot in one figure.
     → `sns.barplot(x='day', y='total_bill', data=tips); sns.lineplot(x='day', y='tip', data=tips, color='red')`

182. Create heatmap with hierarchical clustering.
     → `sns.clustermap(tips.corr())`

183. Apply masks to heatmaps for selective visualization.
     → `mask = np.triu(np.ones_like(tips.corr(), dtype=bool)); sns.heatmap(tips.corr(), mask=mask)`

184. Customize annotations and formatting in heatmaps.
     → `sns.heatmap(tips.corr(), annot=True, fmt=".2f", cmap='coolwarm')`

185. Use diverging color palettes for heatmaps.
     → `sns.heatmap(tips.corr(), cmap='RdBu_r')`

186. Create multi-index heatmap from pivot table.
     → `pivot = tips.pivot_table(index='day', columns='sex', values='total_bill'); sns.heatmap(pivot)`

187. Combine seaborn plots with matplotlib annotations.
     → `sns.scatterplot(x='total_bill', y='tip', data=tips); plt.text(20, 5, 'Annotation')`

188. Add custom text to figure using `plt.text()`.
     → `plt.text(15, 5, 'Custom Text', fontsize=12, color='red')`

189. Overlay shapes on seaborn plots using `plt.axvline()` or `axhline()`.
     → `plt.axvline(x=20, color='red'); plt.axhline(y=5, color='blue')`

190. Highlight regions in seaborn plots.
     → `plt.axvspan(15, 25, color='yellow', alpha=0.3)`

191. Customize legend across multiple seaborn plots.
     → `plt.legend(title='Gender', loc='upper left')`

192. Move legend to custom position.
     → `plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`

193. Update legend font and style.
     → `plt.legend(fontsize=12, title_fontsize=14)`

194. Combine multiple datasets in one plot using seaborn.
     → `sns.scatterplot(x='total_bill', y='tip', data=tips); sns.scatterplot(x='total_bill', y='tip', data=other_tips, color='red')`

195. Handle missing data in seaborn plots.
     → `sns.scatterplot(x='total_bill', y='tip', data=tips.dropna())`

196. Apply log transformation to numeric variables.
     → `sns.scatterplot(x=np.log(tips['total_bill']), y='tip', data=tips)`

197. Normalize numeric variables before plotting.
     → `sns.scatterplot(x=(tips['total_bill']-tips['total_bill'].mean())/tips['total_bill'].std(), y='tip', data=tips)`

198. Plot multiple regression lines with `lmplot()`.
     → `sns.lmplot(x='total_bill', y='tip', hue='sex', data=tips)`

199. Use `col` and `row` in `lmplot()` for faceting.
     → `sns.lmplot(x='total_bill', y='tip', hue='sex', col='day', row='smoker', data=tips)`

200. Build complete dashboard-like figure using seaborn: multiple plots, facets, overlays, heatmaps, and custom annotations.
     → Combine `plt.subplots()`, `sns.scatterplot()`, `sns.boxplot()`, `sns.heatmap()`, `sns.FacetGrid()`, overlays, and `plt.text()`/`plt.annotate()` in one figure, arranging subplots and customizing titles, axes, colors, and legends.


---

# **NLTK Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, corpora, basic text handling, tokenization*

1. Install NLTK and import it using `import nltk`.
   → `!pip install nltk; import nltk`

2. Check NLTK version.
   → `nltk.__version__`

3. Download the `punkt` tokenizer.
   → `nltk.download('punkt')`

4. Download the `stopwords` corpus.
   → `nltk.download('stopwords')`

5. Download the `wordnet` corpus.
   → `nltk.download('wordnet')`

6. Load a built-in text corpus, e.g., `nltk.corpus.gutenberg`.
   → `from nltk.corpus import gutenberg`

7. Display file IDs in `gutenberg`.
   → `gutenberg.fileids()`

8. Read raw text from one file.
   → `text = gutenberg.raw('austen-emma.txt')`

9. Tokenize text into sentences using `sent_tokenize()`.
   → `from nltk.tokenize import sent_tokenize; sentences = sent_tokenize(text)`

10. Tokenize text into words using `word_tokenize()`.
    → `from nltk.tokenize import word_tokenize; words = word_tokenize(text)`

11. Convert all words to lowercase.
    → `words = [w.lower() for w in words]`

12. Remove punctuation from tokenized words.
    → `import string; words = [w for w in words if w.isalpha()]`

13. Count the number of words in a text.
    → `len(words)`

14. Count the number of sentences in a text.
    → `len(sentences)`

15. Get the first 10 words of a text.
    → `words[:10]`

16. Get the last 10 words of a text.
    → `words[-10:]`

17. Slice tokens from position 10 to 20.
    → `words[10:20]`

18. Create a frequency distribution of words using `FreqDist`.
    → `from nltk import FreqDist; fdist = FreqDist(words)`

19. Find the 10 most common words.
    → `fdist.most_common(10)`

20. Plot the frequency distribution.
    → `fdist.plot(10)`

21. Remove stopwords using `stopwords.words('english')`.
    → `from nltk.corpus import stopwords; stop_words = set(stopwords.words('english')); words = [w for w in words if w not in stop_words]`

22. Compute frequency distribution after removing stopwords.
    → `fdist_no_stop = FreqDist(words)`

23. Filter out words shorter than 3 characters.
    → `words = [w for w in words if len(w) >= 3]`

24. Stem words using `PorterStemmer`.
    → `from nltk.stem import PorterStemmer; ps = PorterStemmer(); stemmed = [ps.stem(w) for w in words]`

25. Stem words using `LancasterStemmer`.
    → `from nltk.stem import LancasterStemmer; ls = LancasterStemmer(); stemmed_l = [ls.stem(w) for w in words]`

26. Lemmatize words using `WordNetLemmatizer`.
    → `from nltk.stem import WordNetLemmatizer; lemmatizer = WordNetLemmatizer(); lemmas = [lemmatizer.lemmatize(w) for w in words]`

27. Compare results of stemming vs lemmatization.
    → `list(zip(stemmed[:10], stemmed_l[:10], lemmas[:10]))`

28. Find synonyms of a word using WordNet.
    → `from nltk.corpus import wordnet; syns = wordnet.synsets('good'); [s.lemma_names() for s in syns]`

29. Find antonyms of a word using WordNet.
    → `antonyms = []; for syn in wordnet.synsets('good'): for l in syn.lemmas(): if l.antonyms(): antonyms.append(l.antonyms()[0].name()); antonyms`

30. Get definitions of a word using WordNet.
    → `[s.definition() for s in wordnet.synsets('good')]`

31. Get part of speech of a word using WordNet.
    → `[s.pos() for s in wordnet.synsets('good')]`

32. Tokenize a paragraph into sentences, then into words.
    → `paragraph = sentences[0]; words_in_para = word_tokenize(paragraph)`

33. Identify vocabulary size of a text (unique words).
    → `len(set(words))`

34. Compute lexical diversity (unique/total words).
    → `len(set(words))/len(words)`

35. Find all words starting with a specific letter.
    → `[w for w in words if w.startswith('t')]`

36. Find all words ending with a specific suffix.
    → `[w for w in words if w.endswith('ing')]`

37. Find all words containing a substring.
    → `[w for w in words if 'ing' in w]`

38. Generate bigrams from a tokenized text.
    → `from nltk import bigrams; list(bigrams(words))[:10]`

39. Generate trigrams from a tokenized text.
    → `from nltk import trigrams; list(trigrams(words))[:10]`

40. Generate n-grams for n=4.
    → `from nltk.util import ngrams; list(ngrams(words, 4))[:10]`

41. Count frequency of bigrams.
    → `fd_bigrams = FreqDist(bigrams(words))`

42. Count frequency of trigrams.
    → `fd_trigrams = FreqDist(trigrams(words))`

43. Create conditional frequency distribution (words by category).
    → `from nltk import ConditionalFreqDist; cfd = ConditionalFreqDist((fileid, w) for fileid in gutenberg.fileids() for w in gutenberg.words(fileid))`

44. Plot conditional frequency distribution.
    → `cfd.plot(['austen-emma.txt', 'austen-persuasion.txt'])`

45. Access raw text from the `inaugural` corpus.
    → `from nltk.corpus import inaugural; inaugural.raw('1789-Washington.txt')`

46. Access words from `inaugural` corpus.
    → `inaugural.words('1789-Washington.txt')`

47. Access sentences from `inaugural` corpus.
    → `inaugural.sents('1789-Washington.txt')`

48. Explore file IDs in `movie_reviews` corpus.
    → `from nltk.corpus import movie_reviews; movie_reviews.fileids()`

49. Access words and categories in `movie_reviews`.
    → `words_pos = movie_reviews.words('pos/cv000_29590.txt'); category = movie_reviews.categories('pos/cv000_29590.txt')`

50. Compute most frequent words per category in `movie_reviews`.
    → `fd_pos = FreqDist(movie_reviews.words(categories='pos')); fd_neg = FreqDist(movie_reviews.words(categories='neg')); fd_pos.most_common(10), fd_neg.most_common(10)`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Tagging, parsing, collocations, concordance, lexical analysis*

51. Tokenize a new text using `word_tokenize()`.
    → `tokens = word_tokenize(new_text)`

52. Apply `nltk.pos_tag()` to the tokenized words.
    → `pos_tags = nltk.pos_tag(tokens)`

53. Identify nouns and verbs in a tagged text.
    → `[word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('VB')]`

54. Count frequency of each POS tag.
    → `from collections import Counter; Counter(tag for word, tag in pos_tags)`

55. Plot frequency distribution of POS tags.
    → `import matplotlib.pyplot as plt; Counter(tag for word, tag in pos_tags).most_common(); plt.bar(*zip(*Counter(tag for word, tag in pos_tags).items())); plt.show()`

56. Extract proper nouns (NNP) from a text.
    → `[word for word, pos in pos_tags if pos == 'NNP']`

57. Extract adjectives (JJ) from a text.
    → `[word for word, pos in pos_tags if pos == 'JJ']`

58. Extract adverbs (RB) from a text.
    → `[word for word, pos in pos_tags if pos.startswith('RB')]`

59. Create a named entity tree using `nltk.ne_chunk()`.
    → `ne_tree = nltk.ne_chunk(pos_tags)`

60. Identify named entities in a text.
    → `[chunk for chunk in ne_tree if hasattr(chunk, 'label')]`

61. Convert named entity tree to list of entities.
    → `[(' '.join(c[0] for c in chunk), chunk.label()) for chunk in ne_tree if hasattr(chunk, 'label')]`

62. Chunk a text using a custom grammar.
    → `grammar = "NP: {<DT>?<JJ>*<NN>}"; cp = nltk.RegexpParser(grammar); tree = cp.parse(pos_tags)`

63. Define a grammar to extract noun phrases.
    → `grammar = "NP: {<DT>?<JJ>*<NN.*>}"`

64. Apply `RegexpParser` with custom grammar.
    → `cp = nltk.RegexpParser(grammar); tree = cp.parse(pos_tags)`

65. Extract all noun phrases from parsed text.
    → `[ ' '.join(w for w, t in subtree.leaves()) for subtree in tree.subtrees() if subtree.label()=='NP']`

66. Extract verb phrases using chunking.
    → `grammar = "VP: {<VB.*><NP|PP|CLAUSE>+$}"; cp = nltk.RegexpParser(grammar); tree = cp.parse(pos_tags)`

67. Perform shallow parsing on a paragraph.
    → `cp = nltk.RegexpParser("NP: {<DT>?<JJ>*<NN.*>}"); tree = cp.parse(pos_tags)`

68. Use `ConcordanceIndex` to find occurrences of a word.
    → `from nltk.text import ConcordanceIndex; ci = ConcordanceIndex(words); ci.offsets('love')`

69. Display concordance for a word in a corpus.
    → `text = nltk.Text(words); text.concordance('love')`

70. Find similar words using `text.similar()`.
    → `text.similar('love')`

71. Find common contexts of words using `text.common_contexts()`.
    → `text.common_contexts(['love','hate'])`

72. Compute dispersion plot of words in a text.
    → `text.dispersion_plot(['love','hate','marriage'])`

73. Use `Collocations` to find common bigrams.
    → `text.collocations()`

74. Find collocations in a text after filtering stopwords.
    → `from nltk.corpus import stopwords; text_filtered = [w for w in words if w not in stopwords.words('english')]; nltk.Text(text_filtered).collocations()`

75. Compute PMI (Pointwise Mutual Information) for bigrams.
    → `from nltk.collocations import BigramCollocationFinder; from nltk.metrics import BigramAssocMeasures; bcf = BigramCollocationFinder.from_words(words); bcf.score_ngrams(BigramAssocMeasures.pmi)`

76. Generate trigrams and compute their frequencies.
    → `from nltk import trigrams; FreqDist(trigrams(words))`

77. Identify hapaxes (words occurring once).
    → `fdist.hapaxes()`

78. Compute average word length in a text.
    → `sum(len(w) for w in words)/len(words)`

79. Compute average sentence length.
    → `sum(len(word_tokenize(s)) for s in sentences)/len(sentences)`

80. Compute readability metrics (basic: words per sentence).
    → `len(words)/len(sentences)`

81. Normalize text (lowercase, remove punctuation).
    → `words_norm = [w.lower() for w in words if w.isalpha()]`

82. Create a custom tokenizer using regex.
    → `from nltk.tokenize import RegexpTokenizer; tokenizer = RegexpTokenizer(r'\w+'); tokens = tokenizer.tokenize(text)`

83. Apply regex tokenizer to extract email addresses.
    → `email_tokenizer = RegexpTokenizer(r'\b[\w.-]+?@\w+?\.\w+?\b'); emails = email_tokenizer.tokenize(text)`

84. Extract hashtags from a social media text.
    → `hashtag_tokenizer = RegexpTokenizer(r'#\w+'); hashtags = hashtag_tokenizer.tokenize(text)`

85. Extract URLs from text using regex tokenizer.
    → `url_tokenizer = RegexpTokenizer(r'http[s]?://\S+'); urls = url_tokenizer.tokenize(text)`

86. Apply tokenization to multiple documents in a loop.
    → `[word_tokenize(doc) for doc in documents]`

87. Build a vocabulary from multiple texts.
    → `vocab = set(word for doc in documents for word in word_tokenize(doc))`

88. Create a frequency distribution across multiple texts.
    → `fd = FreqDist(word for doc in documents for word in word_tokenize(doc))`

89. Filter tokens by frequency threshold.
    → `[w for w in fd if fd[w] >= 5]`

90. Apply lemmatization to filtered tokens.
    → `[lemmatizer.lemmatize(w) for w in filtered_tokens]`

91. Create a POS-tagged frequency distribution.
    → `fd_pos = FreqDist(tag for doc in documents for word, tag in nltk.pos_tag(word_tokenize(doc)))`

92. Compute frequency of verbs across a corpus.
    → `verbs = [word for doc in documents for word, tag in nltk.pos_tag(word_tokenize(doc)) if tag.startswith('VB')]; FreqDist(verbs)`

93. Compute frequency of nouns across a corpus.
    → `nouns = [word for doc in documents for word, tag in nltk.pos_tag(word_tokenize(doc)) if tag.startswith('NN')]; FreqDist(nouns)`

94. Compare lexical diversity of two corpora.
    → `len(set(words1))/len(words1), len(set(words2))/len(words2)`

95. Find common words across multiple corpora.
    → `set(words1).intersection(set(words2))`

96. Find words unique to one corpus.
    → `set(words1) - set(words2)`

97. Extract context windows around a target word.
    → `text.concordance('love', width=80, lines=5)`

98. Compute bigram frequency distribution per category.
    → `from nltk.corpus import movie_reviews; fdist_bi = FreqDist(bigrams(movie_reviews.words(categories='pos')))`

99. Compute trigram frequency distribution per category.
    → `fdist_tri = FreqDist(trigrams(movie_reviews.words(categories='pos')))`

100. Display top 10 collocations per category.
     → `text_pos = nltk.Text(movie_reviews.words(categories='pos')); text_neg = nltk.Text(movie_reviews.words(categories='neg')); text_pos.collocations(); text_neg.collocations()`


…*(questions 101–130 continue with medium-level NLP: custom tagging, regular expression tagging, backoff tagging, training unigram/bigram POS taggers, simple parsing, dependency extraction, chunking with custom grammars, named entity recognition from different corpora, and frequency comparisons across categories)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: NLP pipelines, classifiers, text normalization, advanced corpus handling, feature extraction*

131. Train a unigram POS tagger using `nltk.UnigramTagger()`.
     → `from nltk.tag import UnigramTagger; from nltk.corpus import treebank; train_sents = treebank.tagged_sents()[:3000]; unigram_tagger = UnigramTagger(train_sents)`

132. Train a bigram POS tagger with backoff to unigram.
     → `from nltk.tag import BigramTagger; bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)`

133. Evaluate POS tagger accuracy on test corpus.
     → `test_sents = treebank.tagged_sents()[3000:]; bigram_tagger.evaluate(test_sents)`

134. Tag unknown words and handle unknowns with backoff tagger.
     → `tagged = bigram_tagger.tag(word_tokenize("This is a new sentence with unknownword"))`

135. Create a custom tokenizer for multi-word expressions.
     → `from nltk.tokenize import MWETokenizer; mwe_tokenizer = MWETokenizer([('New','York'),('United','States')])`

136. Tokenize a text preserving multi-word expressions.
     → `mwe_tokenizer.tokenize(word_tokenize("I live in New York and the United States"))`

137. Build a simple sentiment analysis dataset.
     → `data = [("I love this movie", "pos"), ("I hate this movie", "neg")]`

138. Extract features for classification (word presence).
     → `def word_feats(words): return {w: True for w in words}`

139. Extract POS features for classification.
     → `def pos_feats(words): return {tag: True for word, tag in nltk.pos_tag(words)}`

140. Train a Naive Bayes classifier on text data.
     → `from nltk import NaiveBayesClassifier; feats = [(word_feats(word_tokenize(text)), label) for text,label in data]; classifier = NaiveBayesClassifier.train(feats)`

141. Evaluate classifier accuracy.
     → `from nltk.classify import accuracy; accuracy(classifier, feats)`

142. Extract n-grams as features for classification.
     → `from nltk import ngrams; def ngram_feats(words, n=2): return {' '.join(g): True for g in ngrams(words, n)}`

143. Apply feature selection to reduce vocabulary.
     → `from nltk.probability import FreqDist; fdist = FreqDist(w for text,_ in data for w in word_tokenize(text)); top_words = set(w for w,_ in fdist.most_common(1000))`

144. Train classifier using bigram features.
     → `feats = [(ngram_feats(word_tokenize(text),2), label) for text,label in data]; classifier = NaiveBayesClassifier.train(feats)`

145. Train classifier using TF-IDF features (with NLTK + scikit-learn).
     → `from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.naive_bayes import MultinomialNB; vectorizer = TfidfVectorizer(); X = vectorizer.fit_transform([text for text,label in data]); y = [label for text,label in data]; clf = MultinomialNB().fit(X,y)`

146. Classify new sentences using trained model.
     → `clf.predict(vectorizer.transform(["I enjoy this movie"]))`

147. Build a confusion matrix for classifier.
     → `from sklearn.metrics import confusion_matrix; y_pred = clf.predict(X); confusion_matrix(y,y_pred)`

148. Plot confusion matrix.
     → `import seaborn as sns; import matplotlib.pyplot as plt; sns.heatmap(confusion_matrix(y,y_pred), annot=True, fmt='d')`

149. Apply stemming as preprocessing before classification.
     → `from nltk.stem import PorterStemmer; ps = PorterStemmer(); stemmed_words = [ps.stem(w) for w in word_tokenize(text)]`

150. Apply lemmatization before classification.
     → `from nltk.stem import WordNetLemmatizer; lemmatizer = WordNetLemmatizer(); lemmas = [lemmatizer.lemmatize(w) for w in word_tokenize(text)]`

151. Remove stopwords before classification.
     → `from nltk.corpus import stopwords; words_filtered = [w for w in word_tokenize(text) if w not in stopwords.words('english')]`

152. Normalize text before classification.
     → `text = text.lower()`

153. Handle punctuation in text preprocessing.
     → `import string; words = [w for w in word_tokenize(text) if w.isalpha()]`

154. Handle numbers in text preprocessing.
     → `words = [w for w in word_tokenize(text) if not w.isdigit()]`

155. Convert text to lowercase.
     → `text = text.lower()`

156. Apply regex for custom text cleaning.
     → `import re; text = re.sub(r'\d+','', text)`

157. Build corpus from multiple text files.
     → `from nltk.corpus import PlaintextCorpusReader; corpus = PlaintextCorpusReader('data_folder', '.*\.txt')`

158. Build labeled dataset from folder structure.
     → `import os; data = [(open(os.path.join(folder,f)).read(), label) for label in os.listdir('data') for f in os.listdir(os.path.join('data',label))]`

159. Serialize preprocessed text to disk.
     → `import pickle; pickle.dump(data, open('preprocessed.pkl','wb'))`

160. Load serialized corpus.
     → `data = pickle.load(open('preprocessed.pkl','rb'))`

161. Compute TF-IDF manually using NLTK functions.
     → `from nltk.probability import FreqDist; import math; tf = FreqDist(words); idf = {w: math.log(len(documents)/(1+sum(1 for doc in documents if w in doc))) for w in tf}; tfidf = {w: tf[w]*idf[w] for w in tf}`

162. Find top TF-IDF words per document.
     → `sorted(tfidf.items(), key=lambda x:x[1], reverse=True)[:10]`

163. Compute document similarity using cosine similarity.
     → `from sklearn.metrics.pairwise import cosine_similarity; cosine_similarity(X,X)`

164. Identify most similar documents in a corpus.
     → `similarity_scores = cosine_similarity(X,X); most_similar_idx = similarity_scores.argmax()`

165. Build word co-occurrence matrix.
     → `from collections import defaultdict; cooc = defaultdict(lambda: defaultdict(int)); for i, w1 in enumerate(words): for w2 in words[i+1:i+5]: cooc[w1][w2] +=1`

166. Compute PMI for word pairs across corpus.
     → `from nltk.metrics import BigramAssocMeasures; bcf = BigramCollocationFinder.from_words(words); bcf.score_ngrams(BigramAssocMeasures.pmi)`

167. Build word network graph from co-occurrence.
     → `import networkx as nx; G = nx.Graph(); for w1 in cooc: for w2 in cooc[w1]: G.add_edge(w1,w2,weight=cooc[w1][w2]); nx.draw(G, with_labels=True)`

168. Apply collocation measures (PMI, chi-squared) to find strong associations.
     → `bcf.score_ngrams(BigramAssocMeasures.chi_sq)`

169. Train a simple bigram language model.
     → `from nltk import bigrams; from nltk.probability import ConditionalFreqDist; cfd = ConditionalFreqDist(bigrams(words))`

170. Train a trigram language model.
     → `from nltk.util import ngrams; cfd_tri = ConditionalFreqDist((w1+w2,w3) for w1,w2,w3 in ngrams(words,3))`

171. Generate text using trained n-gram model.
     → `word = 'I'; sentence = [word]; for _ in range(10): next_word = max(cfd[word], key=cfd[word].get); sentence.append(next_word); word = next_word; ' '.join(sentence)`

172. Compute perplexity of a trained n-gram model.
     → `import math; perplexity = 2**(-sum(math.log2(cfd[word][next_word]/sum(cfd[word].values())) for word,next_word in bigrams(words))/len(words))`

173. Use `ConditionalFreqDist` for word prediction.
     → `cfd[word].max()`

174. Compute conditional probabilities for next-word prediction.
     → `prob = {w: cfd[word][w]/sum(cfd[word].values()) for w in cfd[word]}`

175. Build a simple chatbot using NLTK.
     → `from nltk.chat.util import Chat; pairs = [(r'hi', ['Hello!']), (r'bye', ['Goodbye!'])]; chat = Chat(pairs); chat.converse()`

176. Use regex patterns for chatbot responses.
     → `pairs = [(r'I need (.*)', ['Why do you need %1?'])]`

177. Extract keywords from user input for chatbot.
     → `keywords = [w for w in word_tokenize(user_input) if w.isalpha() and w not in stopwords.words('english')]`

178. Build context-aware responses using POS tagging.
     → `tags = nltk.pos_tag(word_tokenize(user_input))`

179. Build rule-based named entity extraction.
     → `ne_tree = nltk.ne_chunk(tags)`

180. Extract dates from text using regex and NER.
     → `dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)`

181. Extract locations using NER.
     → `[chunk for chunk in ne_tree if hasattr(chunk,'label') and chunk.label()=='GPE']`

182. Extract organizations using NER.
     → `[chunk for chunk in ne_tree if hasattr(chunk,'label') and chunk.label()=='ORGANIZATION']`

183. Merge multiple corpora into one for analysis.
     → `combined_words = words1 + words2 + words3`

184. Compare word frequency distributions across corpora.
     → `FreqDist(words1).most_common(10), FreqDist(words2).most_common(10)`

185. Compare lexical diversity across corpora.
     → `len(set(words1))/len(words1), len(set(words2))/len(words2)`

186. Visualize frequency distributions using `matplotlib` + NLTK.
     → `fdist = FreqDist(words); fdist.plot(30)`

187. Visualize conditional frequency distributions.
     → `cfd = ConditionalFreqDist((fileid, w) for fileid in gutenberg.fileids() for w in gutenberg.words(fileid)); cfd.plot(['austen-emma.txt','austen-persuasion.txt'])`

188. Build cumulative frequency plots.
     → `fdist = FreqDist(words); fdist.plot(cumulative=True)`

189. Plot Zipf’s law for a corpus.
     → `import numpy as np; ranks = np.arange(1,len(fdist)+1); freqs = np.array(sorted(fdist.values(), reverse=True)); plt.loglog(ranks,freqs); plt.show()`

190. Compute word length distributions.
     → `word_lengths = [len(w) for w in words]; FreqDist(word_lengths).plot()`

191. Compute sentence length distributions.
     → `sent_lengths = [len(word_tokenize(s)) for s in sentences]; FreqDist(sent_lengths).plot()`

192. Compute readability metrics for multiple texts.
     → `[len(word_tokenize(doc))/len(sent_tokenize(doc)) for doc in documents]`

193. Perform topic modeling using NLTK + Gensim (basic integration).
     → `from gensim import corpora, models; texts = [word_tokenize(doc) for doc in documents]; dictionary = corpora.Dictionary(texts); corpus_gensim = [dictionary.doc2bow(text) for text in texts]; lda = models.LdaModel(corpus_gensim, num_topics=5, id2word=dictionary)`

194. Preprocess text for topic modeling (tokenize, remove stopwords, lemmatize).
     → `texts = [[lemmatizer.lemmatize(w) for w in word_tokenize(doc) if w.isalpha() and w not in stopwords.words('english')] for doc in documents]`

195. Build dictionary and corpus for topic modeling.
     → `dictionary = corpora.Dictionary(texts); corpus = [dictionary.doc2bow(text) for text in texts]`

196. Visualize top words per topic.
     → `[lda.show_topic(i, topn=10) for i in range(lda.num_topics)]`

197. Integrate NLTK preprocessing with scikit-learn vectorizers.
     → `from sklearn.feature_extraction.text import CountVectorizer; vectorizer = CountVectorizer(tokenizer=word_tokenize, stop_words='english'); X = vectorizer.fit_transform(documents)`

198. Build end-to-end text classification pipeline.
     → `from sklearn.pipeline import Pipeline; from sklearn.naive_bayes import MultinomialNB; pipeline = Pipeline([('vect', CountVectorizer(tokenizer=word_tokenize)), ('clf', MultinomialNB())]); pipeline.fit(train_texts, train_labels)`

199. Apply pipeline to multiple categories and evaluate performance.
     → `pred = pipeline.predict(test_texts); from sklearn.metrics import classification_report; print(classification_report(test_labels, pred))`

200. Build full NLP workflow: text preprocessing, tokenization, tagging, feature extraction, classification, evaluation, and visualization.
     → Combine steps: preprocessing (lowercase, remove punctuation/stopwords), tokenize, POS tag, extract features (n-grams, TF-IDF), train classifier (Naive Bayes/ML), evaluate (accuracy, confusion matrix, classification report), and visualize results (frequency plots, dispersion plots, Zipf’s law).


---

# **Statsmodels Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, datasets, model basics, OLS regression*

1. Install Statsmodels and import it using `import statsmodels.api as sm`.
   → `!pip install statsmodels; import statsmodels.api as sm`

2. Check Statsmodels version.
   → `sm.__version__`

3. Import `statsmodels.formula.api` as `smf`.
   → `import statsmodels.formula.api as smf`

4. Load built-in dataset `mtcars` (or `dataset = sm.datasets.get_rdataset('mtcars').data`).
   → `dataset = sm.datasets.get_rdataset('mtcars').data`

5. Display first 5 rows of dataset.
   → `dataset.head()`

6. Display dataset summary using `.describe()`.
   → `dataset.describe()`

7. Check dataset column names.
   → `dataset.columns`

8. Check data types of columns.
   → `dataset.dtypes`

9. Plot scatter of `mpg` vs `wt` using matplotlib.
   → `import matplotlib.pyplot as plt; plt.scatter(dataset['wt'], dataset['mpg']); plt.xlabel('wt'); plt.ylabel('mpg'); plt.show()`

10. Plot scatter of `mpg` vs `hp`.
    → `plt.scatter(dataset['hp'], dataset['mpg']); plt.xlabel('hp'); plt.ylabel('mpg'); plt.show()`

11. Fit simple OLS regression: `mpg ~ wt`.
    → `model = smf.ols('mpg ~ wt', data=dataset).fit()`

12. Print regression summary using `.summary()`.
    → `model.summary()`

13. Extract coefficients of the model.
    → `model.params`

14. Extract R-squared of the model.
    → `model.rsquared`

15. Extract adjusted R-squared.
    → `model.rsquared_adj`

16. Extract p-values of coefficients.
    → `model.pvalues`

17. Extract standard errors of coefficients.
    → `model.bse`

18. Predict new values using `.predict()`.
    → `new_data = pd.DataFrame({'wt':[2.5,3.0]}); model.predict(new_data)`

19. Plot fitted line on scatter plot.
    → `plt.scatter(dataset['wt'], dataset['mpg']); plt.plot(dataset['wt'], model.fittedvalues, color='red'); plt.show()`

20. Fit multiple linear regression: `mpg ~ wt + hp`.
    → `model2 = smf.ols('mpg ~ wt + hp', data=dataset).fit()`

21. Add interaction term: `mpg ~ wt * hp`.
    → `model3 = smf.ols('mpg ~ wt * hp', data=dataset).fit()`

22. Fit model using categorical variable: `mpg ~ factor(cyl)`.
    → `model4 = smf.ols('mpg ~ C(cyl)', data=dataset).fit()`

23. Fit model using log transformation: `mpg ~ np.log(wt)`.
    → `import numpy as np; model5 = smf.ols('mpg ~ np.log(wt)', data=dataset).fit()`

24. Fit model using polynomial term: `mpg ~ I(wt**2)`.
    → `model6 = smf.ols('mpg ~ I(wt**2)', data=dataset).fit()`

25. Fit model using formula interface with multiple predictors.
    → `model7 = smf.ols('mpg ~ wt + hp + qsec', data=dataset).fit()`

26. Extract confidence intervals of coefficients.
    → `model.conf_int()`

27. Extract residuals of model.
    → `model.resid`

28. Extract fitted values.
    → `model.fittedvalues`

29. Plot residuals vs fitted values.
    → `plt.scatter(model.fittedvalues, model.resid); plt.xlabel('Fitted'); plt.ylabel('Residuals'); plt.axhline(0, color='red'); plt.show()`

30. Plot histogram of residuals.
    → `plt.hist(model.resid, bins=20); plt.show()`

31. Plot Q-Q plot of residuals using `sm.qqplot()`.
    → `sm.qqplot(model.resid, line='45'); plt.show()`

32. Check assumptions: linearity by residual plot.
    → `plt.scatter(model.fittedvalues, model.resid); plt.axhline(0, color='red'); plt.show()`

33. Check assumptions: normality by Q-Q plot.
    → `sm.qqplot(model.resid, line='45'); plt.show()`

34. Check assumptions: homoscedasticity visually.
    → `plt.scatter(model.fittedvalues, model.resid); plt.axhline(0, color='red'); plt.show()`

35. Calculate variance inflation factor (VIF) manually.
    → `from statsmodels.stats.outliers_influence import variance_inflation_factor; X = sm.add_constant(dataset[['wt','hp']]); [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]`

36. Identify multicollinearity using VIF > 10.
    → `vif = pd.DataFrame({'VIF':[variance_inflation_factor(X.values, i) for i in range(X.shape[1])],'feature':X.columns}); vif[vif['VIF']>10]`

37. Drop predictor and refit model to reduce multicollinearity.
    → `model8 = smf.ols('mpg ~ wt', data=dataset).fit()`

38. Fit weighted least squares (WLS) regression.
    → `model_wls = sm.WLS(dataset['mpg'], sm.add_constant(dataset['wt']), weights=1/dataset['wt']).fit()`

39. Fit robust regression using `RLM`.
    → `model_rlm = sm.RLM(dataset['mpg'], sm.add_constant(dataset['wt'])).fit()`

40. Fit regression with missing data using `dropna()`.
    → `dataset2 = dataset.dropna(); model9 = smf.ols('mpg ~ wt + hp', data=dataset2).fit()`

41. Fit regression with missing data using `fillna()`.
    → `dataset2 = dataset.fillna(dataset.mean()); model10 = smf.ols('mpg ~ wt + hp', data=dataset2).fit()`

42. Use `add_constant()` to include intercept.
    → `X = sm.add_constant(dataset[['wt','hp']]); model11 = sm.OLS(dataset['mpg'], X).fit()`

43. Compare models using AIC.
    → `model.aic, model2.aic`

44. Compare models using BIC.
    → `model.bic, model2.bic`

45. Predict confidence intervals for new data.
    → `pred = model.get_prediction(new_data); pred.conf_int()`

46. Predict prediction intervals for new data.
    → `pred.summary_frame(alpha=0.05)`

47. Create model formula dynamically using string variables.
    → `predictors = ['wt','hp']; formula = 'mpg ~ ' + ' + '.join(predictors); model_dynamic = smf.ols(formula, data=dataset).fit()`

48. Extract influence measures using `get_influence()`.
    → `influence = model.get_influence()`

49. Plot leverage vs residuals.
    → `from statsmodels.graphics.regressionplots import plot_leverage_resid2; plot_leverage_resid2(model); plt.show()`

50. Identify high leverage points using `hat_matrix_diag`.
    → `influence.hat_matrix_diag; high_leverage = influence.hat_matrix_diag > 2*X.shape[1]/len(X)`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: ANOVA, GLM, logistic regression, model diagnostics, categorical data*

51. Fit logistic regression using `smf.logit()`.
    → `logit_model = smf.logit('am ~ wt', data=dataset).fit()`

52. Fit logistic regression with multiple predictors.
    → `logit_model2 = smf.logit('am ~ wt + hp', data=dataset).fit()`

53. Extract predicted probabilities.
    → `logit_model2.predict(dataset[['wt','hp']])`

54. Create classification table from logistic model.
    → `pred_class = (logit_model2.predict(dataset[['wt','hp']]) > 0.5).astype(int)`

55. Compute confusion matrix.
    → `from sklearn.metrics import confusion_matrix; confusion_matrix(dataset['am'], pred_class)`

56. Compute accuracy, precision, recall.
    → `from sklearn.metrics import accuracy_score, precision_score, recall_score; accuracy_score(dataset['am'], pred_class), precision_score(dataset['am'], pred_class), recall_score(dataset['am'], pred_class)`

57. Plot ROC curve using statsmodels or sklearn.
    → `from sklearn.metrics import roc_curve, auc; fpr, tpr, thresholds = roc_curve(dataset['am'], logit_model2.predict(dataset[['wt','hp']])); import matplotlib.pyplot as plt; plt.plot(fpr, tpr); plt.show()`

58. Compute AUC for logistic regression.
    → `auc(fpr, tpr)`

59. Fit Probit model.
    → `probit_model = smf.probit('am ~ wt + hp', data=dataset).fit()`

60. Compare logit vs probit coefficients.
    → `logit_model2.params, probit_model.params`

61. Fit GLM with Gaussian family.
    → `glm_gauss = smf.glm('mpg ~ wt + hp', data=dataset, family=sm.families.Gaussian()).fit()`

62. Fit GLM with Binomial family.
    → `glm_binom = smf.glm('am ~ wt + hp', data=dataset, family=sm.families.Binomial()).fit()`

63. Fit GLM with Poisson family.
    → `glm_pois = smf.glm('mpg ~ wt + hp', data=dataset, family=sm.families.Poisson()).fit()`

64. Fit GLM with Negative Binomial family.
    → `glm_nb = smf.glm('mpg ~ wt + hp', data=dataset, family=sm.families.NegativeBinomial()).fit()`

65. Specify link functions (logit, probit, log).
    → `sm.families.Binomial(link=sm.families.links.probit())`, `sm.families.Binomial(link=sm.families.links.logit())`, `sm.families.Poisson(link=sm.families.links.log())`

66. Extract deviance of GLM.
    → `glm_gauss.deviance`

67. Extract Pearson chi-squared statistic.
    → `glm_gauss.pearson_chi2`

68. Fit GLM with robust covariance.
    → `glm_gauss_robust = smf.glm('mpg ~ wt + hp', data=dataset, family=sm.families.Gaussian()).fit(cov_type='HC3')`

69. Conduct likelihood ratio test for nested models.
    → `sm.stats.anova_lm(model, model2, typ=1)`

70. Perform Wald test for coefficients.
    → `model.wald_test('wt = 0')`

71. Conduct t-test for a single coefficient.
    → `model.t_test('wt = 0')`

72. Conduct F-test for multiple coefficients.
    → `model.f_test('wt = hp = 0')`

73. Fit ANOVA model using formula: `y ~ C(group)`.
    → `anova_model = smf.ols('mpg ~ C(cyl)', data=dataset).fit()`

74. Perform one-way ANOVA.
    → `sm.stats.anova_lm(anova_model, typ=2)`

75. Perform two-way ANOVA.
    → `anova_model2 = smf.ols('mpg ~ C(cyl) + C(gear) + C(cyl):C(gear)', data=dataset).fit(); sm.stats.anova_lm(anova_model2, typ=2)`

76. Extract sum of squares (SS) from ANOVA table.
    → `anova_table = sm.stats.anova_lm(anova_model, typ=2); anova_table['sum_sq']`

77. Extract mean squares (MS) from ANOVA table.
    → `anova_table['mean_sq']`

78. Extract F-statistic and p-value from ANOVA table.
    → `anova_table[['F','PR(>F)']]`

79. Perform post-hoc tests manually using pairwise comparisons.
    → `from statsmodels.stats.multicomp import pairwise_tukeyhsd; tukey = pairwise_tukeyhsd(dataset['mpg'], dataset['cyl'])`

80. Use Bonferroni correction for multiple comparisons.
    → `from statsmodels.stats.multitest import multipletests; multipletests(pvals, method='bonferroni')`

81. Fit repeated measures ANOVA.
    → `import statsmodels.stats.anova as anova; rm_model = smf.mixedlm('mpg ~ wt', dataset, groups=dataset['cyl']).fit()`

82. Fit ANCOVA using covariates.
    → `ancova_model = smf.ols('mpg ~ wt + C(cyl)', data=dataset).fit()`

83. Check assumptions of ANOVA: normality of residuals.
    → `sm.qqplot(ancova_model.resid, line='45'); plt.show()`

84. Check homogeneity of variances.
    → `from scipy.stats import levene; levene(*[dataset[dataset['cyl']==c]['mpg'] for c in dataset['cyl'].unique()])`

85. Plot boxplots for group comparison.
    → `dataset.boxplot(column='mpg', by='cyl'); plt.show()`

86. Fit ordinal regression model.
    → `from statsmodels.miscmodels.ordinal_model import OrderedModel; mod = OrderedModel(dataset['cyl'], dataset[['mpg','wt']], distr='logit'); res = mod.fit()`

87. Fit multinomial logistic regression.
    → `from statsmodels.miscmodels.ordinal_model import MNLogit; mnlogit = sm.MNLogit(dataset['cyl'], sm.add_constant(dataset[['mpg','wt']])).fit()`

88. Predict probabilities for multiple outcome classes.
    → `mnlogit.predict(sm.add_constant(dataset[['mpg','wt']]))`

89. Fit count data regression using Poisson.
    → `poisson_model = smf.glm('gear ~ wt + hp', data=dataset, family=sm.families.Poisson()).fit()`

90. Handle overdispersion using Negative Binomial.
    → `nb_model = smf.glm('gear ~ wt + hp', data=dataset, family=sm.families.NegativeBinomial()).fit()`

91. Fit zero-inflated Poisson model.
    → `from statsmodels.discrete.count_model import ZeroInflatedPoisson; zip_model = ZeroInflatedPoisson(endog=dataset['gear'], exog=sm.add_constant(dataset[['wt','hp']]), exog_infl=sm.add_constant(dataset[['wt']]), inflation='logit').fit()`

92. Fit zero-inflated Negative Binomial model.
    → `from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP; zinb_model = ZeroInflatedNegativeBinomialP(endog=dataset['gear'], exog=sm.add_constant(dataset[['wt','hp']]), exog_infl=sm.add_constant(dataset[['wt']])).fit()`

93. Compare nested models using likelihood ratio test.
    → `lr_stat = 2*(model2.llf - model.llf); from scipy.stats import chi2; p_value = chi2.sf(lr_stat, df=model2.df_model - model.df_model)`

94. Fit mixed effects model using `MixedLM`.
    → `mixed_model = smf.mixedlm('mpg ~ wt', dataset, groups=dataset['cyl']).fit()`

95. Specify random intercept in mixed model.
    → `mixed_model = smf.mixedlm('mpg ~ wt', dataset, groups=dataset['cyl']).fit()`

96. Specify random slope in mixed model.
    → `mixed_model = smf.mixedlm('mpg ~ wt', dataset, groups=dataset['cyl'], re_formula='~wt').fit()`

97. Extract variance components from mixed model.
    → `mixed_model.random_effects`

98. Compute ICC (intra-class correlation).
    → `icc = mixed_model.cov_re.iloc[0,0]/(mixed_model.cov_re.iloc[0,0] + mixed_model.scale)`

99. Fit generalized mixed model using GLM family.
    → `from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM; # Example: gmm = PoissonBayesMixedGLM.from_formula('gear ~ wt + hp', '1|cyl', dataset)`

100. Fit repeated measures mixed model.
     → `repeated_model = smf.mixedlm('mpg ~ wt', dataset, groups=dataset['cyl'], re_formula='~wt').fit()`


…*(questions 101–130 continue with medium-level: model selection, stepwise regression, robust covariance estimators, heteroscedasticity-consistent standard errors, influence measures, Cook’s distance, DFBetas, leverage, correlation matrices, residual diagnostics, partial regression plots, interaction effects, polynomial regression, categorical coding methods, dummy variables, multicollinearity handling)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Time series, ARIMA, VAR, forecasting, advanced statistical models, simulations*

131. Load `AirPassengers` or other time series dataset.
     → `import statsmodels.api as sm; data = sm.datasets.get_rdataset('AirPassengers').data`

132. Plot time series.
     → `import matplotlib.pyplot as plt; plt.plot(data['value']); plt.show()`

133. Decompose time series into trend, seasonal, residual.
     → `from statsmodels.tsa.seasonal import seasonal_decompose; decomposition = seasonal_decompose(data['value'], model='multiplicative'); decomposition.plot(); plt.show()`

134. Fit AR model using `AR` or `AutoReg`.
     → `from statsmodels.tsa.ar_model import AutoReg; ar_model = AutoReg(data['value'], lags=1).fit()`

135. Fit MA model using `ARIMA` with AR=0.
     → `from statsmodels.tsa.arima.model import ARIMA; ma_model = ARIMA(data['value'], order=(0,0,1)).fit()`

136. Fit ARMA model using `ARIMA`.
     → `arma_model = ARIMA(data['value'], order=(1,0,1)).fit()`

137. Fit ARIMA model with p,d,q parameters.
     → `arima_model = ARIMA(data['value'], order=(1,1,1)).fit()`

138. Perform grid search for optimal ARIMA parameters.
     → `# Loop over p,d,q; select model with lowest AIC`

139. Fit seasonal ARIMA (SARIMA).
     → `from statsmodels.tsa.statespace.sarimax import SARIMAX; sarima_model = SARIMAX(data['value'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()`

140. Forecast future values using ARIMA model.
     → `forecast = arima_model.get_forecast(steps=12).predicted_mean`

141. Plot forecast with confidence intervals.
     → `pred_ci = arima_model.get_forecast(steps=12).conf_int(); plt.plot(data['value']); plt.plot(forecast); plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='pink', alpha=0.3); plt.show()`

142. Compute forecast error metrics: MAE, MSE, RMSE.
     → `from sklearn.metrics import mean_absolute_error, mean_squared_error; mae = mean_absolute_error(actual, forecast); mse = mean_squared_error(actual, forecast); rmse = mse**0.5`

143. Perform stationarity test using Augmented Dickey-Fuller.
     → `from statsmodels.tsa.stattools import adfuller; adfuller(data['value'])`

144. Difference non-stationary time series.
     → `diff_data = data['value'].diff().dropna()`

145. Plot ACF and PACF.
     → `from statsmodels.graphics.tsaplots import plot_acf, plot_pacf; plot_acf(data['value']); plot_pacf(data['value']); plt.show()`

146. Fit VAR (Vector Autoregression) for multivariate time series.
     → `from statsmodels.tsa.api import VAR; model = VAR(multivariate_data); model_fitted = model.fit()`

147. Forecast using VAR model.
     → `forecast = model_fitted.forecast(multivariate_data.values[-model_fitted.k_ar:], steps=5)`

148. Compute impulse response functions.
     → `irf = model_fitted.irf(10); irf.plot()`

149. Compute forecast error variance decomposition.
     → `fevd = model_fitted.fevd(10); fevd.summary()`

150. Fit ARIMA with exogenous variables (ARIMAX).
     → `exog_model = ARIMA(data['value'], exog=exog_data, order=(1,1,1)).fit()`

151. Fit GLS (Generalized Least Squares) model.
     → `gls_model = sm.GLS(y, X).fit()`

152. Fit GLS with autocorrelation structure.
     → `glsar_model = sm.GLSAR(y, X, rho=1).iterative_fit()`

153. Fit WLS with heteroscedastic weights.
     → `wls_model = sm.WLS(y, X, weights=1/var).fit()`

154. Fit robust regression using Huber’s T.
     → `rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()`

155. Fit quantile regression.
     → `quant_model = smf.quantreg('y ~ x', df).fit(q=0.5)`

156. Extract conditional quantiles from model.
     → `quant_model.predict(df)`

157. Fit survival regression (Cox Proportional Hazards).
     → `from lifelines import CoxPHFitter; cph = CoxPHFitter(); cph.fit(survival_data, duration_col='T', event_col='E')`

158. Plot survival function from model.
     → `cph.plot()`

159. Fit duration models (Weibull, Exponential).
     → `from lifelines import WeibullFitter, ExponentialFitter; wf = WeibullFitter().fit(durations, event_observed); ef = ExponentialFitter().fit(durations, event_observed)`

160. Fit GARCH model for volatility (via statsmodels.tsa).
     → `from arch import arch_model; garch = arch_model(data['returns']); garch_fit = garch.fit()`

161. Perform Ljung-Box test for autocorrelation.
     → `from statsmodels.stats.diagnostic import acorr_ljungbox; acorr_ljungbox(data['value'], lags=[10])`

162. Compute Durbin-Watson statistic for residuals.
     → `from statsmodels.stats.stattools import durbin_watson; durbin_watson(residuals)`

163. Perform Breusch-Pagan test for heteroscedasticity.
     → `from statsmodels.stats.diagnostic import het_breuschpagan; het_breuschpagan(residuals, X)`

164. Perform White’s test for heteroscedasticity.
     → `from statsmodels.stats.diagnostic import het_white; het_white(residuals, X)`

165. Perform Jarque-Bera test for normality of residuals.
     → `from statsmodels.stats.stattools import jarque_bera; jarque_bera(residuals)`

166. Perform Shapiro-Wilk test for residual normality.
     → `from scipy.stats import shapiro; shapiro(residuals)`

167. Fit logistic regression with interaction terms.
     → `logit_int = smf.logit('am ~ wt * hp', data=dataset).fit()`

168. Fit multinomial logistic regression for multi-class outcomes.
     → `mnlogit = sm.MNLogit(dataset['cyl'], sm.add_constant(dataset[['mpg','wt']])).fit()`

169. Fit Probit mixed effects model.
     → `# Not directly available in statsmodels; can use binomial GLMM via other packages`

170. Fit Poisson mixed effects model.
     → `# Use statsmodels MixedLM with Poisson family workaround or use R/other package`

171. Fit Negative Binomial mixed effects model.
     → `# Similarly, requires GLMM approach not directly in statsmodels`

172. Simulate data for regression analysis.
     → `import numpy as np; X = np.random.randn(100,2); y = 2*X[:,0] + 3*X[:,1] + np.random.randn(100)`

173. Fit regression on simulated data and verify coefficients.
     → `sim_model = sm.OLS(y, sm.add_constant(X)).fit(); sim_model.params`

174. Perform Monte Carlo simulation for model parameters.
     → `# Loop: simulate data, fit model, store coefficients, analyze distribution`

175. Use bootstrap to compute confidence intervals.
     → `# Resample data with replacement, fit model on each sample, compute quantiles of coefficients`

176. Fit panel data regression using `PanelOLS`.
     → `from linearmodels.panel import PanelOLS; panel_model = PanelOLS(y, X, entity_effects=True).fit()`

177. Fit fixed-effects model.
     → `PanelOLS(y, X, entity_effects=True).fit()`

178. Fit random-effects model.
     → `from linearmodels.panel import RandomEffects; re_model = RandomEffects(y, X).fit()`

179. Compare fixed vs random effects using Hausman test.
     → `from linearmodels.panel import compare; compare(fe_model, re_model)`

180. Fit instrumental variable regression using `IV2SLS`.
     → `from linearmodels.iv import IV2SLS; iv_model = IV2SLS(y, X, instrument=Z).fit()`

181. Fit two-stage least squares manually.
     → `# Stage 1: regress X on instruments Z; Stage 2: regress y on predicted X`

182. Compute robust standard errors for IV regression.
     → `iv_model = IV2SLS(y, X, instrument=Z).fit(cov_type='robust')`

183. Conduct hypothesis testing for coefficient equality.
     → `iv_model.f_test('x1 = x2')`

184. Compare nested models using F-test.
     → `sm.stats.anova_lm(model_small, model_large)`

185. Compare non-nested models using AIC/BIC.
     → `model_small.aic, model_large.aic; model_small.bic, model_large.bic`

186. Use Wald test for multiple restrictions.
     → `model.wald_test('x1 = x2 = 0')`

187. Compute likelihood ratio test for nested models.
     → `lr_stat = 2*(model_large.llf - model_small.llf); from scipy.stats import chi2; p_value = chi2.sf(lr_stat, df=model_large.df_model - model_small.df_model)`

188. Conduct simulation study for time series forecasts.
     → `# Simulate multiple series, fit models, compute forecast errors`

189. Perform out-of-sample validation.
     → `train, test = data[:100], data[100:]; model = ARIMA(train, order=(1,1,1)).fit(); forecast = model.forecast(steps=len(test))`

190. Plot predicted vs actual values.
     → `plt.plot(test); plt.plot(forecast); plt.show()`

191. Compute prediction intervals for regression.
     → `pred = model.get_prediction(steps=len(test)); pred.summary_frame(alpha=0.05)`

192. Plot residual diagnostics for time series models.
     → `residuals = model.resid; plt.plot(residuals); plt.show()`

193. Fit exponential smoothing model.
     → `from statsmodels.tsa.holtwinters import SimpleExpSmoothing; ses_model = SimpleExpSmoothing(data['value']).fit()`

194. Fit Holt-Winters additive model.
     → `from statsmodels.tsa.holtwinters import ExponentialSmoothing; hw_add = ExponentialSmoothing(data['value'], seasonal='add', seasonal_periods=12).fit()`

195. Fit Holt-Winters multiplicative model.
     → `hw_mul = ExponentialSmoothing(data['value'], seasonal='mul', seasonal_periods=12).fit()`

196. Forecast using Holt-Winters model.
     → `forecast = hw_mul.forecast(steps=12)`

197. Combine multiple models for ensemble forecasting.
     → `# Average forecasts from ARIMA, Holt-Winters, etc.`

198. Build end-to-end workflow: preprocessing → regression → diagnostics → forecasting.
     → `# Combine steps: clean series, fit ARIMA/HW models, evaluate residuals, forecast, compute accuracy metrics`

199. Automate model selection and evaluation for multiple datasets.
     → `# Loop over datasets, try multiple models, store evaluation metrics, select best`

200. Build full workflow: exploratory analysis, regression, diagnostics, time series forecasting, visualization, and reporting.
     → `# Integrate EDA (plots, decomposition), fit models (ARIMA, GLM, etc.), run diagnostics, forecast, visualize results, generate report`


---

# **LightGBM Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, data handling, simple model training, basic evaluation*

1. Install LightGBM using pip and import `lightgbm` as `lgb`.
   → `!pip install lightgbm` and `import lightgbm as lgb`

2. Check LightGBM version.
   → `lgb.__version__`

3. Load a sample dataset (e.g., sklearn’s `load_boston`).
   → `from sklearn.datasets import load_boston; data = load_boston()`

4. Convert dataset to Pandas DataFrame.
   → `import pandas as pd; df = pd.DataFrame(data.data, columns=data.feature_names)`

5. Split dataset into features (`X`) and target (`y`).
   → `X = df; y = data.target`

6. Split dataset into train and test sets using `train_test_split`.
   → `from sklearn.model_selection import train_test_split; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`

7. Create a LightGBM dataset using `lgb.Dataset()`.
   → `train_data = lgb.Dataset(X_train, label=y_train)`

8. Train a simple LightGBM model using `lgb.train()` with default parameters.
   → `model = lgb.train({}, train_data)`

9. Train a LightGBM classifier for binary classification.
   → `model = lgb.LGBMClassifier().fit(X_train, y_train)`

10. Train a LightGBM regressor for regression.
    → `model = lgb.LGBMRegressor().fit(X_train, y_train)`

11. Print model parameters.
    → `model.get_params()`

12. Make predictions on training data.
    → `y_pred_train = model.predict(X_train)`

13. Make predictions on test data.
    → `y_pred_test = model.predict(X_test)`

14. Evaluate regression using RMSE.
    → `from sklearn.metrics import mean_squared_error; import numpy as np; np.sqrt(mean_squared_error(y_test, y_pred_test))`

15. Evaluate regression using MAE.
    → `from sklearn.metrics import mean_absolute_error; mean_absolute_error(y_test, y_pred_test)`

16. Evaluate classification using accuracy.
    → `from sklearn.metrics import accuracy_score; accuracy_score(y_test, y_pred_test)`

17. Evaluate classification using AUC-ROC.
    → `from sklearn.metrics import roc_auc_score; roc_auc_score(y_test, y_pred_test)`

18. Plot ROC curve.
    → `from sklearn.metrics import roc_curve; import matplotlib.pyplot as plt; fpr, tpr, _ = roc_curve(y_test, y_pred_test); plt.plot(fpr, tpr)`

19. Plot Precision-Recall curve.
    → `from sklearn.metrics import precision_recall_curve; precision, recall, _ = precision_recall_curve(y_test, y_pred_test); plt.plot(recall, precision)`

20. Compute confusion matrix.
    → `from sklearn.metrics import confusion_matrix; confusion_matrix(y_test, y_pred_test)`

21. Plot feature importance using `plot_importance()`.
    → `lgb.plot_importance(model); plt.show()`

22. Extract feature importance values programmatically.
    → `model.feature_importances_`

23. Save trained model to file.
    → `model.save_model('model.txt')`

24. Load model from file.
    → `model = lgb.Booster(model_file='model.txt')`

25. Update model with additional training data.
    → `model.update(train_data)`

26. Use `early_stopping_rounds` for model training.
    → `lgb.train(params, train_data, valid_sets=[valid_data], early_stopping_rounds=10)`

27. Use validation data in `lgb.train()`.
    → `lgb.train(params, train_data, valid_sets=[valid_data])`

28. Set `num_boost_round` manually.
    → `lgb.train(params, train_data, num_boost_round=100)`

29. Set `learning_rate` parameter.
    → `params = {'learning_rate': 0.1}`

30. Set `max_depth` parameter.
    → `params = {'max_depth': 5}`

31. Set `num_leaves` parameter.
    → `params = {'num_leaves': 31}`

32. Set `min_data_in_leaf` parameter.
    → `params = {'min_data_in_leaf': 20}`

33. Set `feature_fraction` parameter.
    → `params = {'feature_fraction': 0.8}`

34. Set `bagging_fraction` parameter.
    → `params = {'bagging_fraction': 0.8}`

35. Set `bagging_freq` parameter.
    → `params = {'bagging_freq': 5}`

36. Set `lambda_l1` parameter.
    → `params = {'lambda_l1': 0.1}`

37. Set `lambda_l2` parameter.
    → `params = {'lambda_l2': 0.1}`

38. Set `objective` for regression.
    → `params = {'objective': 'regression'}`

39. Set `objective` for binary classification.
    → `params = {'objective': 'binary'}`

40. Set `objective` for multiclass classification.
    → `params = {'objective': 'multiclass', 'num_class': 3}`

41. Specify `metric` for regression.
    → `params = {'metric': 'rmse'}`

42. Specify `metric` for classification.
    → `params = {'metric': 'binary_logloss'}`

43. Set categorical features manually.
    → `train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['cat_column'])`

44. Use automatic categorical detection for LightGBM.
    → `lgb.LGBMClassifier(categorical_feature='auto')`

45. Handle missing values automatically.
    → LightGBM automatically handles `NaN` values.

46. Train using GPU (`device='gpu'`).
    → `lgb.LGBMClassifier(device='gpu').fit(X_train, y_train)`

47. Train using CPU (`device='cpu'`).
    → `lgb.LGBMClassifier(device='cpu').fit(X_train, y_train)`

48. Extract number of trees used in model.
    → `model.num_trees()`

49. Visualize individual tree using `plot_tree()`.
    → `lgb.plot_tree(model, tree_index=0); plt.show()`

50. Limit tree depth during visualization.
    → `lgb.plot_tree(model, tree_index=0, max_depth=3); plt.show()`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced model training, hyperparameter tuning, cross-validation, feature engineering*

51. Perform k-fold cross-validation using `lgb.cv()`.
    → `lgb.cv(params, train_data, nfold=5)`

52. Set `nfold=5` in cross-validation.
    → `lgb.cv(params, train_data, nfold=5)`

53. Use early stopping in cross-validation.
    → `lgb.cv(params, train_data, early_stopping_rounds=10)`

54. Use stratified k-fold for classification.
    → `lgb.cv(params, train_data, nfold=5, stratified=True)`

55. Perform grid search manually with `for` loops.
    → Use nested `for` loops over parameter values and track best metric.

56. Use learning rate scheduler.
    → Adjust `learning_rate` parameter dynamically via a function in `lgb.train()`.

57. Use column sampling (`feature_fraction`) in cross-validation.
    → Set `params = {'feature_fraction': 0.8}` in `lgb.cv()`

58. Use row sampling (`bagging_fraction`).
    → Set `params = {'bagging_fraction': 0.8, 'bagging_freq': 1}` in `lgb.cv()`

59. Handle categorical variables properly in CV.
    → Specify `categorical_feature` in `lgb.Dataset()`

60. Extract best number of boosting rounds from CV.
    → `len(cv_results['rmse-mean'])` or `cv_results['best_iteration']`

61. Train model using best number of boosting rounds.
    → `lgb.train(params, train_data, num_boost_round=cv_results['best_iteration'])`

62. Tune `num_leaves` for best performance.
    → Try different `num_leaves` values and track CV metric.

63. Tune `max_depth` for best performance.
    → Try different `max_depth` values in CV and select best metric.

64. Tune `min_data_in_leaf` for best performance.
    → Adjust `min_data_in_leaf` and check CV results.

65. Tune `learning_rate` for best performance.
    → Test multiple `learning_rate` values with CV.

66. Tune `feature_fraction` for best performance.
    → Vary `feature_fraction` parameter in CV.

67. Tune `bagging_fraction` for best performance.
    → Vary `bagging_fraction` parameter in CV.

68. Tune `bagging_freq` for best performance.
    → Adjust `bagging_freq` and evaluate CV score.

69. Tune `lambda_l1` for best performance.
    → Try different `lambda_l1` values and monitor CV metric.

70. Tune `lambda_l2` for best performance.
    → Test multiple `lambda_l2` values with CV.

71. Use randomized search for hyperparameter tuning.
    → Use `RandomizedSearchCV` from sklearn with `LGBMClassifier` or `LGBMRegressor`.

72. Use Bayesian optimization for hyperparameter tuning.
    → Use libraries like `bayes_opt` to maximize CV metric over parameter space.

73. Use Optuna with LightGBM.
    → Define objective function and use `optuna.create_study()` for optimization.

74. Save CV results for later analysis.
    → Store `cv_results` dictionary to file using `pickle` or `json`.

75. Plot CV metrics over boosting rounds.
    → `lgb.plot_metric(cv_results); plt.show()`

76. Visualize feature importance after CV.
    → Train final model and use `lgb.plot_importance(model)`

77. Extract SHAP values for features.
    → `import shap; explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)`

78. Plot SHAP summary plot.
    → `shap.summary_plot(shap_values, X)`

79. Plot SHAP dependence plot.
    → `shap.dependence_plot("feature_name", shap_values, X)`

80. Identify most impactful features using SHAP.
    → Sort `np.abs(shap_values).mean(0)` to rank features.

81. Handle imbalanced dataset by setting `is_unbalance=True`.
    → `params = {'is_unbalance': True}`

82. Handle imbalanced dataset by adjusting `scale_pos_weight`.
    → `params = {'scale_pos_weight': ratio_of_neg_to_pos}`

83. Train multiclass classification model.
    → `model = lgb.LGBMClassifier(objective='multiclass', num_class=3).fit(X_train, y_train)`

84. Evaluate multiclass classification using logloss.
    → `from sklearn.metrics import log_loss; log_loss(y_test, model.predict_proba(X_test))`

85. Compute multiclass AUC.
    → `from sklearn.metrics import roc_auc_score; roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')`

86. Use label encoding for multiclass target.
    → `from sklearn.preprocessing import LabelEncoder; y_encoded = LabelEncoder().fit_transform(y)`

87. Use one-hot encoding for features.
    → `pd.get_dummies(X)`

88. Handle missing values using imputation before training.
    → `from sklearn.impute import SimpleImputer; X_imputed = SimpleImputer().fit_transform(X)`

89. Generate polynomial features for LightGBM.
    → `from sklearn.preprocessing import PolynomialFeatures; X_poly = PolynomialFeatures(degree=2).fit_transform(X)`

90. Generate interaction features.
    → `PolynomialFeatures(degree=2, interaction_only=True)`

91. Use target encoding for categorical variables.
    → Replace categorical column with mean target value per category.

92. Use mean encoding for categorical variables.
    → Same as target encoding: group by category and take mean target.

93. Remove highly correlated features before training.
    → Compute correlation matrix and drop features with high correlation (>0.9).

94. Use PCA for dimensionality reduction.
    → `from sklearn.decomposition import PCA; X_pca = PCA(n_components=5).fit_transform(X)`

95. Use feature selection with `SelectKBest`.
    → `from sklearn.feature_selection import SelectKBest, f_regression; X_new = SelectKBest(f_regression, k=10).fit_transform(X, y)`

96. Create custom metric function.
    → `def custom_metric(y_true, y_pred): return 'name', metric_value, True`

97. Pass custom metric to `lgb.train()`.
    → `lgb.train(params, train_data, feval=custom_metric)`

98. Create custom objective function.
    → `def custom_obj(y_true, y_pred): grad = ...; hess = ...; return grad, hess`

99. Pass custom objective to `lgb.train()`.
    → `lgb.train(params, train_data, fobj=custom_obj)`

100. Implement multi-output regression using LightGBM.
     → Train separate `LGBMRegressor` for each target column or use `MultiOutputRegressor` wrapper from sklearn.


…*(questions 101–130 continue with medium-level: advanced cross-validation strategies, nested CV, time-series CV, feature interaction exploration, early stopping with custom metrics, boosting from scratch, combining LightGBM with pipelines, LightGBM with sklearn wrappers, integrating LightGBM with Pandas and NumPy workflows, memory-efficient training for large datasets)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Ensemble methods, stacking, advanced time series, model interpretation, deployment*

131. Train LightGBM model on large dataset using `Dataset` API.
     → `train_data = lgb.Dataset(large_X, label=large_y); lgb.train(params, train_data)`

132. Use categorical feature handling for large datasets.
     → Specify `categorical_feature` in `lgb.Dataset()` for large data.

133. Use LightGBM with Dask for distributed training.
     → `from dask_ml.xgboost import DaskLGBMClassifier` or `lgb.dask.DaskLGBMClassifier()`

134. Use LightGBM with Spark for distributed datasets.
     → Use `mmlspark.LightGBMClassifier` for Spark MLlib integration.

135. Train model incrementally using `init_model`.
     → `lgb.train(params, train_data, init_model=pretrained_model)`

136. Combine LightGBM with XGBoost in a stacking ensemble.
     → Use sklearn `StackingClassifier` or `StackingRegressor` with both models.

137. Combine LightGBM with CatBoost in stacking.
     → Same as above: include `LGBMClassifier` and `CatBoostClassifier` in stacking.

138. Train model for regression, then use residuals in second LightGBM model.
     → Fit first model, compute residuals, train second model on residuals.

139. Use LightGBM with sklearn `Pipeline`.
     → `from sklearn.pipeline import Pipeline; Pipeline([('lgb', lgb.LGBMClassifier())])`

140. Use LightGBM as part of voting classifier.
     → `from sklearn.ensemble import VotingClassifier; VotingClassifier(estimators=[('lgb', model1), ...])`

141. Use LightGBM in bagging ensemble.
     → Use `BaggingClassifier(base_estimator=LGBMClassifier(), n_estimators=10)`

142. Use LightGBM with cross-validated feature selection.
     → Use `SelectFromModel(LGBMClassifier()).fit(X, y)` with CV.

143. Train multi-step time series forecasting using LightGBM.
     → Create lag features for each future step and train separate models or multi-output regressor.

144. Train recursive time series model using LightGBM.
     → Predict one step ahead, append prediction, repeat for next step.

145. Use lag features for time series prediction.
     → `X['lag1'] = y.shift(1)`

146. Use rolling window features.
     → `X['rolling_mean'] = y.rolling(window=3).mean()`

147. Use expanding window features.
     → `X['expanding_mean'] = y.expanding().mean()`

148. Incorporate calendar features (month, day, weekday) into model.
     → Extract from datetime: `X['month'] = dt.month; X['weekday'] = dt.weekday`

149. Incorporate holidays into model features.
     → Create binary feature for holiday dates.

150. Incorporate trend features for time series.
     → Use linear regression over time to create trend feature.

151. Evaluate time series forecasts using RMSE.
     → `np.sqrt(mean_squared_error(y_true, y_pred))`

152. Evaluate MAPE for time series forecasts.
     → `np.mean(np.abs((y_true - y_pred)/y_true))*100`

153. Evaluate MAE for time series forecasts.
     → `mean_absolute_error(y_true, y_pred)`

154. Plot predicted vs actual time series.
     → `plt.plot(y_true); plt.plot(y_pred)`

155. Plot residuals over time.
     → `plt.plot(y_true - y_pred)`

156. Detect anomalies in residuals.
     → Flag points where `abs(residual) > threshold`.

157. Use LightGBM for ranking tasks.
     → `lgb.LGBMRanker()`

158. Set `objective='lambdarank'` for ranking.
     → `params = {'objective': 'lambdarank'}`

159. Set group parameter for ranking.
     → `train_data = lgb.Dataset(X, label=y, group=group_sizes)`

160. Evaluate ranking model using NDCG.
     → Use `ndcg_score(y_true, y_pred)` from sklearn.

161. Evaluate ranking model using MAP.
     → Compute mean average precision over queries.

162. Visualize ranking predictions.
     → Plot predicted ranks vs true ranks per query.

163. Optimize hyperparameters for ranking.
     → Use CV or Optuna targeting NDCG or MAP.

164. Implement LightGBM with early stopping for ranking.
     → `lgb.train(params, train_data, valid_sets=[valid_data], early_stopping_rounds=10)`

165. Use monotone constraints in LightGBM.
     → `params = {'monotone_constraints': (1, -1, 0)}`

166. Apply monotone constraints for selected features.
     → Set tuple with 1 (increasing), -1 (decreasing), 0 (no constraint).

167. Apply categorical constraints.
     → Specify `categorical_feature` in `lgb.Dataset()`

168. Use custom evaluation function for ranking.
     → Define `feval(y_true, y_pred)` returning `(name, value, is_higher_better)`

169. Train LightGBM on streaming data.
     → Use `init_model` and update with new batch `lgb.train()`

170. Incrementally update LightGBM model with new batches.
     → `lgb.train(params, new_data, init_model=old_model)`

171. Extract leaf indices for training data.
     → `model.predict(X_train, pred_leaf=True)`

172. Use leaf indices for interaction feature engineering.
     → Treat leaf indices as categorical features in second model.

173. Extract split gain for each feature.
     → `model.feature_importance(importance_type='gain')`

174. Identify weak features using gain.
     → Features with low gain values contribute little.

175. Visualize tree structure for advanced interpretation.
     → `lgb.plot_tree(model, tree_index=0); plt.show()`

176. Visualize top k trees.
     → Loop `lgb.plot_tree(model, tree_index=i)` for i in top k indices.

177. Visualize depth of trees.
     → Use `model.max_depth()` per tree or plot tree with `max_depth`.

178. Visualize leaf value distributions.
     → Extract `model.predict(X, pred_leaf=True)` and plot histogram of leaf values.

179. Use SHAP interaction values.
     → `shap.TreeExplainer(model).shap_interaction_values(X)`

180. Visualize SHAP interaction heatmap.
     → `shap.summary_plot(shap_interaction_values, X, plot_type='heatmap')`

181. Deploy LightGBM model with pickle.
     → `import pickle; pickle.dump(model, open('model.pkl', 'wb'))`

182. Deploy LightGBM model using joblib.
     → `import joblib; joblib.dump(model, 'model.joblib')`

183. Deploy LightGBM in a REST API.
     → Wrap model predict call in Flask/FastAPI endpoint.

184. Deploy LightGBM using Flask.
     → `from flask import Flask, request, jsonify; app.route('/predict')`

185. Deploy LightGBM using FastAPI.
     → `from fastapi import FastAPI; app = FastAPI()`

186. Deploy LightGBM using Streamlit.
     → `import streamlit as st; st.button('Predict')`

187. Monitor model performance over time.
     → Track metrics periodically on new data.

188. Retrain model on new data periodically.
     → Schedule retraining with `lgb.train()` on updated dataset.

189. Detect concept drift in streaming data.
     → Monitor distribution change in features or prediction errors.

190. Retrain incrementally on drifted data.
     → Use `init_model` with drifted batch data.

191. Automate hyperparameter tuning pipeline.
     → Combine CV, Optuna/random search, logging, and model selection.

192. Combine LightGBM predictions with other models (stacking/blending).
     → Use sklearn `StackingClassifier` or weighted blending of predictions.

193. Use LightGBM in a Kaggle competition workflow.
     → Standard pipeline: preprocessing, CV, feature engineering, tuning, submission.

194. Use LightGBM with cross-validation folds for robust performance.
     → `lgb.cv(params, train_data, nfold=5, stratified=True)`

195. Combine LightGBM with feature selection for best results.
     → Use `SelectFromModel(LGBMClassifier())` or recursive feature elimination.

196. Interpret SHAP values to explain predictions to stakeholders.
     → Visualize summary plots and dependence plots for top features.

197. Compute global feature importance using SHAP.
     → `np.abs(shap_values).mean(0)`

198. Compute local feature importance for individual predictions.
     → `shap_values[i]` for ith prediction.

199. Visualize model predictions with feature explanations.
     → Use SHAP force plot: `shap.force_plot(explainer.expected_value, shap_values[i], X.iloc[i])`

200. Build full end-to-end workflow: data preprocessing, LightGBM training, hyperparameter tuning, evaluation, interpretation, and deployment.
     → Combine all previous steps sequentially: preprocess data → feature engineering → CV & hyperparameter tuning → train final model → evaluate → interpret with SHAP → deploy via pickle/Flask/FastAPI/Streamlit.


---

# **XGBoost Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, dataset loading, simple model training, basic evaluation*

1. Install XGBoost using pip and import `xgboost` as `xgb`.
   → `!pip install xgboost` and `import xgboost as xgb`

2. Check XGBoost version.
   → `xgb.__version__`

3. Load a sample dataset (e.g., sklearn’s `load_boston`).
   → `from sklearn.datasets import load_boston; data = load_boston()`

4. Convert dataset to Pandas DataFrame.
   → `import pandas as pd; df = pd.DataFrame(data.data, columns=data.feature_names)`

5. Split dataset into features (`X`) and target (`y`).
   → `X = df; y = data.target`

6. Split dataset into train and test sets using `train_test_split`.
   → `from sklearn.model_selection import train_test_split; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`

7. Create DMatrix for training using `xgb.DMatrix()`.
   → `dtrain = xgb.DMatrix(X_train, label=y_train)`

8. Train a simple XGBoost model with default parameters.
   → `model = xgb.train({}, dtrain)`

9. Train an XGBoost classifier for binary classification.
   → `model = xgb.XGBClassifier().fit(X_train, y_train)`

10. Train an XGBoost regressor for regression.
    → `model = xgb.XGBRegressor().fit(X_train, y_train)`

11. Print model parameters.
    → `model.get_params()`

12. Make predictions on training data.
    → `y_pred_train = model.predict(X_train)`

13. Make predictions on test data.
    → `y_pred_test = model.predict(X_test)`

14. Evaluate regression using RMSE.
    → `from sklearn.metrics import mean_squared_error; import numpy as np; np.sqrt(mean_squared_error(y_test, y_pred_test))`

15. Evaluate regression using MAE.
    → `from sklearn.metrics import mean_absolute_error; mean_absolute_error(y_test, y_pred_test)`

16. Evaluate classification using accuracy.
    → `from sklearn.metrics import accuracy_score; accuracy_score(y_test, y_pred_test)`

17. Evaluate classification using AUC-ROC.
    → `from sklearn.metrics import roc_auc_score; roc_auc_score(y_test, y_pred_test)`

18. Plot ROC curve.
    → `from sklearn.metrics import roc_curve; import matplotlib.pyplot as plt; fpr, tpr, _ = roc_curve(y_test, y_pred_test); plt.plot(fpr, tpr)`

19. Plot Precision-Recall curve.
    → `from sklearn.metrics import precision_recall_curve; precision, recall, _ = precision_recall_curve(y_test, y_pred_test); plt.plot(recall, precision)`

20. Compute confusion matrix.
    → `from sklearn.metrics import confusion_matrix; confusion_matrix(y_test, y_pred_test)`

21. Plot feature importance using `plot_importance()`.
    → `xgb.plot_importance(model); plt.show()`

22. Extract feature importance values programmatically.
    → `model.feature_importances_`

23. Save trained model to file using `save_model()`.
    → `model.save_model('xgb_model.json')`

24. Load model from file using `load_model()`.
    → `model.load_model('xgb_model.json')`

25. Update model with additional training data using `xgb.train()`.
    → `xgb.train(params, xgb.DMatrix(new_X, label=new_y), xgb_model=model)`

26. Use `early_stopping_rounds` for model training.
    → `xgb.train(params, dtrain, num_boost_round=100, evals=[(dvalid, 'val')], early_stopping_rounds=10)`

27. Use validation data in `xgb.train()`.
    → `xgb.train(params, dtrain, evals=[(dvalid, 'validation')])`

28. Set `num_boost_round` manually.
    → `xgb.train(params, dtrain, num_boost_round=100)`

29. Set `learning_rate` (eta) parameter.
    → `params = {'eta': 0.1}`

30. Set `max_depth` parameter.
    → `params = {'max_depth': 5}`

31. Set `min_child_weight` parameter.
    → `params = {'min_child_weight': 1}`

32. Set `gamma` parameter for regularization.
    → `params = {'gamma': 0.1}`

33. Set `subsample` parameter.
    → `params = {'subsample': 0.8}`

34. Set `colsample_bytree` parameter.
    → `params = {'colsample_bytree': 0.8}`

35. Set `reg_alpha` parameter.
    → `params = {'reg_alpha': 0.1}`

36. Set `reg_lambda` parameter.
    → `params = {'reg_lambda': 1}`

37. Set `objective` for regression.
    → `params = {'objective': 'reg:squarederror'}`

38. Set `objective` for binary classification.
    → `params = {'objective': 'binary:logistic'}`

39. Set `objective` for multiclass classification.
    → `params = {'objective': 'multi:softprob', 'num_class': 3}`

40. Specify `eval_metric` for regression.
    → `params = {'eval_metric': 'rmse'}`

41. Specify `eval_metric` for classification.
    → `params = {'eval_metric': 'logloss'}`

42. Handle categorical features manually.
    → Convert categorical columns to numeric encoding before training.

43. Handle missing values automatically.
    → XGBoost automatically handles `NaN` values.

44. Train using GPU (`tree_method='gpu_hist'`).
    → `xgb.XGBClassifier(tree_method='gpu_hist').fit(X_train, y_train)`

45. Train using CPU (`tree_method='hist'`).
    → `xgb.XGBClassifier(tree_method='hist').fit(X_train, y_train)`

46. Extract number of trees used in model.
    → `model.get_booster().best_iteration`

47. Visualize individual tree using `plot_tree()`.
    → `xgb.plot_tree(model, num_trees=0); plt.show()`

48. Limit tree depth during visualization.
    → `xgb.plot_tree(model, num_trees=0, rankdir='UT'); plt.show()`

49. Plot multiple trees in one figure.
    → Loop `xgb.plot_tree(model, num_trees=i)` for i in range(k) or use subplots.

50. Extract leaf indices from the model.
    → `model.apply(X)`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced model training, cross-validation, hyperparameter tuning, feature engineering*

51. Perform k-fold cross-validation using `xgb.cv()`.
    → `xgb.cv(params, dtrain, num_boost_round=100, nfold=5)`

52. Set `nfold=5` in cross-validation.
    → `xgb.cv(params, dtrain, num_boost_round=100, nfold=5)`

53. Use early stopping in cross-validation.
    → `xgb.cv(params, dtrain, num_boost_round=100, nfold=5, early_stopping_rounds=10)`

54. Use stratified k-fold for classification.
    → `xgb.cv(params, dtrain, nfold=5, stratified=True)`

55. Perform grid search manually with `for` loops.
    → Loop over parameter values, call `xgb.cv()` for each, track best metric.

56. Use learning rate scheduler.
    → Pass `callbacks=[xgb.callback.LearningRateScheduler(lambda i: 0.1*(0.99**i))]` in `xgb.train()`.

57. Use column sampling (`colsample_bytree`) in cross-validation.
    → `params = {'colsample_bytree': 0.8}` in `xgb.cv()`

58. Use row sampling (`subsample`).
    → `params = {'subsample': 0.8}`

59. Handle categorical variables properly in CV.
    → Encode categories numerically before DMatrix creation.

60. Extract best number of boosting rounds from CV.
    → `cv_results['test-rmse-mean'].idxmin()` or `len(cv_results)`

61. Train model using best number of boosting rounds.
    → `xgb.train(params, dtrain, num_boost_round=cv_results['best_iteration'])`

62. Tune `max_depth` for best performance.
    → Loop over values, evaluate CV metric, pick best.

63. Tune `min_child_weight` for best performance.
    → Vary `min_child_weight` in CV, select best metric.

64. Tune `gamma` for best performance.
    → Vary `gamma` in CV, track test metric.

65. Tune `subsample` for best performance.
    → Try different `subsample` ratios in CV.

66. Tune `colsample_bytree` for best performance.
    → Test various values in CV and select best.

67. Tune `learning_rate` (eta) for best performance.
    → Try multiple `eta` values in CV and pick best iteration.

68. Tune `reg_alpha` for best performance.
    → Vary `reg_alpha` in CV and check metric improvement.

69. Tune `reg_lambda` for best performance.
    → Vary `reg_lambda` in CV and monitor performance.

70. Use randomized search for hyperparameter tuning.
    → Use `RandomizedSearchCV` with `XGBClassifier` or `XGBRegressor`.

71. Use Bayesian optimization for hyperparameter tuning.
    → Use `bayes_opt` library to optimize CV metric.

72. Use Optuna with XGBoost.
    → Define objective function and run `optuna.create_study()` to optimize hyperparameters.

73. Save CV results for later analysis.
    → Save `cv_results` using `pickle` or `json`.

74. Plot CV metrics over boosting rounds.
    → `xgb.plot_metric(cv_results); plt.show()`

75. Visualize feature importance after CV.
    → Train final model and use `xgb.plot_importance(model)`

76. Extract SHAP values for features.
    → `import shap; explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)`

77. Plot SHAP summary plot.
    → `shap.summary_plot(shap_values, X)`

78. Plot SHAP dependence plot.
    → `shap.dependence_plot("feature_name", shap_values, X)`

79. Identify most impactful features using SHAP.
    → Sort `np.abs(shap_values).mean(0)` to rank features.

80. Handle imbalanced dataset by setting `scale_pos_weight`.
    → `params = {'scale_pos_weight': ratio_of_neg_to_pos}`

81. Train multiclass classification model.
    → `model = xgb.XGBClassifier(objective='multi:softprob', num_class=3).fit(X_train, y_train)`

82. Evaluate multiclass classification using logloss.
    → `from sklearn.metrics import log_loss; log_loss(y_test, model.predict_proba(X_test))`

83. Compute multiclass AUC.
    → `roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')`

84. Use label encoding for multiclass target.
    → `from sklearn.preprocessing import LabelEncoder; y_encoded = LabelEncoder().fit_transform(y)`

85. Use one-hot encoding for features.
    → `pd.get_dummies(X)`

86. Handle missing values using imputation before training.
    → `from sklearn.impute import SimpleImputer; X_imputed = SimpleImputer().fit_transform(X)`

87. Generate polynomial features.
    → `from sklearn.preprocessing import PolynomialFeatures; X_poly = PolynomialFeatures(degree=2).fit_transform(X)`

88. Generate interaction features.
    → `PolynomialFeatures(degree=2, interaction_only=True)`

89. Use target encoding for categorical variables.
    → Replace categorical column with mean target per category.

90. Use mean encoding for categorical variables.
    → Same as target encoding: group by category and compute mean target.

91. Remove highly correlated features before training.
    → Compute correlation matrix and drop features with correlation > 0.9.

92. Use PCA for dimensionality reduction.
    → `from sklearn.decomposition import PCA; X_pca = PCA(n_components=5).fit_transform(X)`

93. Use feature selection with `SelectKBest`.
    → `from sklearn.feature_selection import SelectKBest, f_regression; X_new = SelectKBest(f_regression, k=10).fit_transform(X, y)`

94. Create custom evaluation metric.
    → `def custom_metric(y_pred, dtrain): return 'metric_name', metric_value`

95. Pass custom metric to `xgb.train()`.
    → `xgb.train(params, dtrain, feval=custom_metric)`

96. Create custom objective function.
    → `def custom_obj(y_pred, dtrain): grad = ...; hess = ...; return grad, hess`

97. Pass custom objective to `xgb.train()`.
    → `xgb.train(params, dtrain, obj=custom_obj)`

98. Implement multi-output regression using XGBoost.
    → Use `MultiOutputRegressor(XGBRegressor()).fit(X, y_multi)`

99. Train a regressor with monotone constraints.
    → `xgb.XGBRegressor(monotone_constraints=(1,0,-1)).fit(X_train, y_train)`

100. Visualize partial dependence plot for a feature.
     → `from sklearn.inspection import PartialDependenceDisplay; PartialDependenceDisplay.from_estimator(model, X, ['feature_name'])`


…*(questions 101–130 continue with medium-level: advanced CV strategies, nested CV, time-series CV, feature interaction exploration, early stopping with custom metrics, XGBoost pipelines with sklearn, memory-efficient training, handling sparse matrices, categorical encoding, regularization tuning, incremental training, early stopping diagnostics)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Ensemble methods, stacking, advanced time series, model interpretation, deployment*

131. Train XGBoost model on large dataset using `DMatrix` API.
     → `dtrain = xgb.DMatrix(large_X, label=large_y); xgb.train(params, dtrain)`

132. Use categorical feature handling for large datasets.
     → Encode categorical columns numerically before DMatrix creation.

133. Use XGBoost with Dask for distributed training.
     → `from dask_ml.xgboost import DaskXGBClassifier; model = DaskXGBClassifier().fit(X_dask, y_dask)`

134. Use XGBoost with Spark for distributed datasets.
     → Use `xgboost.spark.XGBoostClassifier()` for Spark MLlib integration.

135. Train model incrementally using `xgb.train()` with `xgb_model`.
     → `xgb.train(params, dtrain_new, xgb_model=old_model)`

136. Combine XGBoost with LightGBM in a stacking ensemble.
     → Use sklearn `StackingClassifier` or `StackingRegressor` with both models.

137. Combine XGBoost with CatBoost in stacking.
     → Include `XGBClassifier` and `CatBoostClassifier` in a `StackingClassifier`.

138. Train model for regression, then use residuals in second XGBoost model.
     → Fit first model, compute residuals, train second model on residuals.

139. Use XGBoost with sklearn `Pipeline`.
     → `from sklearn.pipeline import Pipeline; Pipeline([('xgb', xgb.XGBClassifier())])`

140. Use XGBoost as part of voting classifier.
     → `from sklearn.ensemble import VotingClassifier; VotingClassifier(estimators=[('xgb', model1), ...])`

141. Use XGBoost in bagging ensemble.
     → `from sklearn.ensemble import BaggingClassifier; BaggingClassifier(base_estimator=XGBClassifier(), n_estimators=10)`

142. Use XGBoost with cross-validated feature selection.
     → `from sklearn.feature_selection import SelectFromModel; SelectFromModel(XGBClassifier()).fit(X, y)`

143. Train multi-step time series forecasting using XGBoost.
     → Create lag features for each future step and train separate models or multi-output regressor.

144. Train recursive time series model using XGBoost.
     → Predict one step ahead, append prediction, repeat for next step.

145. Use lag features for time series prediction.
     → `X['lag1'] = y.shift(1)`

146. Use rolling window features.
     → `X['rolling_mean'] = y.rolling(window=3).mean()`

147. Use expanding window features.
     → `X['expanding_mean'] = y.expanding().mean()`

148. Incorporate calendar features (month, day, weekday).
     → Extract from datetime: `X['month'] = dt.month; X['weekday'] = dt.weekday`

149. Incorporate holidays into model features.
     → Create binary feature for holiday dates.

150. Incorporate trend features for time series.
     → Compute linear trend feature over time.

151. Evaluate time series forecasts using RMSE.
     → `np.sqrt(mean_squared_error(y_true, y_pred))`

152. Evaluate MAPE for time series forecasts.
     → `np.mean(np.abs((y_true - y_pred)/y_true))*100`

153. Evaluate MAE for time series forecasts.
     → `mean_absolute_error(y_true, y_pred)`

154. Plot predicted vs actual time series.
     → `plt.plot(y_true); plt.plot(y_pred)`

155. Plot residuals over time.
     → `plt.plot(y_true - y_pred)`

156. Detect anomalies in residuals.
     → Flag points where `abs(residual) > threshold`.

157. Use XGBoost for ranking tasks.
     → `xgb.XGBRanker()`

158. Set `objective='rank:pairwise'` for ranking.
     → `params = {'objective': 'rank:pairwise'}`

159. Set group parameter for ranking.
     → `dtrain = xgb.DMatrix(X, label=y, group=group_sizes)`

160. Evaluate ranking model using NDCG.
     → Use `sklearn.metrics.ndcg_score(y_true, y_pred)`

161. Evaluate ranking model using MAP.
     → Compute mean average precision over queries.

162. Visualize ranking predictions.
     → Plot predicted ranks vs true ranks per query.

163. Optimize hyperparameters for ranking.
     → Use CV or Optuna targeting NDCG or MAP.

164. Implement XGBoost with early stopping for ranking.
     → `xgb.train(params, dtrain, evals=[(dvalid, 'val')], early_stopping_rounds=10)`

165. Use monotone constraints in XGBoost.
     → `params = {'monotone_constraints': (1, -1, 0)}`

166. Apply monotone constraints for selected features.
     → Set tuple with 1 (increasing), -1 (decreasing), 0 (no constraint).

167. Apply categorical constraints.
     → Encode categorical features numerically before training.

168. Use custom evaluation function for ranking.
     → Define `feval(y_pred, dtrain)` returning `(name, value, is_higher_better)`

169. Train XGBoost on streaming data.
     → Use `xgb.train()` with `xgb_model` to update incrementally.

170. Incrementally update XGBoost model with new batches.
     → `xgb.train(params, dtrain_new, xgb_model=old_model)`

171. Extract leaf indices for training data.
     → `model.apply(X_train)`

172. Use leaf indices for interaction feature engineering.
     → Treat leaf indices as categorical features in second model.

173. Extract split gain for each feature.
     → `model.get_booster().get_score(importance_type='gain')`

174. Identify weak features using gain.
     → Features with low gain values contribute little to model.

175. Visualize tree structure for advanced interpretation.
     → `xgb.plot_tree(model, num_trees=0); plt.show()`

176. Visualize top k trees.
     → Loop `xgb.plot_tree(model, num_trees=i)` for i in top k indices.

177. Visualize depth of trees.
     → Use `plot_tree` with depth limit or inspect tree attributes.

178. Visualize leaf value distributions.
     → Extract leaf indices with `model.apply(X)` and plot histogram.

179. Use SHAP interaction values.
     → `shap.TreeExplainer(model).shap_interaction_values(X)`

180. Visualize SHAP interaction heatmap.
     → `shap.summary_plot(shap_interaction_values, X, plot_type='heatmap')`

181. Deploy XGBoost model with pickle.
     → `import pickle; pickle.dump(model, open('model.pkl', 'wb'))`

182. Deploy XGBoost model using joblib.
     → `import joblib; joblib.dump(model, 'model.joblib')`

183. Deploy XGBoost in a REST API.
     → Wrap model predict call in Flask/FastAPI endpoint.

184. Deploy XGBoost using Flask.
     → `from flask import Flask, request, jsonify; @app.route('/predict')`

185. Deploy XGBoost using FastAPI.
     → `from fastapi import FastAPI; app = FastAPI()`

186. Deploy XGBoost using Streamlit.
     → `import streamlit as st; st.button('Predict')`

187. Monitor model performance over time.
     → Track metrics periodically on new data.

188. Retrain model on new data periodically.
     → Schedule retraining using `xgb.train()` on updated dataset.

189. Detect concept drift in streaming data.
     → Monitor feature distribution changes or prediction errors.

190. Retrain incrementally on drifted data.
     → Use `xgb.train(params, dtrain_new, xgb_model=old_model)`

191. Automate hyperparameter tuning pipeline.
     → Combine CV, Optuna/random search, logging, and model selection.

192. Combine XGBoost predictions with other models (stacking/blending).
     → Use sklearn `StackingClassifier` or weighted blending of predictions.

193. Use XGBoost in a Kaggle competition workflow.
     → Standard pipeline: preprocessing → CV → feature engineering → tuning → submission.

194. Use XGBoost with cross-validation folds for robust performance.
     → `xgb.cv(params, dtrain, nfold=5, stratified=True)`

195. Combine XGBoost with feature selection for best results.
     → Use `SelectFromModel(XGBClassifier())` or recursive feature elimination.

196. Interpret SHAP values to explain predictions to stakeholders.
     → Visualize summary and dependence plots for top features.

197. Compute global feature importance using SHAP.
     → `np.abs(shap_values).mean(0)`

198. Compute local feature importance for individual predictions.
     → `shap_values[i]` for ith prediction.

199. Visualize model predictions with feature explanations.
     → `shap.force_plot(explainer.expected_value, shap_values[i], X.iloc[i])`

200. Build full end-to-end workflow: data preprocessing, XGBoost training, hyperparameter tuning, evaluation, interpretation, and deployment.
     → Combine all previous steps sequentially: preprocess → feature engineering → CV & tuning → train final model → evaluate → interpret with SHAP → deploy via pickle/Flask/FastAPI/Streamlit.


---

# **CatBoost Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, data handling, simple model training, basic evaluation*

1. Install CatBoost using pip and import `catboost`.
   → `!pip install catboost` and `import catboost`

2. Check CatBoost version.
   → `catboost.__version__`

3. Load a sample dataset (e.g., sklearn’s `load_boston`).
   → `from sklearn.datasets import load_boston; data = load_boston()`

4. Convert dataset to Pandas DataFrame.
   → `import pandas as pd; df = pd.DataFrame(data.data, columns=data.feature_names)`

5. Split dataset into features (`X`) and target (`y`).
   → `X = df; y = data.target`

6. Split dataset into train and test sets using `train_test_split`.
   → `from sklearn.model_selection import train_test_split; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`

7. Initialize CatBoostRegressor with default parameters.
   → `from catboost import CatBoostRegressor; model = CatBoostRegressor()`

8. Initialize CatBoostClassifier with default parameters.
   → `from catboost import CatBoostClassifier; model = CatBoostClassifier()`

9. Train a simple CatBoost model on training data.
   → `model.fit(X_train, y_train)`

10. Make predictions on training data.
    → `y_pred_train = model.predict(X_train)`

11. Make predictions on test data.
    → `y_pred_test = model.predict(X_test)`

12. Evaluate regression using RMSE.
    → `from sklearn.metrics import mean_squared_error; import numpy as np; np.sqrt(mean_squared_error(y_test, y_pred_test))`

13. Evaluate regression using MAE.
    → `from sklearn.metrics import mean_absolute_error; mean_absolute_error(y_test, y_pred_test)`

14. Evaluate classification using accuracy.
    → `from sklearn.metrics import accuracy_score; accuracy_score(y_test, y_pred_test)`

15. Evaluate classification using AUC-ROC.
    → `from sklearn.metrics import roc_auc_score; roc_auc_score(y_test, y_pred_test)`

16. Plot ROC curve for classifier.
    → `from sklearn.metrics import roc_curve; import matplotlib.pyplot as plt; fpr, tpr, _ = roc_curve(y_test, y_pred_test); plt.plot(fpr, tpr)`

17. Plot Precision-Recall curve.
    → `from sklearn.metrics import precision_recall_curve; precision, recall, _ = precision_recall_curve(y_test, y_pred_test); plt.plot(recall, precision)`

18. Compute confusion matrix.
    → `from sklearn.metrics import confusion_matrix; confusion_matrix(y_test, y_pred_test)`

19. Print model parameters.
    → `model.get_params()`

20. Extract feature importance values using `.get_feature_importance()`.
    → `model.get_feature_importance()`

21. Plot feature importance using CatBoost built-in plotting.
    → `from catboost import Pool; model.plot_feature_importance(Pool(X_train, y_train))`

22. Save trained model using `.save_model()`.
    → `model.save_model('catboost_model.cbm')`

23. Load model using `.load_model()`.
    → `model.load_model('catboost_model.cbm')`

24. Use `eval_set` for validation during training.
    → `model.fit(X_train, y_train, eval_set=(X_test, y_test))`

25. Use `early_stopping_rounds` to avoid overfitting.
    → `model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10)`

26. Set `iterations` parameter for number of boosting rounds.
    → `CatBoostRegressor(iterations=500)`

27. Set `learning_rate` parameter.
    → `CatBoostRegressor(learning_rate=0.1)`

28. Set `depth` parameter.
    → `CatBoostRegressor(depth=6)`

29. Set `l2_leaf_reg` parameter.
    → `CatBoostRegressor(l2_leaf_reg=3)`

30. Set `border_count` parameter for numerical features.
    → `CatBoostRegressor(border_count=32)`

31. Set `bagging_temperature` for sampling.
    → `CatBoostRegressor(bagging_temperature=1)`

32. Handle categorical features automatically.
    → `CatBoostClassifier(cat_features='auto')`

33. Specify categorical features manually.
    → `model.fit(X_train, y_train, cat_features=[0, 3, 5])`

34. Use ordered boosting (`boosting_type='Ordered'`).
    → `CatBoostRegressor(boosting_type='Ordered')`

35. Use plain boosting (`boosting_type='Plain'`).
    → `CatBoostRegressor(boosting_type='Plain')`

36. Handle missing values automatically.
    → CatBoost handles `NaN` internally without preprocessing.

37. Enable verbose training for monitoring.
    → `model.fit(X_train, y_train, verbose=True)`

38. Use `random_seed` for reproducibility.
    → `CatBoostRegressor(random_seed=42)`

39. Extract best iteration using `.get_best_iteration()`.
    → `model.get_best_iteration()`

40. Plot loss curve using `.plot_loss_curve()`.
    → `model.plot_loss_curve()`

41. Limit trees visualized during plotting.
    → `model.plot_tree(tree_idx=0)`

42. Extract prediction probabilities using `.predict_proba()`.
    → `model.predict_proba(X_test)`

43. Convert DataFrame to CatBoost Pool.
    → `from catboost import Pool; train_pool = Pool(X_train, y_train, cat_features=[0,3])`

44. Train model using CatBoost Pool.
    → `model.fit(train_pool)`

45. Monitor evaluation metrics using `eval_metric`.
    → `model.fit(X_train, y_train, eval_set=(X_test, y_test), eval_metric='RMSE')`

46. Use multiple evaluation metrics simultaneously.
    → `eval_metric=['RMSE','MAE']` in `.fit()`

47. Extract evaluation results programmatically.
    → `model.get_evals_result()`

48. Train model with GPU (`task_type='GPU'`).
    → `CatBoostRegressor(task_type='GPU').fit(X_train, y_train)`

49. Train model with CPU (`task_type='CPU'`).
    → `CatBoostRegressor(task_type='CPU').fit(X_train, y_train)`

50. Apply model to new unseen data.
    → `y_pred_new = model.predict(X_new)`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced model training, cross-validation, hyperparameter tuning, feature engineering*

51. Perform k-fold cross-validation using `cv()` function.
    → `from catboost import cv; cv(params, Pool(X, y), fold_count=5)`

52. Set `fold_count=5` in CV.
    → `cv(params, Pool(X, y), fold_count=5)`

53. Use stratified folds for classification.
    → `cv(params, Pool(X, y), fold_count=5, stratified=True)`

54. Use early stopping in CV.
    → `cv(params, Pool(X, y), early_stopping_rounds=10)`

55. Perform manual grid search for hyperparameters.
    → Loop over parameter combinations, run `cv()`, track best metric.

56. Perform randomized search for hyperparameters.
    → Use `RandomizedSearchCV` with `CatBoostClassifier` or `CatBoostRegressor`.

57. Tune `learning_rate` for optimal performance.
    → Test multiple `learning_rate` values with `cv()` and select best.

58. Tune `depth` for optimal performance.
    → Test different `depth` values using CV.

59. Tune `l2_leaf_reg` for optimal performance.
    → Loop over `l2_leaf_reg` values with CV.

60. Tune `bagging_temperature` for optimal performance.
    → Test multiple values in CV to select best.

61. Tune `border_count` for optimal performance.
    → Adjust `border_count` and evaluate via CV.

62. Tune `iterations` for optimal performance.
    → Use `cv()` with different `iterations` and track metrics.

63. Tune boosting type (`Ordered` vs `Plain`).
    → Test both `boosting_type='Ordered'` and `'Plain'` in CV.

64. Use cross-validation to determine best iterations.
    → `cv_results = cv(params, Pool(X, y)); best_iter = cv_results.shape[0]`

65. Use cross-validation for early stopping.
    → `cv(params, Pool(X, y), early_stopping_rounds=10)`

66. Plot CV metrics over iterations.
    → `import matplotlib.pyplot as plt; plt.plot(cv_results['test-RMSE-mean'])`

67. Extract feature importance from CV results.
    → Train model on full dataset using best params, then `model.get_feature_importance()`.

68. Compute SHAP values for feature interpretation.
    → `import shap; explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)`

69. Plot SHAP summary plot.
    → `shap.summary_plot(shap_values, X)`

70. Plot SHAP dependence plot.
    → `shap.dependence_plot('feature_name', shap_values, X)`

71. Identify most impactful features using SHAP.
    → Sort `np.abs(shap_values).mean(0)` to rank features.

72. Handle imbalanced dataset by adjusting `class_weights`.
    → `model = CatBoostClassifier(class_weights=[1, 5])`

73. Handle imbalanced dataset using `auto_class_weights`.
    → `model = CatBoostClassifier(auto_class_weights='Balanced')`

74. Train multiclass classification model.
    → `CatBoostClassifier(loss_function='MultiClass').fit(X_train, y_train)`

75. Evaluate multiclass model using multi-class logloss.
    → `eval_metric='MultiClass'` in `.fit()` and check `get_evals_result()`.

76. Evaluate multiclass AUC.
    → `roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')`

77. Use label encoding for multiclass target.
    → `from sklearn.preprocessing import LabelEncoder; y_encoded = LabelEncoder().fit_transform(y)`

78. Use one-hot encoding for categorical features.
    → `pd.get_dummies(X)`

79. Handle missing values using imputation before training.
    → `from sklearn.impute import SimpleImputer; X_imputed = SimpleImputer().fit_transform(X)`

80. Generate polynomial features.
    → `from sklearn.preprocessing import PolynomialFeatures; X_poly = PolynomialFeatures(degree=2).fit_transform(X)`

81. Generate interaction features.
    → `PolynomialFeatures(degree=2, interaction_only=True)`

82. Remove highly correlated features before training.
    → Compute correlation matrix, drop features with correlation > 0.9.

83. Use PCA for dimensionality reduction.
    → `from sklearn.decomposition import PCA; X_pca = PCA(n_components=5).fit_transform(X)`

84. Use feature selection (`SelectKBest`) with CatBoost.
    → `from sklearn.feature_selection import SelectKBest, f_classif; X_new = SelectKBest(f_classif, k=10).fit_transform(X, y)`

85. Use custom loss function.
    → Define function `(y_true, y_pred)` and pass to `loss_function` in `.fit()`.

86. Pass custom evaluation metric.
    → Use `eval_metric=custom_metric` in `.fit()`

87. Train model on sparse matrix input.
    → `from scipy.sparse import csr_matrix; X_sparse = csr_matrix(X); model.fit(X_sparse, y)`

88. Train model on large dataset using Pool.
    → `from catboost import Pool; train_pool = Pool(X, y); model.fit(train_pool)`

89. Extract leaf indices for training data.
    → `model.calc_leaf_indexes(X_train)`

90. Use leaf indices for feature engineering.
    → Treat leaf indices as categorical features for second model.

91. Combine CatBoost with sklearn `Pipeline`.
    → `from sklearn.pipeline import Pipeline; Pipeline([('cat', CatBoostClassifier())])`

92. Combine CatBoost with other classifiers in stacking.
    → Use `StackingClassifier(estimators=[('cat', cat_model), ...])`

93. Combine CatBoost with LightGBM in ensemble.
    → Include both models in `VotingClassifier` or `StackingClassifier`.

94. Train regression model with monotone constraints.
    → `CatBoostRegressor(monotone_constraints=[1, 0, -1]).fit(X_train, y_train)`

95. Train classification model with monotone constraints.
    → `CatBoostClassifier(monotone_constraints=[1, 0, -1]).fit(X_train, y_train)`

96. Implement early stopping with custom metric.
    → `model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10, eval_metric=custom_metric)`

97. Use multiple evaluation sets simultaneously.
    → `eval_set=[(X_val1, y_val1), (X_val2, y_val2)]` in `.fit()`

98. Extract per-iteration evaluation results.
    → `model.get_evals_result()`

99. Visualize per-class feature importance.
    → `model.get_feature_importance(type='PredictionValuesChange', prettified=True)`

100. Visualize cumulative feature importance.
     → Sort features by importance and plot cumulative sum using `matplotlib`.


…*(questions 101–130 continue with medium-level: cross-validation strategies, nested CV, time-series CV, categorical encoding techniques, memory-efficient training, incremental training, interaction features, tuning regularization parameters, early stopping diagnostics, feature selection, hyperparameter optimization with Optuna, LightGBM vs CatBoost comparisons)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Ensemble methods, stacking, advanced time series, model interpretation, deployment*

131. Train CatBoost model on large dataset using Pool API.
     → `from catboost import Pool; train_pool = Pool(large_X, large_y); model.fit(train_pool)`

132. Use categorical feature handling for large datasets.
     → Specify `cat_features` in `Pool()` or use `cat_features='auto'`.

133. Use CatBoost with Dask for distributed training.
     → Use `from dask_ml.xgboost import DaskCatBoostClassifier` or `CatBoostClassifier().fit(X_dask, y_dask)`

134. Use CatBoost with GPU for large datasets.
     → `CatBoostClassifier(task_type='GPU').fit(X_train, y_train)`

135. Train model incrementally using `init_model`.
     → `CatBoostRegressor().fit(X_new, y_new, init_model=old_model)`

136. Combine CatBoost with LightGBM in stacking ensemble.
     → Use sklearn `StackingClassifier` or `StackingRegressor` with both models.

137. Combine CatBoost with XGBoost in stacking.
     → Include `CatBoostClassifier` and `XGBClassifier` in `StackingClassifier`.

138. Train model for regression, then use residuals in second CatBoost model.
     → Fit first model, compute residuals, train second model on residuals.

139. Use CatBoost with sklearn `Pipeline`.
     → `from sklearn.pipeline import Pipeline; Pipeline([('cat', CatBoostClassifier())])`

140. Use CatBoost as part of voting classifier.
     → `from sklearn.ensemble import VotingClassifier; VotingClassifier(estimators=[('cat', cat_model), ...])`

141. Use CatBoost in bagging ensemble.
     → `from sklearn.ensemble import BaggingClassifier; BaggingClassifier(base_estimator=CatBoostClassifier(), n_estimators=10)`

142. Use CatBoost with cross-validated feature selection.
     → `from sklearn.feature_selection import SelectFromModel; SelectFromModel(CatBoostClassifier()).fit(X, y)`

143. Train multi-step time series forecasting using CatBoost.
     → Create lag features for each step and train separate models or multi-output regressor.

144. Train recursive time series model using CatBoost.
     → Predict one step ahead, append prediction, repeat for next step.

145. Use lag features for time series prediction.
     → `X['lag1'] = y.shift(1)`

146. Use rolling window features.
     → `X['rolling_mean'] = y.rolling(window=3).mean()`

147. Use expanding window features.
     → `X['expanding_mean'] = y.expanding().mean()`

148. Incorporate calendar features (month, day, weekday).
     → Extract from datetime: `X['month'] = dt.month; X['weekday'] = dt.weekday`

149. Incorporate holidays into model features.
     → Create binary feature for holiday dates.

150. Incorporate trend features for time series.
     → Compute linear trend feature over time.

151. Evaluate time series forecasts using RMSE.
     → `np.sqrt(mean_squared_error(y_true, y_pred))`

152. Evaluate MAPE for time series forecasts.
     → `np.mean(np.abs((y_true - y_pred)/y_true))*100`

153. Evaluate MAE for time series forecasts.
     → `mean_absolute_error(y_true, y_pred)`

154. Plot predicted vs actual time series.
     → `plt.plot(y_true); plt.plot(y_pred)`

155. Plot residuals over time.
     → `plt.plot(y_true - y_pred)`

156. Detect anomalies in residuals.
     → Flag points where `abs(residual) > threshold`.

157. Use CatBoost for ranking tasks.
     → `CatBoostRanker()`

158. Set `objective='YetiRank'` for ranking.
     → `params = {'loss_function': 'YetiRank'}`

159. Set group parameter for ranking tasks.
     → `train_pool = Pool(X, y, group_id=group_sizes)`

160. Evaluate ranking model using NDCG.
     → Use `ndcg_score(y_true, y_pred)`

161. Evaluate ranking model using MAP.
     → Compute mean average precision over queries.

162. Visualize ranking predictions.
     → Plot predicted ranks vs true ranks per query.

163. Optimize hyperparameters for ranking.
     → Use CV or Optuna targeting NDCG or MAP.

164. Implement CatBoost with early stopping for ranking.
     → `CatBoostRanker().fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)`

165. Use monotone constraints in CatBoost.
     → `CatBoostRegressor(monotone_constraints=[1,0,-1])`

166. Apply monotone constraints for selected features.
     → Set tuple with 1 (increasing), -1 (decreasing), 0 (no constraint).

167. Apply categorical constraints.
     → Specify `cat_features` in `Pool()`.

168. Use custom evaluation function for ranking.
     → Define `custom_metric(y_true, y_pred)` returning `(name, value, is_higher_better)`

169. Train CatBoost on streaming data.
     → Use `init_model` and fit incrementally on batches.

170. Incrementally update CatBoost model with new batches.
     → `model.fit(X_new, y_new, init_model=old_model)`

171. Extract leaf indices for training data.
     → `model.calc_leaf_indexes(X_train)`

172. Use leaf indices for interaction feature engineering.
     → Treat leaf indices as categorical features for second model.

173. Extract split gain for each feature.
     → `model.get_feature_importance(type='PredictionValuesChange')`

174. Identify weak features using gain.
     → Features with low gain values contribute little to model.

175. Visualize tree structure for advanced interpretation.
     → `model.plot_tree(tree_idx=0)`

176. Visualize top k trees.
     → Loop `model.plot_tree(tree_idx=i)` for i in top k indices.

177. Visualize depth of trees.
     → Inspect tree attributes or plot with depth limit.

178. Visualize leaf value distributions.
     → Extract leaf indices and plot histogram.

179. Compute SHAP interaction values.
     → `shap.TreeExplainer(model).shap_interaction_values(X)`

180. Visualize SHAP interaction heatmap.
     → `shap.summary_plot(shap_interaction_values, X, plot_type='heatmap')`

181. Deploy CatBoost model with pickle.
     → `import pickle; pickle.dump(model, open('model.pkl', 'wb'))`

182. Deploy CatBoost model using joblib.
     → `import joblib; joblib.dump(model, 'model.joblib')`

183. Deploy CatBoost in a REST API.
     → Wrap `model.predict()` in Flask/FastAPI endpoint.

184. Deploy CatBoost using Flask.
     → `from flask import Flask, request, jsonify; @app.route('/predict')`

185. Deploy CatBoost using FastAPI.
     → `from fastapi import FastAPI; app = FastAPI()`

186. Deploy CatBoost using Streamlit.
     → `import streamlit as st; st.button('Predict')`

187. Monitor model performance over time.
     → Track metrics on new data periodically.

188. Retrain model on new data periodically.
     → Schedule retraining using `.fit()` on updated dataset.

189. Detect concept drift in streaming data.
     → Monitor feature distribution changes or prediction errors.

190. Retrain incrementally on drifted data.
     → `model.fit(X_drift, y_drift, init_model=old_model)`

191. Automate hyperparameter tuning pipeline.
     → Combine CV, Optuna/random search, logging, and model selection.

192. Combine CatBoost predictions with other models (stacking/blending).
     → Use sklearn `StackingClassifier` or weighted blending.

193. Use CatBoost in a Kaggle competition workflow.
     → Standard pipeline: preprocess → CV → feature engineering → tuning → submission.

194. Use CatBoost with cross-validation folds for robust performance.
     → `cv(params, Pool(X, y), fold_count=5, stratified=True)`

195. Combine CatBoost with feature selection for best results.
     → `SelectFromModel(CatBoostClassifier())` or recursive feature elimination.

196. Interpret SHAP values to explain predictions to stakeholders.
     → Visualize summary and dependence plots for top features.

197. Compute global feature importance using SHAP.
     → `np.abs(shap_values).mean(0)`

198. Compute local feature importance for individual predictions.
     → `shap_values[i]` for ith prediction.

199. Visualize model predictions with feature explanations.
     → `shap.force_plot(explainer.expected_value, shap_values[i], X.iloc[i])`

200. Build full end-to-end workflow: data preprocessing, CatBoost training, hyperparameter tuning, evaluation, interpretation, and deployment.
     → Combine all previous steps sequentially: preprocess → feature engineering → CV & tuning → train final model → evaluate → interpret with SHAP → deploy via pickle/Flask/FastAPI/Streamlit.


---

# **Optuna Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, simple study creation, optimization basics*

1. Install Optuna using pip and import `optuna`.
   → `pip install optuna` and `import optuna`.

2. Check Optuna version.
   → `optuna.__version__`.

3. Understand the concept of a “study” in Optuna.
   → A study manages optimization trials and keeps track of the best parameters.

4. Create a simple study using `optuna.create_study()`.
   → `study = optuna.create_study(direction='minimize')`.

5. Understand objective function structure in Optuna.
   → Objective function receives a `trial` object and returns a numerical value to minimize/maximize.

6. Write a simple objective function for `y = x^2`.
   → `def objective(trial): x = trial.suggest_float('x', -10, 10); return x**2`.

7. Optimize the objective function using `study.optimize()`.
   → `study.optimize(objective, n_trials=100)`.

8. Specify number of trials using `n_trials`.
   → `study.optimize(objective, n_trials=50)`.

9. Retrieve the best value using `study.best_value`.
   → `study.best_value`.

10. Retrieve the best parameters using `study.best_params`.
    → `study.best_params`.

11. Access all trials using `study.trials`.
    → `study.trials` returns a list of `FrozenTrial` objects.

12. Access a single trial using `study.trials[0]`.
    → `trial0 = study.trials[0]`.

13. Extract trial value using `trial.value`.
    → `trial0.value`.

14. Extract trial parameters using `trial.params`.
    → `trial0.params`.

15. Use `trial.suggest_int()` to select integer hyperparameters.
    → `trial.suggest_int('n_estimators', 10, 100)`.

16. Use `trial.suggest_float()` to select float hyperparameters.
    → `trial.suggest_float('learning_rate', 0.001, 0.1, log=True)`.

17. Use `trial.suggest_categorical()` to select categorical hyperparameters.
    → `trial.suggest_categorical('activation', ['relu', 'tanh'])`.

18. Implement a simple two-hyperparameter optimization.
    → Use `trial.suggest_float()` or `trial.suggest_int()` twice in the objective function.

19. Visualize study history using `optuna.visualization.plot_optimization_history()`.
    → `optuna.visualization.plot_optimization_history(study)`.

20. Visualize parameter importance using `optuna.visualization.plot_param_importances()`.
    → `optuna.visualization.plot_param_importances(study)`.

21. Visualize parallel coordinate plot using `plot_parallel_coordinate()`.
    → `optuna.visualization.plot_parallel_coordinate(study)`.

22. Visualize contour plot for two parameters.
    → `optuna.visualization.plot_contour(study, params=['x', 'y'])`.

23. Create a new study with `direction='maximize'`.
    → `study = optuna.create_study(direction='maximize')`.

24. Optimize multiple objectives using multi-objective study.
    → `study = optuna.create_study(directions=['minimize', 'maximize'])`.

25. Understand trial states: `TrialState.COMPLETE`.
    → Trial finished successfully with a valid objective value.

26. Understand trial states: `TrialState.PRUNED`.
    → Trial was stopped early by a pruning mechanism.

27. Understand trial states: `TrialState.RUNNING`.
    → Trial is currently being evaluated.

28. Prune a trial manually using `trial.should_prune()`.
    → Call inside objective function and raise `optuna.exceptions.TrialPruned` if True.

29. Use `trial.report()` to log intermediate values.
    → `trial.report(value, step)` reports progress for pruning visualization.

30. Save a study using `RDBStorage` with SQLite.
    → `study = optuna.create_study(storage='sqlite:///example.db')`.

31. Load a study from a database.
    → `study = optuna.load_study(study_name='my_study', storage='sqlite:///example.db')`.

32. Stop optimization early using `timeout`.
    → `study.optimize(objective, timeout=600)` stops after 10 minutes.

33. Use `catch` argument in study optimization.
    → `study.optimize(objective, n_trials=50, catch=(ValueError,))` ignores specified exceptions.

34. Use `show_progress_bar=True` for progress visualization.
    → `study.optimize(objective, n_trials=50, show_progress_bar=True)`.

35. Understand difference between `optimize` and `enqueue_trial`.
    → `optimize` runs objective automatically; `enqueue_trial` adds specific parameter sets for next evaluation.

36. Add a trial manually using `study.enqueue_trial()`.
    → `study.enqueue_trial({'x': 2.5, 'y': 7})`.

37. Handle exceptions inside the objective function.
    → Use `try/except` inside objective to prevent study interruption.

38. Set random seed for reproducibility.
    → `study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))`.

39. Optimize a noisy function and visualize convergence.
    → Use `study.optimize()` and `optuna.visualization.plot_optimization_history(study)`.

40. Extract all intermediate values from trials.
    → Access `trial.intermediate_values` from each `FrozenTrial`.

41. Compute average best value across multiple studies.
    → Collect `study.best_value` from multiple studies and compute mean.

42. Compare multiple studies using visualization.
    → `optuna.visualization.plot_contour([study1, study2], params=['x','y'])`.

43. Plot slice plot using `plot_slice()`.
    → `optuna.visualization.plot_slice(study)`.

44. Filter trials by state using `study.trials_dataframe()`.
    → `df = study.trials_dataframe(attrs=('number', 'value', 'state'))`.

45. Compute statistics of all trial parameters.
    → Use `pandas` on `study.trials_dataframe()` to analyze parameter distributions.

46. Retrieve intermediate values for pruning.
    → Access `trial.intermediate_values` and call `trial.should_prune()`.

47. Handle categorical variables for ML tuning.
    → Use `trial.suggest_categorical()` in objective function.

48. Understand how Optuna chooses next trial using TPE sampler.
    → TPE models probability distributions of good vs bad trials to propose next parameters.

49. Create a study using `TPESampler()`.
    → `study = optuna.create_study(sampler=optuna.samplers.TPESampler())`.

50. Understand alternative samplers: `RandomSampler()`.
    → `RandomSampler` selects trial parameters uniformly at random instead of modeling distribution.


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced objective design, ML model tuning, hyperparameter spaces, pruning, study analysis*

51. Optimize hyperparameters for `sklearn.GradientBoostingClassifier`.
    → Use `trial.suggest_*` for `n_estimators`, `learning_rate`, `max_depth`, etc., inside the objective function.

52. Optimize hyperparameters for `sklearn.RandomForestClassifier`.
    → Tune `n_estimators`, `max_depth`, `max_features`, `min_samples_split` using Optuna.

53. Optimize hyperparameters for `sklearn.SVC`.
    → Tune `C`, `kernel`, `gamma`, and `degree` with Optuna’s `trial.suggest_*`.

54. Optimize hyperparameters for `sklearn.LogisticRegression`.
    → Tune `C`, `penalty`, `solver` as trial parameters.

55. Optimize hyperparameters for `XGBoostClassifier`.
    → Suggest `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

56. Optimize hyperparameters for `XGBoostRegressor`.
    → Similar to classifier but for regression metrics (RMSE, R²).

57. Optimize hyperparameters for `LightGBMClassifier`.
    → Tune `num_leaves`, `learning_rate`, `max_depth`, `min_data_in_leaf`, `subsample`.

58. Optimize hyperparameters for `LightGBMRegressor`.
    → Same as classifier, optimize for regression objectives.

59. Optimize hyperparameters for `CatBoostClassifier`.
    → Tune `depth`, `learning_rate`, `iterations`, `l2_leaf_reg`.

60. Optimize hyperparameters for `CatBoostRegressor`.
    → Optimize regression loss parameters and `depth`, `learning_rate`.

61. Use `train_test_split` inside objective function for validation.
    → Split data inside objective and compute validation metric for returned value.

62. Use cross-validation inside objective function.
    → Use `cross_val_score` with suggested hyperparameters to get mean metric.

63. Return average metric from CV as objective value.
    → Return `-np.mean(scores)` for minimization or mean score for maximization.

64. Use multiple hyperparameters for tuning simultaneously.
    → Suggest several parameters in same trial function.

65. Suggest integer hyperparameters for tree depth.
    → `trial.suggest_int('max_depth', 3, 10)`.

66. Suggest float hyperparameters for learning rate.
    → `trial.suggest_float('learning_rate', 0.001, 0.3, log=True)`.

67. Suggest categorical hyperparameters for booster type.
    → `trial.suggest_categorical('booster', ['gbtree', 'dart', 'gblinear'])`.

68. Apply pruning based on intermediate metric.
    → Call `trial.report(metric, step)` and `if trial.should_prune(): raise optuna.TrialPruned()`.

69. Use `optuna.integration.XGBoostPruningCallback` for XGBoost pruning.
    → Pass callback to `xgb.train` or `XGBClassifier.fit`.

70. Use `optuna.integration.LightGBMPruningCallback` for LightGBM pruning.
    → Pass callback to `lgb.train` or `LGBMClassifier.fit`.

71. Implement early stopping for CatBoost in Optuna.
    → Set `early_stopping_rounds` in `CatBoost` with `eval_set` inside objective.

72. Track trial durations and compare performance.
    → Access `trial.duration` or log timestamps inside objective.

73. Use logging to monitor trial progress.
    → Use Python `logging` or print statements inside objective.

74. Store study results in SQLite for later analysis.
    → `study = optuna.create_study(storage='sqlite:///study.db')`.

75. Load study and continue optimization.
    → `study = optuna.load_study(study_name='my_study', storage='sqlite:///study.db')`.

76. Optimize hyperparameters with constraints.
    → Use conditional logic or `trial.suggest_*` ranges to enforce constraints.

77. Optimize hyperparameters with conditional logic.
    → Example: choose `subsample` only if `booster='gbtree'`.

78. Use multi-objective optimization: maximize accuracy, minimize time.
    → Create study with `directions=['maximize', 'minimize']`.

79. Visualize Pareto front for multi-objective optimization.
    → `optuna.visualization.plot_pareto_front(study)`.

80. Compare importance of parameters in multi-objective study.
    → Use `plot_param_importances(study)` to see influence on objectives.

81. Use pruning thresholds to stop unpromising trials early.
    → Implement `trial.should_prune()` based on intermediate metric.

82. Use custom sampler strategies for exploration.
    → Replace `TPESampler` with `RandomSampler` or `CmaEsSampler`.

83. Implement Optuna study for hyperparameter tuning on time series.
    → Use `TimeSeriesSplit` for CV inside objective function.

84. Tune lag features for time series model.
    → Suggest integer lag values as hyperparameters.

85. Tune rolling window sizes.
    → Suggest window sizes in `trial.suggest_int()` and evaluate rolling features.

86. Use nested cross-validation with Optuna.
    → Outer loop splits data, inner loop uses Optuna to tune hyperparameters.

87. Optimize sklearn pipeline parameters with Optuna.
    → Suggest pipeline step parameters inside objective function.

88. Optimize feature selection thresholds inside objective.
    → Suggest threshold for `SelectKBest` or other selector as trial parameter.

89. Optimize dimensionality reduction components (e.g., PCA n_components).
    → Suggest number of components with `trial.suggest_int('n_components', 2, X.shape[1])`.

90. Tune regularization parameters (alpha, lambda) for ML models.
    → Use `trial.suggest_float('alpha', 1e-5, 1)` for Ridge/Lasso or similar.

91. Tune dropout rates for neural networks with Optuna.
    → Suggest dropout `trial.suggest_float('dropout', 0.0, 0.5)`.

92. Tune hidden layer sizes in neural networks.
    → Suggest integer neurons per layer using `trial.suggest_int()`.

93. Tune learning rate schedules in neural networks.
    → Suggest initial learning rate and decay parameters as trial hyperparameters.

94. Optimize XGBoost `max_depth` and `min_child_weight`.
    → Use `trial.suggest_int('max_depth',3,10)` and `trial.suggest_int('min_child_weight',1,10)`.

95. Optimize LightGBM `num_leaves` and `min_data_in_leaf`.
    → Use `trial.suggest_int('num_leaves', 20, 150)` and `trial.suggest_int('min_data_in_leaf', 10, 50)`.

96. Optimize CatBoost `depth` and `l2_leaf_reg`.
    → Use `trial.suggest_int('depth', 4, 10)` and `trial.suggest_float('l2_leaf_reg', 1, 10)`.

97. Evaluate optimized model on holdout set.
    → Train best parameters on full train data, predict on test set, compute metrics.

98. Track best trial and parameters programmatically.
    → `study.best_trial` gives trial object with params and value.

99. Save best trial parameters to JSON.
    → `import json; json.dump(study.best_params, open('best_params.json','w'))`.

100. Visualize parameter importance for regression tasks.
     → `optuna.visualization.plot_param_importances(study)` applied to regression study.


…*(questions 101–130 continue with medium-level: more ML tuning scenarios, cross-validation integration, complex conditional hyperparameter spaces, pruning strategies, integration with sklearn pipelines, optuna logging, parallel optimization, study storage, time-series optimization, ensemble optimization, nested optimization strategies, validation set handling)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Multi-objective optimization, parallel/distributed optimization, advanced pruning, deployment, integration*

131. Use `MultiObjectiveStudy` for optimizing multiple metrics.
     → Create study with `directions=['maximize', 'minimize']` to handle multiple objectives.

132. Optimize accuracy and inference time simultaneously.
     → Define objective returning `[accuracy, inference_time]` for multi-objective study.

133. Visualize trade-offs in multi-objective study.
     → Use `optuna.visualization.plot_pareto_front(study)` to see metric trade-offs.

134. Use `CmaEsSampler` for advanced sampling.
     → `study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())`.

135. Use `NSGAIISampler` for multi-objective TSP optimization.
     → `study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.NSGAIISampler())`.

136. Run Optuna in parallel with multiprocessing.
     → Use `n_jobs > 1` in `study.optimize()` for concurrent trials.

137. Run Optuna with multiple machines using RDBStorage.
     → Set `storage='sqlite:///example.db'` or other RDBMS; multiple processes/machines can connect.

138. Integrate Optuna with MLflow for experiment tracking.
     → Use `optuna.integration.MLflowCallback` to log trials and metrics.

139. Integrate Optuna with Weights & Biases for logging.
     → Use `optuna.integration.WandbCallback` to track study metrics.

140. Resume interrupted optimization from database.
     → Load study from RDBStorage and call `study.optimize()` again.

141. Use pruning for extremely long-running trials.
     → Call `trial.report()` and `trial.should_prune()` inside objective function.

142. Tune hyperparameters for deep learning model in PyTorch.
     → Suggest learning rate, batch size, dropout, optimizer type in objective function.

143. Tune hyperparameters for TensorFlow/Keras model.
     → Wrap model creation in objective function; suggest layer units, activation, learning rate.

144. Use pruning callback in Keras model training.
     → Pass `optuna.integration.TFKerasPruningCallback(trial, monitor='val_loss')` to `fit()`.

145. Track GPU memory usage during tuning.
     → Use `torch.cuda.memory_allocated()` or TensorFlow memory logs inside objective.

146. Optimize neural network architectures with Optuna.
     → Suggest number of layers, units per layer, activation functions dynamically.

147. Use Optuna to find best optimizer and learning rate.
     → Suggest `optimizer` categorical and `learning_rate` float in objective.

148. Optimize batch size in neural networks.
     → Suggest batch size via `trial.suggest_int('batch_size', min, max)`.

149. Optimize number of epochs with early stopping.
     → Suggest epochs or use max epochs with pruning for early stop.

150. Apply conditional hyperparameters based on model type.
     → Example: if optimizer='Adam', suggest beta1, beta2; else skip.

151. Compare multiple study results programmatically.
     → Access `study.best_trial` and metrics; aggregate or compare in code.

152. Visualize hyperparameter evolution over trials.
     → Use `optuna.visualization.plot_optimization_history(study)`.

153. Visualize intermediate values for pruning decisions.
     → `optuna.visualization.plot_intermediate_values(study)`.

154. Use `Trial.suggest_loguniform()` for log-scale hyperparameters.
     → `trial.suggest_loguniform('lr', 1e-5, 1e-1)`.

155. Use `Trial.suggest_discrete_uniform()` for discrete values.
     → `trial.suggest_discrete_uniform('dropout', 0.0, 0.5, 0.05)`.

156. Use `Trial.suggest_int(step=…)` for stepwise search.
     → `trial.suggest_int('layers', 1, 10, step=1)`.

157. Optimize ensemble weights in stacking models.
     → Suggest weights for each base model via `trial.suggest_float()` and normalize.

158. Optimize hyperparameters for multiple datasets in one study.
     → Include dataset selection in objective; return aggregated performance.

159. Optimize multiple ML pipelines simultaneously.
     → Suggest pipeline choices and parameters as trial variables.

160. Track trial runtime distribution.
     → Record `trial.duration` for analysis and visualization.

161. Visualize early pruning effectiveness.
     → Compare pruned vs completed trials using `study.trials_dataframe()`.

162. Identify bottlenecks in objective function.
     → Time each step with `time.time()` and log durations.

163. Optimize hyperparameters for NLP models (e.g., Transformers).
     → Suggest learning rate, batch size, sequence length, number of layers in objective.

164. Optimize learning rate scheduler parameters.
     → Suggest initial LR, decay rate, step size inside objective.

165. Tune dropout and attention parameters.
     → Use `trial.suggest_float()` for dropout, `trial.suggest_int()` for attention heads.

166. Optimize sequence length for NLP model.
     → `trial.suggest_int('seq_len', min_len, max_len)` and pad/truncate input.

167. Optimize optimizer type (Adam, AdamW, SGD).
     → Use `trial.suggest_categorical('optimizer', ['Adam','AdamW','SGD'])`.

168. Optimize weight decay.
     → `trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)`.

169. Optimize number of layers and units in deep model.
     → Suggest `trial.suggest_int('n_layers', 1, 10)` and units per layer dynamically.

170. Use Optuna for AutoML tasks.
     → Define objective to select pipelines, preprocessing, models, and hyperparameters.

171. Integrate Optuna with TPOT.
     → Use Optuna to tune TPOT hyperparameters like population size, generations, mutation rate.

172. Integrate Optuna with AutoKeras.
     → Wrap AutoKeras model fit in Optuna objective and suggest parameters.

173. Use Optuna to find best augmentation parameters.
     → Suggest image flip, rotation, brightness, zoom, etc., in objective function.

174. Optimize image preprocessing parameters.
     → Tune resizing, normalization, color adjustments using trial suggestions.

175. Use Optuna for feature engineering parameters.
     → Suggest lag sizes, polynomial degrees, or feature thresholds dynamically.

176. Optimize polynomial feature degree.
     → `trial.suggest_int('degree', 1, 5)` in objective.

177. Optimize feature selection thresholds.
     → Suggest `k` in `SelectKBest` or importance thresholds in tree-based models.

178. Use Optuna to prune unpromising feature sets.
     → Report intermediate score after subset evaluation and prune if below threshold.

179. Track best feature subsets across trials.
     → Store feature set associated with best trial in study metadata.

180. Store optimized models alongside study results.
     → Save model objects to disk or database after each trial or best trial.

181. Deploy best model parameters to production.
     → Export best trial parameters and retrain model for production use.

182. Automate retraining workflow with Optuna.
     → Schedule script to retrain model periodically using latest best parameters.

183. Schedule periodic re-optimization using Optuna.
     → Use cron or Airflow to trigger periodic study optimization.

184. Combine Optuna with FastAPI for automated tuning service.
     → Expose API endpoints to trigger Optuna optimization and fetch best params.

185. Use Optuna for continuous hyperparameter tuning in production.
     → Continuously feed new data and rerun study to adapt hyperparameters.

186. Monitor production model performance and trigger re-optimization.
     → Track metrics; if performance drops, call Optuna to update parameters.

187. Integrate Optuna with ML monitoring tools (e.g., Evidently).
     → Feed study results and metrics into monitoring dashboards.

188. Optimize hyperparameters for reinforcement learning agents.
     → Suggest learning rate, gamma, epsilon, network size in RL objective function.

189. Optimize reward shaping parameters.
     → Include shaping parameters as trial suggestions to improve learning.

190. Optimize environment hyperparameters for RL.
     → Suggest environment settings like state dimensions, action scaling.

191. Track multiple objective metrics in RL.
     → Return `[reward, training_time]` or `[success_rate, steps_taken]` for multi-objective study.

192. Prune long RL episodes based on reward.
     → Call `trial.report()` per episode and prune if cumulative reward low.

193. Optimize GAN parameters with Optuna.
     → Suggest generator/discriminator learning rates, layer sizes, latent dimension.

194. Optimize generator and discriminator architecture.
     → Suggest number of layers, units, activation functions via trial.

195. Optimize learning rates for generator and discriminator.
     → Use separate `trial.suggest_float()` for each optimizer.

196. Use Optuna for multi-stage training pipelines.
     → Suggest hyperparameters for each stage and combine their scores in objective.

197. Automate experiment logging, comparison, and reporting.
     → Use callbacks, RDB storage, MLflow/W&B integration for automated reporting.

198. Create a dashboard for Optuna study visualization.
     → Use Streamlit, Dash, or W&B to visualize trials, metrics, and parameter importance.

199. Analyze study trends and hyperparameter interactions.
     → Use `plot_parallel_coordinate`, `plot_slice`, `plot_param_importances`.

200. Build full end-to-end workflow: define objective, tune ML/DL model, analyze study, interpret results, and deploy best model.
     → Define objective → run Optuna study → visualize/interpret results → save best model → deploy to production.

---

# **PyCaret Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, simple experiments, and basic model evaluation*

1. Install PyCaret using pip and import a module (e.g., `classification` or `regression`).
   → `pip install pycaret` and `from pycaret.classification import *` or `from pycaret.regression import *`.

2. Check PyCaret version.
   → `pycaret.__version__`.

3. Load a sample dataset (e.g., `pycaret.datasets.get_data('diabetes')`).
   → `from pycaret.datasets import get_data; data = get_data('diabetes')`.

4. Convert dataset to Pandas DataFrame if not already.
   → `import pandas as pd; df = pd.DataFrame(data)`.

5. Inspect dataset using `.head()` and `.info()`.
   → `df.head()` and `df.info()`.

6. Identify categorical and numerical features.
   → Use `df.dtypes` or PyCaret `setup()` auto-detection.

7. Identify the target column.
   → Decide which column represents the outcome variable (e.g., `'target'`).

8. Initialize a classification experiment using `setup()`.
   → `exp_clf = setup(data=df, target='target')`.

9. Initialize a regression experiment using `setup()`.
   → `exp_reg = setup(data=df, target='target', session_id=42)`.

10. Understand the parameters of `setup()`.
    → Includes `target`, `train_size`, `numeric_features`, `categorical_features`, `normalize`, `session_id`, etc.

11. Automatically encode categorical variables.
    → `setup()` with default `categorical_features` auto-encodes using one-hot or label encoding.

12. Automatically handle missing values.
    → `setup()` with `imputation_type='simple'` or default handles NaNs.

13. Automatically normalize or scale features.
    → Use `normalize=True` or `normalize_method='zscore'` in `setup()`.

14. Apply feature transformation (e.g., log transformation).
    → `transformation=True` in `setup()`.

15. Apply polynomial features.
    → `polynomial_features=True` and optionally `polynomial_degree=2` in `setup()`.

16. Apply train/test split automatically.
    → `setup()` uses `train_size=0.7` by default.

17. Apply custom train/test split ratio.
    → Set `train_size=0.8` or desired fraction in `setup()`.

18. Enable session reproducibility using `session_id`.
    → Pass `session_id=42` in `setup()`.

19. Understand experiment log output.
    → Logs include preprocessing steps, data types, transformations, and model pipeline info.

20. Compare multiple models using `compare_models()`.
    → `best_model = compare_models()`.

21. Sort models based on metric.
    → `compare_models(sort='Accuracy')` or any metric.

22. Display top N models from comparison.
    → `compare_models(n_select=5)` shows top 5 models.

23. Select the best model automatically.
    → `best_model = compare_models()` returns model with highest default metric.

24. Create a model using `create_model()`.
    → `model = create_model('rf')` for Random Forest, etc.

25. Print model parameters.
    → `print(model)` or `get_config('X_train')` for underlying pipeline info.

26. View model performance using `plot_model()`.
    → `plot_model(model, plot='auc')` or other plot types.

27. Plot AUC curve.
    → `plot_model(model, plot='auc')`.

28. Plot Confusion Matrix.
    → `plot_model(model, plot='confusion_matrix')`.

29. Plot Precision-Recall curve.
    → `plot_model(model, plot='pr')`.

30. Plot Feature Importance.
    → `plot_model(model, plot='feature')`.

31. Plot Residuals (regression).
    → `plot_model(model, plot='residuals')`.

32. Plot Learning Curve.
    → `plot_model(model, plot='learning')`.

33. Plot Prediction Error.
    → `plot_model(model, plot='error')`.

34. Plot Feature Interaction.
    → `plot_model(model, plot='interaction')`.

35. Plot Class Prediction Probability.
    → `plot_model(model, plot='class_report')` or probability plots.

36. Evaluate model metrics with `evaluate_model()`.
    → `evaluate_model(model)` opens interactive dashboard.

37. Make predictions on holdout set using `predict_model()`.
    → `predictions = predict_model(model, data=df_test)`.

38. Interpret predictions with SHAP using `interpret_model()`.
    → `interpret_model(model)` shows feature impact using SHAP.

39. Tune hyperparameters using `tune_model()`.
    → `tuned_model = tune_model(model)` for automatic hyperparameter optimization.

40. Save a trained model using `save_model()`.
    → `save_model(model, 'my_model')`.

41. Load a saved model using `load_model()`.
    → `loaded_model = load_model('my_model')`.

42. Finalize model for deployment using `finalize_model()`.
    → `final_model = finalize_model(model)` fits on entire dataset.

43. Understand difference between `create_model()` and `finalize_model()`.
    → `create_model()` trains on training set only; `finalize_model()` retrains on full dataset for deployment.

44. Compare models on multiple metrics.
    → `compare_models(sort='F1')` or `sort='R2'`.

45. Apply cross-validation inside PyCaret.
    → `create_model('rf', fold=5)` runs 5-fold CV.

46. Understand different cross-validation folds.
    → Use `fold=3,5,10` to control number of splits; `fold_strategy` can be stratified, time-series, etc.

47. Automatically handle imbalanced datasets using SMOTE.
    → `setup(..., fix_imbalance=True)` applies SMOTE.

48. Manually specify categorical features.
    → `setup(..., categorical_features=['col1','col2'])`.

49. Manually specify numeric features.
    → `setup(..., numeric_features=['col3','col4'])`.

50. Enable logging of all experiments.
    → `setup(..., log_experiment=True, experiment_name='my_exp')`.


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced model training, ensembling, tuning, feature engineering, and experiment optimization*

51. Tune a model’s hyperparameters with `tune_model()`.
    → `tuned_model = tune_model(model)`.

52. Specify optimization metric in `tune_model()`.
    → `tuned_model = tune_model(model, optimize='F1')`.

53. Use Bayesian optimization in tuning.
    → `tune_model(model, search_algorithm='bayesian')`.

54. Use random grid search in tuning.
    → `tune_model(model, search_algorithm='random')`.

55. Use learning rate search for boosting models.
    → `tune_model(model, choose_better=True)` or specify `learning_rate` grid.

56. Create ensemble using `blend_models()`.
    → `blended_model = blend_models([model1, model2, model3])`.

57. Create a voting ensemble using `stack_models()`.
    → `stacked_model = stack_models([model1, model2], meta_model=model_meta)`.

58. Select meta-model for stacking.
    → Pass `meta_model=create_model('lr')` or any other model.

59. Evaluate stacked model performance.
    → Use `evaluate_model(stacked_model)` or `predict_model(stacked_model)`.

60. Perform bagging ensemble with `create_model()`.
    → `bagged_model = create_model('rf', ensemble=True, method='Bagging')`.

61. Perform boosting ensemble with `create_model()`.
    → `boosted_model = create_model('gbc', ensemble=True, method='Boosting')`.

62. Combine multiple feature engineering transformations.
    → Use `setup(..., polynomial_features=True, pca=True, transformation=True)`.

63. Apply feature selection with `feature_selection=True`.
    → `setup(..., feature_selection=True)`.

64. Manually select top K features.
    → Use `top_features = feature_selection(data, target, k=10)` before setup.

65. Apply PCA using `pca=True`.
    → `setup(..., pca=True)`.

66. Specify number of PCA components.
    → `setup(..., pca=True, pca_components=5)`.

67. Apply polynomial features selectively.
    → `setup(..., polynomial_features=True, polynomial_degree=2)`.

68. Apply outlier removal during preprocessing.
    → `setup(..., remove_outliers=True, outliers_threshold=0.05)`.

69. Handle missing values with specific strategies.
    → `setup(..., imputation_type='simple', numeric_imputation='mean', categorical_imputation='mode')`.

70. Encode target variable for classification.
    → `setup(..., target='target', encode_target=True)`.

71. Automatically encode multi-class targets.
    → `setup(..., encode_target=True)` handles multi-class automatically.

72. Compare models using multiple metrics (accuracy, F1, AUC).
    → `compare_models(sort='F1')` or use `compare_models()` with different `sort` values.

73. Sort comparison results by custom metric.
    → `compare_models(sort='AUC')`.

74. Select top N models for further tuning.
    → `top_models = compare_models(n_select=3)`.

75. Automate iterative model tuning pipeline.
    → Loop through top models: `for m in top_models: tune_model(m)`.

76. Plot hyperparameter importance after tuning.
    → `plot_model(tuned_model, plot='parameter')`.

77. Visualize learning curves after tuning.
    → `plot_model(tuned_model, plot='learning')`.

78. Interpret tuned model with SHAP summary plot.
    → `interpret_model(tuned_model)`.

79. Extract SHAP values for individual predictions.
    → `shap_values = shap.TreeExplainer(tuned_model).shap_values(X_test)`.

80. Evaluate regression models using MAE, MSE, RMSE.
    → `evaluate_model(model)` or `predict_model(model)` with metrics logged.

81. Evaluate classification models using accuracy, F1, ROC-AUC.
    → Same as above; metrics available in `evaluate_model()` and `predict_model()`.

82. Track experiment results programmatically.
    → Use `pull()` to get comparison table: `results = pull()`.

83. Save comparison results to CSV.
    → `results.to_csv('model_comparison.csv', index=False)`.

84. Plot residuals for ensemble models.
    → `plot_model(blended_model, plot='residuals')`.

85. Plot prediction error for ensemble models.
    → `plot_model(blended_model, plot='error')`.

86. Compare model performances on validation set.
    → `compare_models()` uses internal CV; optionally split holdout for validation.

87. Compare model performances on holdout set.
    → `predict_model(model, data=holdout_data)`.

88. Select best model from comparison for deployment.
    → `best_model = compare_models()[0]`.

89. Automate model finalization pipeline.
    → `final_model = finalize_model(best_model)`.

90. Deploy model in production-ready format.
    → Save with `save_model(final_model, 'prod_model')`.

91. Generate predictions for new unseen data.
    → `predictions = predict_model(final_model, data=new_data)`.

92. Save pipeline with preprocessing + model together.
    → `save_model(final_model, 'full_pipeline')`.

93. Load pipeline for end-to-end predictions.
    → `loaded_model = load_model('full_pipeline')`.

94. Understand `fold_strategy` in cross-validation.
    → Determines type of CV: stratified, time-series, group, etc.

95. Change `fold_strategy` to time series split.
    → `setup(..., fold_strategy='timeseries')`.

96. Change `fold_strategy` to stratified KFold.
    → `setup(..., fold_strategy='stratifiedkfold')`.

97. Enable advanced feature interaction.
    → `setup(..., feature_interaction=True)`.

98. Automatically detect categorical interactions.
    → `setup(..., feature_interaction=True)` auto-selects categorical interactions.

99. Apply natural log transformations to skewed features.
    → `setup(..., transformation=True, transformation_method='yeo-johnson')` or `log_transform=True`.

100. Apply Yeo-Johnson transformations to numeric features.
     → `setup(..., transformation=True, transformation_method='yeo-johnson')`.


…*(questions 101–130 continue with medium-level: hyperparameter tuning for all PyCaret model types, ensembling strategies, conditional tuning, iterative experiments, feature engineering combinations, automatic logging, comparison of multiple pipelines, integration with MLflow, automated pruning, batch predictions, experiment tracking, cross-validation strategies, time-series model tuning, NLP model tuning, regression vs classification specific tuning)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Time series, NLP, deployment, automation, interpretability, production pipelines*

131. Initialize a time-series regression experiment in PyCaret.
     → `from pycaret.time_series import *; ts_exp = setup(data, target='target')`.

132. Apply lag features for time series.
     → `setup(..., create_lag_features=True, lag=3)`.

133. Apply rolling mean/median features.
     → `setup(..., create_rolling_features=True, window=3, method='mean')`.

134. Automatically split training and test set for time series.
     → `setup(..., fold_strategy='timeseries', fold=3)`.

135. Tune time-series models using `tune_model()`.
     → `tuned_model = tune_model(model)`.

136. Compare time-series models using `compare_models()`.
     → `best_model = compare_models()`.

137. Stack time-series models using `stack_models()`.
     → `stacked_model = stack_models([model1, model2], meta_model=model_meta)`.

138. Use XGBoost, LightGBM, and CatBoost for time series.
     → `create_model('xgboost'); create_model('lightgbm'); create_model('catboost')`.

139. Automate feature engineering for time-series.
     → Enable `create_lag_features`, `create_rolling_features`, `create_date_features` in `setup()`.

140. Apply seasonality features (month, day, weekday).
     → `setup(..., create_date_features=True)`.

141. Apply trend features (linear, exponential).
     → Include trend transformations in `setup()` or manually engineer trend features.

142. Evaluate predictions using MAE, RMSE, MAPE.
     → `evaluate_model(model)` displays metrics; `predict_model(model)` for holdout.

143. Plot predicted vs actual values for time series.
     → `plot_model(model, plot='forecast')`.

144. Deploy time-series model for forecasting.
     → Finalize model and use `predict_model()` on future periods.

145. Initialize NLP experiment in PyCaret.
     → `from pycaret.nlp import *; nlp_exp = setup(data, target='text_column')`.

146. Preprocess text (tokenization, lowercasing, stopword removal).
     → `setup(..., text_features='text_column', custom_pipeline=True)` or defaults handle basic preprocessing.

147. Apply TF-IDF feature engineering.
     → `setup(..., feature_engineering=True)` includes TF-IDF.

148. Apply word embeddings in PyCaret NLP.
     → Use embedding models via `nlp.create_model('lda')` or integrate custom embeddings.

149. Compare multiple NLP models using `compare_models()`.
     → `best_model = compare_models()`.

150. Tune hyperparameters for NLP models.
     → `tuned_model = tune_model(model)`.

151. Stack NLP models for improved performance.
     → `stacked_model = stack_models([model1, model2], meta_model=model_meta)`.

152. Evaluate NLP models using F1, accuracy, ROC-AUC.
     → Metrics available via `evaluate_model(model)` and `predict_model(model)`.

153. Generate predictions for new text data.
     → `predictions = predict_model(model, data=new_text_data)`.

154. Interpret NLP model predictions.
     → `interpret_model(model)` visualizes feature importance or topic contribution.

155. Deploy NLP model with preprocessing + model.
     → `final_model = finalize_model(model); save_model(final_model, 'nlp_model')`.

156. Automate model deployment using `save_model()` and `load_model()`.
     → `save_model(final_model, 'model'); loaded_model = load_model('model')`.

157. Export PyCaret pipeline for API integration.
     → Use `save_model(final_model, 'pipeline.pkl')` for API consumption.

158. Use PyCaret with FastAPI for deployment.
     → Load pipeline in FastAPI endpoint and return `predict_model()` results.

159. Use PyCaret with Streamlit for visualization.
     → Streamlit widgets take input → call `predict_model()` → display predictions.

160. Integrate PyCaret with MLflow for experiment tracking.
     → `setup(..., log_experiment=True, experiment_name='exp', experiment_custom_tags={'type':'time_series'})`.

161. Automate experiment logging for multiple datasets.
     → Loop `setup()` + `compare_models()` with `log_experiment=True`.

162. Compare multiple datasets with PyCaret experiments.
     → Run `setup()` per dataset and aggregate `pull()` results.

163. Automate iterative pipeline creation and testing.
     → Loop through models, tuning, ensembling, evaluation for each dataset.

164. Perform ensemble of regression and classification pipelines.
     → Use `blend_models()` or `stack_models()` with selected models.

165. Analyze feature importance across models.
     → `plot_model(model, plot='feature')` or extract importance programmatically.

166. Interpret ensemble predictions using SHAP.
     → `interpret_model(ensemble_model)`.

167. Generate model reports automatically.
     → Use `evaluate_model()` or export plots and tables.

168. Automate selection of best model across experiments.
     → Use `compare_models()` with consistent sorting metric.

169. Apply custom metric in `tune_model()`.
     → `tune_model(model, optimize='CustomMetric')`.

170. Implement iterative hyperparameter tuning.
     → Loop `tune_model()` with updated parameters or top N models.

171. Track experiment metrics programmatically.
     → Use `pull()` after `compare_models()` or `predict_model()`.

172. Visualize performance trends over multiple experiments.
     → Collect metrics and use matplotlib/seaborn for trend plots.

173. Deploy multiple models with ensemble predictions.
     → `blend_models([model1, model2, model3])` or `stack_models()`.

174. Automate batch predictions for large datasets.
     → Use `predict_model(final_model, data=large_dataframe)`.

175. Monitor production model performance.
     → Track predictions vs actuals and metrics over time.

176. Schedule model retraining automatically.
     → Use cron, Airflow, or task scheduler with PyCaret scripts.

177. Detect concept drift and trigger retraining.
     → Compare rolling metrics; retrain if performance drops below threshold.

178. Implement continuous learning pipeline.
     → Continuously append new data, retrain using `finalize_model()`.

179. Integrate PyCaret pipelines with SQL/NoSQL databases.
     → Load data from DB, run `predict_model()`, store results back.

180. Apply PyCaret pipelines on cloud storage datasets.
     → Read/write from S3/GCS; feed DataFrame to PyCaret pipeline.

181. Export model + pipeline to pickle.
     → `save_model(final_model, 'model.pkl')`.

182. Export model + pipeline to joblib.
     → `import joblib; joblib.dump(final_model, 'model.joblib')`.

183. Automate experiment report generation in HTML.
     → Use `save_model()` + `evaluate_model()` plots, combine into HTML report.

184. Compare experiment results across multiple sessions.
     → Aggregate `pull()` outputs from multiple runs.

185. Track experiment versioning.
     → Include `session_id` and timestamp in logs and saved models.

186. Perform cross-validation with custom metrics.
     → `create_model('rf', fold=5, custom_metric=metric_function)`.

187. Customize preprocessing steps in pipeline.
     → Pass parameters in `setup()` to enable/disable scaling, encoding, outlier removal.

188. Enable/disable specific feature engineering steps.
     → `setup(..., polynomial_features=False, feature_selection=True)`.

189. Automate feature selection across multiple experiments.
     → Set `feature_selection=True` in all `setup()` calls.

190. Apply robust scaling for outlier-heavy datasets.
     → `setup(..., normalize=True, normalize_method='robust')`.

191. Integrate PyCaret with Optuna for advanced tuning.
     → `tune_model(model, search_algorithm='bayesian')`.

192. Use PyCaret for AutoML workflow.
     → `compare_models()`, `tune_model()`, `blend_models()`, `stack_models()`.

193. Compare PyCaret AutoML with manual pipelines.
     → Evaluate metrics on same dataset for manual vs AutoML pipelines.

194. Evaluate final model on completely new dataset.
     → `predict_model(final_model, data=new_dataset)`.

195. Visualize model predictions and residuals.
     → `plot_model(final_model, plot='error')` or `plot_model(final_model, plot='residuals')`.

196. Document experiment results automatically.
     → Save `pull()` tables, plots, metrics to files.

197. Build end-to-end ML workflow with PyCaret.
     → Data prep → setup → compare_models → tune → ensemble → finalize → evaluate.

198. Deploy workflow as API with preprocessing.
     → Export final model and preprocessing; wrap in FastAPI endpoint.

199. Monitor predictions for drift or anomalies.
     → Track metrics on new data; trigger retraining if drift detected.

200. Build full end-to-end automated pipeline: data prep → model training → tuning → ensembling → evaluation → interpretation → deployment → monitoring.
     → Chain all PyCaret functionalities: `setup()` → `compare_models()` → `tune_model()` → `stack_models()` → `finalize_model()` → `predict_model()` → monitoring scripts.


---

# **TPOT Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, simple AutoML runs, and basic evaluation*

1. Install TPOT using pip and import `TPOTClassifier` or `TPOTRegressor`.
   → `pip install tpot` and `from tpot import TPOTClassifier, TPOTRegressor`.

2. Check TPOT version.
   → `import tpot; tpot.__version__`.

3. Load a sample dataset (e.g., sklearn’s `load_breast_cancer`).
   → `from sklearn.datasets import load_breast_cancer; data = load_breast_cancer()`.

4. Convert dataset to Pandas DataFrame.
   → `import pandas as pd; df = pd.DataFrame(data.data, columns=data.feature_names)`.

5. Split dataset into features (`X`) and target (`y`).
   → `X = df; y = data.target`.

6. Split dataset into train and test sets using `train_test_split`.
   → `from sklearn.model_selection import train_test_split; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`.

7. Initialize a `TPOTClassifier` with default parameters.
   → `tpot = TPOTClassifier(random_state=42)`.

8. Initialize a `TPOTRegressor` with default parameters.
   → `tpot = TPOTRegressor(random_state=42)`.

9. Fit TPOT model on training data.
   → `tpot.fit(X_train, y_train)`.

10. Make predictions on test data.
    → `y_pred = tpot.predict(X_test)`.

11. Evaluate classification accuracy.
    → `from sklearn.metrics import accuracy_score; accuracy_score(y_test, y_pred)`.

12. Evaluate regression RMSE.
    → `from sklearn.metrics import mean_squared_error; import numpy as np; np.sqrt(mean_squared_error(y_test, y_pred))`.

13. Evaluate classification F1 score.
    → `from sklearn.metrics import f1_score; f1_score(y_test, y_pred)`.

14. Evaluate classification ROC-AUC.
    → `from sklearn.metrics import roc_auc_score; roc_auc_score(y_test, tpot.predict_proba(X_test)[:,1])`.

15. Evaluate regression R² score.
    → `from sklearn.metrics import r2_score; r2_score(y_test, y_pred)`.

16. Print TPOT configuration dictionary.
    → `print(tpot.config_dict)`.

17. Understand TPOT optimization algorithm (genetic programming).
    → TPOT evolves pipelines using selection, crossover, and mutation to maximize scoring metrics.

18. Set `generations` parameter.
    → `tpot = TPOTClassifier(generations=5, random_state=42)`.

19. Set `population_size` parameter.
    → `tpot = TPOTClassifier(population_size=50, random_state=42)`.

20. Use default `scoring` metric.
    → Defaults to accuracy for classification, R² for regression.

21. Specify custom `scoring` metric.
    → `tpot = TPOTClassifier(scoring='f1', random_state=42)`.

22. Set `cv` folds.
    → `tpot = TPOTClassifier(cv=5, random_state=42)`.

23. Enable early stopping with `max_time_mins`.
    → `tpot = TPOTClassifier(max_time_mins=10, random_state=42)`.

24. Limit training time using `max_time_mins`.
    → Same as above; sets total optimization time in minutes.

25. Enable verbose output during optimization.
    → `tpot = TPOTClassifier(verbosity=3, random_state=42)`.

26. Understand the pipeline structure generated by TPOT.
    → Exported pipeline shows preprocessing steps + estimator with tuned hyperparameters.

27. Export optimized pipeline to Python script using `export()`.
    → `tpot.export('optimized_pipeline.py')`.

28. Use TPOT’s `warm_start=True` for incremental optimization.
    → Allows continuing optimization from previous run.

29. Resume optimization from previous run.
    → Reload pipeline with `warm_start=True` and call `fit()` again.

30. Use `random_state` for reproducibility.
    → Ensures same pipeline evolution across runs.

31. Evaluate feature importance after pipeline selection.
    → Use `pipeline.named_steps['model'].feature_importances_` for tree models.

32. Visualize confusion matrix using predictions.
    → `from sklearn.metrics import confusion_matrix; import seaborn as sns; sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)`.

33. Plot learning curves for the optimized pipeline.
    → Use `sklearn.model_selection.learning_curve` and plot train vs. validation scores.

34. Handle missing values in dataset.
    → Impute with `SimpleImputer` before feeding data to TPOT.

35. Handle categorical features with one-hot encoding.
    → Use `pd.get_dummies` or `OneHotEncoder` in preprocessing.

36. Scale numeric features before TPOT optimization.
    → Use `StandardScaler` or `MinMaxScaler` in preprocessing pipeline.

37. Use `TPOTClassifier` for binary classification.
    → Works for two-class problems; default scoring is accuracy.

38. Use `TPOTClassifier` for multiclass classification.
    → Supports >2 classes; scoring like `f1_macro` recommended.

39. Use `TPOTRegressor` for regression tasks.
    → Fits numeric targets and evaluates with R²/MAE/RMSE.

40. Apply simple hyperparameter constraints in configuration dictionary.
    → Limit parameter ranges in `config_dict` for controlled search.

41. Include or exclude specific estimators in configuration.
    → Modify `config_dict` to add/remove desired models.

42. Include or exclude preprocessing operators.
    → Adjust `config_dict` to include/exclude transformations like PCA or scaling.

43. Limit pipeline depth.
    → Set `max_pipeline_length` parameter in TPOT.

44. Limit number of operators in pipeline.
    → Same as above; controls complexity and speed.

45. Export pipeline to `.py` script and examine code.
    → `tpot.export('pipeline.py')`; review code for steps and hyperparameters.

46. Import exported pipeline for retraining.
    → `from pipeline import exported_pipeline; exported_pipeline.fit(X_train, y_train)`.

47. Evaluate exported pipeline on test data.
    → `exported_pipeline.score(X_test, y_test)`.

48. Enable parallelism with `n_jobs=-1`.
    → Runs multiple evaluations concurrently to speed up optimization.

49. Track generation progress in verbose mode.
    → `verbosity=3` prints score and pipeline info each generation.

50. Compare pipeline performance against baseline sklearn models.
    → Train baseline model (e.g., LogisticRegression) and compare metrics with TPOT results.


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced optimization, custom configuration, integration, and pipeline analysis*

51. Customize TPOT configuration dictionary for specific model types.
    → Use `config_dict` to specify exactly which models and operators TPOT should consider.

52. Include only tree-based models in optimization.
    → Restrict `config_dict` to models like `DecisionTree`, `RandomForest`, `ExtraTrees`, `GradientBoosting`.

53. Include only linear models in optimization.
    → Limit `config_dict` to `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`.

54. Restrict preprocessing operators (e.g., exclude PCA).
    → Modify `config_dict` to remove undesired preprocessing steps.

55. Optimize pipeline using custom scoring metrics (F1, RMSE, AUC).
    → Set `scoring='f1'` or `scoring='roc_auc'` in TPOT initialization.

56. Increase number of generations for deeper search.
    → Set `generations` parameter higher to allow more evolutionary iterations.

57. Increase population size for broader exploration.
    → Increase `population_size` to explore more candidate pipelines per generation.

58. Limit optimization runtime per generation.
    → Use `max_time_mins` or control early stopping to constrain runtime.

59. Use genetic algorithm parameters: mutation rate.
    → Set `mutation_rate` to control the likelihood of random pipeline mutations.

60. Use genetic algorithm parameters: crossover rate.
    → Set `crossover_rate` to control the likelihood of combining two pipelines.

61. Analyze evolution of pipelines over generations.
    → Track best scores per generation and operator changes.

62. Plot best score per generation.
    → Use matplotlib or seaborn to plot `score_history` vs. generation.

63. Evaluate top pipelines on holdout set.
    → Use `pipeline.score(X_holdout, y_holdout)` or custom metrics.

64. Extract final optimized model from TPOT.
    → The exported pipeline (`pipeline.export()`) contains the final fitted model.

65. Inspect hyperparameters of each operator in the pipeline.
    → Check the `get_params()` of each step in the exported pipeline.

66. Integrate TPOT pipelines with sklearn `Pipeline`.
    → Include TPOT-generated steps inside `sklearn.pipeline.Pipeline` for end-to-end workflows.

67. Use TPOT for feature selection automatically.
    → TPOT includes `SelectKBest`, `RFE`, and similar operators for automated selection.

68. Preselect features before TPOT optimization.
    → Filter features manually or via domain knowledge before feeding data into TPOT.

69. Remove highly correlated features before optimization.
    → Compute correlation matrix and drop features above a correlation threshold.

70. Apply dimensionality reduction before TPOT.
    → Use PCA, TruncatedSVD, or similar outside TPOT as preprocessing.

71. Apply PCA in TPOT pipelines.
    → Include `PCA` operator in `config_dict` for pipeline evolution.

72. Apply scaling in TPOT pipelines.
    → Include `StandardScaler` or `MinMaxScaler` in `config_dict` for numeric features.

73. Apply normalization in TPOT pipelines.
    → Use `Normalizer` as a TPOT operator to normalize feature vectors.

74. Handle imbalanced datasets with SMOTE in TPOT.
    → Wrap `SMOTE` as a transformer or use a pipeline with oversampling before model.

75. Use custom preprocessing functions in TPOT pipelines.
    → Define `TransformerMixin` class and include in TPOT `config_dict`.

76. Save and load TPOT models using `joblib`.
    → `joblib.dump(pipeline, 'model.pkl')` and `joblib.load('model.pkl')`.

77. Compare multiple TPOT runs for consistency.
    → Run TPOT multiple times with same config and compare exported pipelines and scores.

78. Apply multi-class classification with TPOT.
    → TPOTClassifier supports multi-class natively; use appropriate scoring like `f1_macro`.

79. Optimize pipeline for regression tasks.
    → Use `TPOTRegressor` and set target metric (e.g., R², RMSE).

80. Analyze feature importance in tree-based pipelines.
    → Access `feature_importances_` of tree-based models in exported pipeline.

81. Interpret linear model coefficients in pipelines.
    → Extract coefficients from linear estimators to see feature impact.

82. Evaluate cross-validation performance for all pipelines.
    → Use TPOT’s CV or manual `cross_val_score` on each pipeline.

83. Analyze variance in pipelines across CV folds.
    → Compute mean and standard deviation of scores from cross-validation.

84. Evaluate TPOT pipelines on unseen datasets.
    → Test pipeline on held-out or external validation data to check generalization.

85. Limit number of pipeline operators to improve speed.
    → Set `max_pipeline_length` in TPOT to reduce pipeline complexity.

86. Set `memory` parameter to cache intermediate transformations.
    → Pass `memory=joblib.Memory(location)` to TPOT to speed up repeated computations.

87. Use ensemble operators in TPOT pipelines.
    → Include `VotingClassifier`, `StackingClassifier`, or similar in `config_dict`.

88. Enable warm-start for iterative improvement.
    → Load previous pipeline or set `warm_start=True` to continue evolution.

89. Export multiple top pipelines for comparison.
    → Save top pipelines manually after evolution using `export()` multiple times.

90. Track optimization metrics per pipeline.
    → Log pipeline scores and parameters per generation for analysis.

91. Use TPOT with parallel processing (`n_jobs`).
    → Set `n_jobs=-1` to utilize all CPU cores during pipeline evolution.

92. Integrate TPOT pipelines into larger ML workflow.
    → Include TPOT pipeline as a step in a broader preprocessing → training → deployment workflow.

93. Optimize for multiple metrics in separate runs.
    → Run TPOT multiple times, each with a different scoring metric to compare pipelines.

94. Compare performance of tree-based vs linear pipelines.
    → Evaluate TPOT runs restricted to tree-based or linear models and compare metrics.

95. Apply feature interactions automatically.
    → Include `PolynomialFeatures` or interaction operators in TPOT config.

96. Optimize preprocessing + model jointly.
    → Let TPOT evolve pipelines that include both preprocessing and estimator steps.

97. Apply one-hot encoding selectively.
    → Include `OneHotEncoder` in config_dict and restrict to categorical columns.

98. Apply categorical feature transformations automatically.
    → Use transformers like `OrdinalEncoder` or `OneHotEncoder` in TPOT pipelines.

99. Use TPOT for automated hyperparameter tuning of ensemble models.
    → TPOT evolves ensemble hyperparameters like depth, number of estimators, and weights.

100. Evaluate improvement over manually tuned sklearn models.
     → Compare TPOT pipeline metrics against baseline manually tuned models.


…*(questions 101–130 continue with medium-level: complex pipeline design, feature engineering combinations, conditional operator inclusion, nested CV, advanced scoring metrics, parallel optimization, feature importance interpretation, iterative pipeline refinement, automated exporting and deployment, integration with other AutoML tools, comparison across multiple datasets)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Multi-dataset optimization, time series, NLP, deployment, customization, production pipelines*

131. Use TPOT for multi-dataset optimization.
     → TPOT can optimize pipelines across multiple datasets by iterating through each and selecting generalized pipelines.

132. Integrate TPOT with preprocessing pipelines for time series.
     → Use `Pipeline` with `TPOTClassifier/Regressor`, including time series-specific transformers like `TimeSeriesSplit`.

133. Create lag features for time series tasks.
     → Generate lagged versions of target/feature columns to capture temporal dependencies.

134. Apply rolling/expanding features automatically in TPOT.
     → Incorporate rolling/expanding aggregations as custom transformers in the TPOT pipeline.

135. Use TPOT for multi-step forecasting.
     → Model each step individually or use recursive strategy within TPOT pipelines.

136. Optimize pipeline for time series regression.
     → Configure TPOTRegressor with time series CV like `TimeSeriesSplit`.

137. Compare multiple time-series pipelines.
     → Evaluate each TPOT-generated pipeline on validation metrics (e.g., RMSE, MAE).

138. Integrate TPOT with NLP feature engineering (TF-IDF, embeddings).
     → Wrap TF-IDF or embedding transformations in a custom transformer and include in TPOT pipeline.

139. Optimize pipelines for text classification tasks.
     → TPOT can evolve classifiers after applying text preprocessing steps like TF-IDF vectorization.

140. Apply custom tokenization in TPOT NLP pipelines.
     → Include a custom tokenizer in `CountVectorizer` or `TfidfVectorizer` inside the TPOT pipeline.

141. Evaluate TPOT-generated NLP pipelines on validation data.
     → Use `pipeline.score(X_val, y_val)` or compute metrics like accuracy/F1.

142. Export TPOT NLP pipelines for deployment.
     → Use `pipeline.export('pipeline.py')` to save the optimized pipeline.

143. Automate pipeline retraining with new data.
     → Schedule scripts that reload TPOT-exported pipeline and call `fit()` on new data.

144. Monitor TPOT pipelines in production.
     → Track metrics and logs; use dashboards or monitoring tools like MLflow/W&B.

145. Deploy TPOT pipeline as REST API using Flask.
     → Wrap `pipeline.predict()` in a Flask endpoint to serve predictions.

146. Deploy TPOT pipeline using FastAPI.
     → Create FastAPI endpoint and return `pipeline.predict()` results in JSON.

147. Deploy TPOT pipeline with Streamlit frontend.
     → Use Streamlit widgets to input data and display `pipeline.predict()` output.

148. Apply batch predictions with TPOT pipelines.
     → Call `pipeline.predict(batch_dataframe)` for multiple samples simultaneously.

149. Track pipeline performance over time.
     → Log metrics per batch or time interval for historical comparison.

150. Detect performance drift and trigger retraining.
     → Compare current predictions vs. baseline metrics; retrain if significant drift detected.

151. Automate retraining pipelines with TPOT exports.
     → Script a scheduled retraining routine using saved `pipeline.py`.

152. Combine multiple TPOT-generated pipelines in ensemble.
     → Use `VotingClassifier` or `VotingRegressor` on several exported pipelines.

153. Stack pipelines for better performance.
     → Feed predictions of base TPOT pipelines as features to a meta-model.

154. Blend pipelines for improved metrics.
     → Weighted averaging of predictions from multiple pipelines can improve robustness.

155. Optimize ensemble weights using validation set.
     → Use grid search or optimization algorithms to set voting weights for best performance.

156. Analyze feature importance across pipelines.
     → Extract `feature_importances_` from tree-based models or permutation importance.

157. Extract and interpret SHAP values from tree-based pipelines.
     → Use SHAP library on fitted tree estimators to interpret contribution of each feature.

158. Evaluate residuals in regression pipelines.
     → Plot residuals (`y_true - y_pred`) to check for bias or heteroscedasticity.

159. Compare pipeline performance on multiple metrics.
     → Compute and log metrics like accuracy, F1, RMSE, R² depending on task.

160. Generate automated reports for TPOT experiments.
     → Create scripts to summarize pipeline performance, hyperparameters, and selected features.

161. Use TPOT with custom genetic algorithm parameters.
     → Set `population_size`, `generations`, `mutation_rate`, and `crossover_rate` in TPOTRegressor/Classifier.

162. Tune mutation and crossover rates for better optimization.
     → Adjust `mutation_rate` and `crossover_rate` in TPOT initialization to control exploration.

163. Restrict or enforce specific operators in pipeline evolution.
     → Use `config_dict` to specify allowed operators in TPOT pipelines.

164. Apply warm-start for incremental improvements.
     → Load previous TPOT run and continue evolution to improve pipelines.

165. Track optimization progress programmatically.
     → Use `verbosity` parameter and logs to monitor pipeline evolution.

166. Visualize pipeline evolution using matplotlib or seaborn.
     → Plot generations vs. best pipeline score over time.

167. Evaluate multiple exports for robustness.
     → Test several top TPOT-exported pipelines on validation/test sets.

168. Save multiple top pipelines for experimentation.
     → Manually export top-performing pipelines from TPOT using `export()` multiple times.

169. Integrate TPOT with MLflow for experiment tracking.
     → Log pipeline, metrics, and parameters to MLflow in each run.

170. Integrate TPOT with Weights & Biases.
     → Use W&B logging to track evolution, metrics, and hyperparameters.

171. Automate hyperparameter tuning with TPOT and Optuna hybrid.
     → Use Optuna to tune TPOT parameters (population size, mutation rate, operators).

172. Optimize preprocessing + model selection jointly.
     → Include preprocessing transformers in TPOT pipeline and let evolutionary algorithm select them.

173. Automate feature selection and engineering using TPOT operators.
     → TPOT’s built-in operators like `SelectKBest` or `PolynomialFeatures` handle this automatically.

174. Evaluate generalization on completely new datasets.
     → Test final pipeline on out-of-distribution datasets for robustness.

175. Handle imbalanced datasets with custom pipelines.
     → Include `SMOTE`, `RandomOverSampler`, or class weighting in pipeline.

176. Evaluate F1 score, precision, recall for classification pipelines.
     → Use `sklearn.metrics` on validation predictions.

177. Evaluate R², RMSE, MAE for regression pipelines.
     → Compute metrics using `sklearn.metrics` functions.

178. Deploy ensemble pipelines for production use.
     → Wrap ensemble predictions in API or batch pipeline for inference.

179. Automate retraining based on data drift.
     → Monitor performance, trigger retraining if metrics fall below threshold.

180. Schedule TPOT AutoML runs with cron or scheduler.
     → Use system cron jobs or Airflow to trigger TPOT scripts periodically.

181. Integrate TPOT pipelines with SQL/NoSQL databases.
     → Load input data from DB, transform, predict, and optionally store results.

182. Apply TPOT pipelines to cloud-based datasets (S3, GCS).
     → Use `boto3` or cloud SDK to read/write data, then feed into TPOT pipeline.

183. Track pipeline runtime and resource usage.
     → Log timestamps and memory/CPU usage per pipeline run.

184. Optimize pipelines under runtime constraints.
     → Set `max_time_mins` in TPOT to limit optimization duration.

185. Optimize pipelines under memory constraints.
     → Limit data size or model complexity; monitor resource usage.

186. Customize TPOT operator set for specific domain.
     → Use `config_dict` to include only relevant transformers and models.

187. Compare TPOT results with PyCaret/Optuna/LGBM/XGBoost workflows.
     → Benchmark performance on same datasets using consistent metrics.

188. Export optimized pipeline as reusable Python module.
     → Use `pipeline.export('pipeline.py')` to save as standalone Python file.

189. Automate versioning of exported pipelines.
     → Save with timestamped filenames or integrate with git for version control.

190. Create dashboard for TPOT optimization progress.
     → Use Streamlit, Dash, or W&B to visualize generations and scores.

191. Automate reporting for multiple TPOT experiments.
     → Aggregate logs and metrics from multiple runs into automated reports.

192. Combine TPOT with preprocessing + postprocessing scripts.
     → Wrap TPOT pipeline inside a full pipeline that includes custom pre/post processing.

193. Integrate TPOT in CI/CD workflow for ML.
     → Automate testing, retraining, and deployment of pipelines via CI/CD tools.

194. Evaluate effect of pipeline depth on performance.
     → Vary `max_pipeline_length` in TPOT and compare metrics.

195. Apply TPOT to multi-class classification with many classes.
     → TPOTClassifier handles multi-class natively; ensure metrics handle multiple classes.

196. Apply TPOT to multi-output regression tasks.
     → Use `MultiOutputRegressor` with TPOT inside the wrapper.

197. Optimize TPOT pipeline for sparse datasets.
     → Use sparse-compatible models and transformers; avoid dense-only operators.

198. Optimize TPOT pipeline for high-dimensional datasets.
     → Include dimensionality reduction operators like PCA or SelectKBest.

199. Track and compare multiple TPOT runs for reproducibility.
     → Save seeds, configs, and logs; compare metrics systematically.

200. Build full end-to-end AutoML workflow: dataset preprocessing → TPOT optimization → pipeline export → evaluation → deployment → monitoring.
     → Chain all steps: preprocess data → run TPOT → export pipeline → validate → deploy via API/Streamlit → monitor performance over time.


---

# **Common AutoML Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, dataset handling, basic training and evaluation*

1. Install the AutoML framework.
   → Use `pip install <framework>` (e.g., `pip install pycaret`) to install.

2. Check version of the framework.
   → Use `<framework>.__version__` to verify installed version.

3. Import main AutoML class (`PyCaret: setup/create_model`, `H2O: H2OAutoML`, `TPOT: TPOTClassifier/Regressor`, `Auto-sklearn: AutoSklearnClassifier/Regressor`, `FLAML: AutoML`).
   → Import the core AutoML class appropriate to your chosen framework.

4. Load a sample dataset.
   → Use datasets from `sklearn.datasets`, `pycaret.datasets`, or CSV files.

5. Convert dataset to Pandas DataFrame if required.
   → Use `pd.DataFrame()` to ensure compatibility with AutoML pipelines.

6. Split dataset into features (`X`) and target (`y`).
   → Separate predictors and target variable for model training.

7. Split dataset into train and test sets.
   → Use `train_test_split(X, y, test_size=0.2, random_state=42)`.

8. Handle missing values automatically.
   → Let AutoML impute missing values using mean, median, or mode.

9. Handle categorical features automatically.
   → AutoML frameworks detect and encode categorical columns.

10. Encode categorical variables.
    → Convert categories to numeric using one-hot, label, or target encoding.

11. Normalize or scale numeric features automatically.
    → Apply scaling like MinMax or StandardScaler during preprocessing.

12. Initialize AutoML model with default parameters.
    → Instantiate the AutoML class without custom hyperparameters.

13. Train AutoML model on training data.
    → Fit the model using `fit()` or framework-specific training methods.

14. Make predictions on test data.
    → Use `predict()` to obtain predictions for unseen data.

15. Evaluate regression models using RMSE.
    → Compute `sqrt(mean_squared_error(y_test, y_pred))`.

16. Evaluate regression models using MAE.
    → Compute `mean_absolute_error(y_test, y_pred)`.

17. Evaluate classification models using accuracy.
    → Compute `accuracy_score(y_test, y_pred)`.

18. Evaluate classification models using F1 score.
    → Use `f1_score(y_test, y_pred, average='macro')`.

19. Evaluate classification models using ROC-AUC.
    → Use `roc_auc_score(y_test, y_proba)` for probabilistic predictions.

20. Compare multiple models generated by AutoML.
    → Review AutoML leaderboard or sorted metrics for all candidate models.

21. Sort models by performance metric.
    → Order models by chosen metric, e.g., descending accuracy or ascending RMSE.

22. Display top N models.
    → Show first N models from leaderboard or sorted list.

23. Select best model automatically.
    → Pick the highest-performing model based on the primary metric.

24. Print model parameters.
    → Access via `.get_params()` or framework-specific methods.

25. Access training logs/output.
    → Check logs for model training details and progress.

26. Access cross-validation metrics.
    → Retrieve CV scores or use framework methods to get mean/std metrics.

27. Use default train/test split.
    → Let AutoML internally split data if no custom split is provided.

28. Use custom train/test split ratio.
    → Provide explicit `train_size` or `test_size` when splitting data.

29. Use random seed for reproducibility.
    → Set `random_state` to ensure consistent results.

30. Track AutoML training time.
    → Use timing functions or framework logs to measure duration.

31. Limit AutoML training time.
    → Set `max_time` or `time_budget` to restrict runtime.

32. Limit maximum iterations or generations.
    → Specify `max_iter` or `generations` parameter in AutoML configuration.

33. Enable verbose output during training.
    → Turn on verbose/logging to monitor progress and intermediate results.

34. Save trained model/pipeline.
    → Use `save_model()` or `joblib.dump()` to persist the model.

35. Load trained model/pipeline.
    → Use `load_model()` or `joblib.load()` to restore a saved model.

36. Finalize model/pipeline for deployment.
    → Confirm the best model for predictions and lock parameters.

37. Make predictions on new/unseen data.
    → Feed new data into `.predict()` or deployed pipeline for inference.

38. Evaluate holdout dataset performance.
    → Assess metrics on a reserved validation or test set.

39. Apply simple feature preprocessing.
    → Manual preprocessing like filling missing values, encoding, or scaling.

40. Apply automated feature preprocessing.
    → Let AutoML handle cleaning, encoding, scaling, and transformations.

41. Plot performance metrics (if supported).
    → Generate plots like accuracy over iterations or learning curves.

42. Plot confusion matrix.
    → Visualize predicted vs actual classes for classification tasks.

43. Plot ROC curve.
    → Display True Positive Rate vs False Positive Rate graphically.

44. Plot Precision-Recall curve.
    → Visualize precision against recall for probabilistic classification models.

45. Track best trial/model automatically.
    → AutoML keeps a record of the top-performing configuration.

46. Export pipeline/code for production.
    → Save the full pipeline as Python code or serialized object for deployment.

47. Handle imbalanced datasets automatically.
    → Use built-in oversampling, undersampling, or class weighting.

48. Apply cross-validation automatically.
    → Let AutoML perform k-fold or stratified CV internally.

49. Evaluate cross-validation performance.
    → Review mean and standard deviation of CV metrics for model assessment.

50. Extract feature importance values.
    → Access feature importances via `.feature_importances_` or framework-specific methods.


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Hyperparameter tuning, pipeline customization, ensembling, advanced feature handling*

51. Tune hyperparameters automatically.
    → Use AutoML’s built-in hyperparameter optimization to find the best parameter values without manual intervention.

52. Specify optimization metric.
    → Choose a target metric like accuracy, RMSE, or F1-score to guide AutoML optimization.

53. Limit search space for hyperparameters.
    → Restrict the range or set of hyperparameters to speed up search or meet constraints.

54. Apply Bayesian optimization (if supported).
    → Use probabilistic models to efficiently explore hyperparameter space.

55. Apply grid search (if supported).
    → Exhaustively evaluate all combinations of selected hyperparameters.

56. Apply random search.
    → Sample hyperparameter combinations randomly for faster exploration than grid search.

57. Use multiple evaluation metrics.
    → Track several metrics simultaneously, like accuracy and F1-score, for balanced assessment.

58. Enable early stopping for unpromising models.
    → Stop training models that show poor performance early to save time.

59. Set maximum model complexity (e.g., depth, iterations).
    → Limit tree depth, iterations, or layers to prevent overfitting or long runtimes.

60. Include/exclude specific model types.
    → Specify which algorithms should or should not be considered in AutoML.

61. Include/exclude specific preprocessing operators.
    → Control which preprocessing steps (scaling, encoding, PCA) AutoML can apply.

62. Apply feature selection automatically.
    → Let AutoML choose the most relevant features for model performance.

63. Apply feature selection manually.
    → Manually select features based on domain knowledge or correlation analysis.

64. Apply PCA/dimensionality reduction.
    → Reduce feature space using PCA or other techniques to simplify models.

65. Apply polynomial or interaction features.
    → Create new features by multiplying or combining existing features for added complexity.

66. Handle high-cardinality categorical features.
    → Use target encoding, hashing, or embeddings to manage categories with many levels.

67. Handle sparse datasets.
    → Use sparse-friendly algorithms or representations to optimize memory and speed.

68. Handle numeric features with skewed distribution.
    → Apply transformations like log or Box-Cox to normalize skewed numeric features.

69. Apply target encoding for categorical features.
    → Encode categories based on the mean of the target variable for predictive power.

70. Apply one-hot encoding for categorical features.
    → Convert categorical variables into binary vectors representing each category.

71. Compare ensemble models.
    → Evaluate different ensemble strategies like bagging, boosting, and stacking.

72. Create voting ensemble.
    → Combine predictions from multiple models by majority vote (classification) or averaging (regression).

73. Create stacking ensemble.
    → Train a meta-model to combine predictions from base models for improved accuracy.

74. Combine multiple models in pipeline.
    → Chain preprocessing and multiple models sequentially or in parallel within a single pipeline.

75. Evaluate ensemble models.
    → Assess ensemble performance using metrics like accuracy, RMSE, or F1-score.

76. Extract ensemble model parameters.
    → Retrieve hyperparameters and weights of individual models in the ensemble.

77. Track model performance per fold.
    → Record metrics for each fold in cross-validation to understand variability.

78. Use k-fold cross-validation.
    → Split data into k folds and train/test k times to assess model stability.

79. Use stratified k-fold for classification.
    → Ensure class proportions are preserved in each fold for balanced evaluation.

80. Evaluate cross-validation mean metrics.
    → Calculate the average performance metric across all folds for an overall estimate.

81. Evaluate cross-validation standard deviation.
    → Measure variability of performance across folds to gauge reliability.

82. Track best hyperparameter configuration.
    → Record which hyperparameters yielded the highest performance.

83. Visualize hyperparameter importance.
    → Plot impact of each hyperparameter on model performance for interpretability.

84. Export top N pipelines/models.
    → Save the best-performing pipelines/models for future use or deployment.

85. Import exported pipelines/models for retraining.
    → Load previously saved models to retrain on new or updated data.

86. Apply custom train/test split inside AutoML.
    → Define your own data splitting strategy rather than using default random splits.

87. Track intermediate metrics for early stopping/pruning.
    → Monitor model progress during training to halt unpromising candidates.

88. Limit training resources (CPU/GPU).
    → Set constraints on computing power to manage resource usage.

89. Limit memory usage.
    → Restrict memory consumption by batch processing or lightweight models.

90. Track number of models trained.
    → Count total candidate models evaluated during AutoML search.

91. Track time per model.
    → Record training and evaluation time for each candidate model.

92. Apply automated preprocessing + model selection pipeline.
    → Let AutoML handle both feature preparation and algorithm selection seamlessly.

93. Handle class imbalance with SMOTE/oversampling.
    → Balance dataset classes by oversampling minority class or synthetic generation.

94. Evaluate multi-class classification performance.
    → Use metrics like accuracy, macro-F1, or confusion matrices for multi-class tasks.

95. Evaluate multi-output regression.
    → Assess predictive performance for models outputting multiple continuous targets.

96. Evaluate multi-label classification.
    → Measure performance for tasks where instances can have multiple simultaneous labels.

97. Track best trial/model in multi-objective scenarios.
    → Identify top-performing models when optimizing multiple metrics simultaneously.

98. Apply automated feature interaction detection.
    → Automatically generate and test feature combinations to improve predictive power.

99. Track per-operator or per-estimator performance.
    → Monitor how individual preprocessing steps or models contribute to overall performance.

100. Export best pipeline as Python code.
     → Save the top AutoML pipeline as a reusable Python script for deployment or analysis.


…*(questions 101–130 continue with medium-level: advanced pipeline customization, iterative AutoML runs, nested CV, automated feature engineering combinations, logging and experiment tracking, integration with sklearn pipelines, conditional hyperparameter search, optimization under time/memory constraints, comparison across multiple datasets)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Multi-dataset, NLP, time series, deployment, production pipelines, automation, interpretability*

131. Apply AutoML to time series regression.
     → Use AutoML frameworks like H2O, AutoGluon, or PyCaret to automatically select models and tune hyperparameters for time series regression tasks.

132. Create lag features automatically.
     → Generate past values of the target variable as new features to capture temporal dependencies.

133. Apply rolling/expanding window features.
     → Compute rolling or cumulative statistics (mean, sum, std) over a moving window to capture trends.

134. Apply seasonality features (month, day, weekday).
     → Encode time-based components like month, day, and weekday as features to capture seasonal effects.

135. Evaluate time series forecasts using RMSE/MAE/MAPE.
     → Use metrics like Root Mean Squared Error, Mean Absolute Error, or Mean Absolute Percentage Error for forecast accuracy.

136. Apply AutoML for NLP text classification.
     → Use AutoML tools that handle text preprocessing, feature extraction, and model selection automatically.

137. Preprocess text automatically.
     → Clean, tokenize, lowercase, remove stopwords, and optionally lemmatize using AutoML pipelines.

138. Apply TF-IDF vectorization.
     → Convert text into numerical features by weighting words based on frequency and uniqueness.

139. Apply word embeddings.
     → Represent words as dense vectors capturing semantic meaning, using models like Word2Vec or GloVe.

140. Compare multiple pipelines for NLP tasks.
     → Evaluate different preprocessing + model combinations to find the most effective NLP pipeline.

141. Evaluate multi-class NLP models.
     → Use metrics like accuracy, F1-score, precision, recall, or confusion matrices for multi-class classification.

142. Generate predictions for new text data.
     → Feed unseen text into the trained NLP pipeline to obtain predicted labels or probabilities.

143. Interpret NLP predictions (feature importance if supported).
     → Use tools like SHAP or LIME to understand which words influenced predictions.

144. Export NLP pipelines for deployment.
     → Save pipelines using joblib, pickle, or framework-specific exporters for production use.

145. Automate retraining for streaming or updated datasets.
     → Schedule periodic retraining or use incremental learning on new incoming data.

146. Monitor deployed model performance over time.
     → Track metrics like accuracy, drift, or latency to ensure consistent model performance.

147. Detect performance drift and trigger retraining.
     → Compare recent predictions to expected distributions; retrain if performance drops below threshold.

148. Deploy pipeline as REST API.
     → Wrap the trained pipeline in a web server (Flask/FastAPI) to serve predictions via HTTP requests.

149. Deploy pipeline with Streamlit or dashboard.
     → Build interactive web apps to showcase model predictions and visualizations.

150. Track batch prediction performance.
     → Log predictions in bulk and evaluate against ground truth to monitor accuracy.

151. Compare multiple AutoML frameworks on same dataset.
     → Run identical tasks across frameworks like H2O, AutoGluon, and PyCaret to benchmark performance.

152. Evaluate robustness of top pipelines.
     → Test pipelines on slightly modified or noisy data to assess stability.

153. Apply multi-objective optimization (accuracy + speed).
     → Optimize models considering multiple goals, such as maximizing accuracy while minimizing inference time.

154. Track model selection process programmatically.
     → Log each candidate model, its parameters, and performance for reproducibility.

155. Visualize pipeline evolution or search history.
     → Plot performance of different pipelines over iterations to see progress of AutoML search.

156. Track per-operator contribution to final pipeline.
     → Analyze which preprocessing steps or models had the most impact on pipeline performance.

157. Automate feature selection + model selection jointly.
     → Let AutoML select both optimal features and models simultaneously for best results.

158. Apply custom metric for optimization.
     → Define your own evaluation metric to guide model selection in AutoML.

159. Evaluate effect of different preprocessing steps.
     → Compare performance with varying preprocessing like scaling, encoding, or text cleaning.

160. Apply domain-specific constraints to pipelines.
     → Impose rules, e.g., enforce certain features or models due to business requirements.

161. Optimize under runtime constraints.
     → Restrict AutoML search to models and parameters that meet execution time limits.

162. Optimize under memory constraints.
     → Limit AutoML to memory-efficient models and batch processing to avoid OOM errors.

163. Use parallel processing if supported.
     → Speed up AutoML by leveraging multiple CPU cores or GPUs concurrently.

164. Track resource usage during AutoML.
     → Monitor CPU, GPU, and memory consumption for optimization and debugging.

165. Apply automated hyperparameter pruning.
     → Stop training poorly performing hyperparameter combinations early to save time.

166. Save all candidate models for comparison.
     → Retain every model explored by AutoML to allow post-analysis and ensemble creation.

167. Track hyperparameter importance across runs.
     → Analyze which hyperparameters most influence model performance.

168. Analyze top N pipelines for interpretability.
     → Examine feature importances, coefficients, or SHAP values for the best pipelines.

169. Generate automated experiment reports.
     → Produce summaries of models, metrics, and plots automatically after each AutoML run.

170. Integrate with MLflow or experiment tracking tool.
     → Log models, parameters, metrics, and artifacts to MLflow for reproducibility.

171. Combine AutoML with custom preprocessing scripts.
     → Preprocess data manually or via scripts before feeding it into AutoML pipelines.

172. Automate retraining workflow with scheduler.
     → Use cron jobs or Airflow to periodically retrain models automatically.

173. Compare pipelines across multiple datasets.
     → Evaluate if pipelines generalize well by running them on different datasets.

174. Apply AutoML for multi-output regression tasks.
     → Predict multiple continuous targets simultaneously using AutoML.

175. Apply AutoML for multi-label classification tasks.
     → Handle tasks where each instance can have multiple class labels.

176. Evaluate top pipeline ensemble performance.
     → Combine top models and assess if ensemble improves accuracy or robustness.

177. Interpret tree-based pipeline features with SHAP.
     → Use SHAP values to explain feature contributions in tree-based models.

178. Analyze linear model coefficients in pipelines.
     → Examine weights to understand the effect of each feature on predictions.

179. Visualize residuals and prediction errors.
     → Plot residuals to detect patterns or model biases.

180. Evaluate model calibration.
     → Check if predicted probabilities match actual outcomes, e.g., via reliability plots.

181. Apply advanced feature engineering (interaction, polynomial).
     → Generate features that are products or powers of original features to capture complexity.

182. Automate pipeline export + deployment.
     → Script saving and deploying pipelines to production without manual steps.

183. Monitor deployed model predictions.
     → Continuously track predictions for anomalies, drift, or unusual distributions.

184. Track model drift over time.
     → Compare new data performance with historical benchmarks to detect degradation.

185. Retrain pipelines automatically when drift detected.
     → Trigger retraining based on defined drift thresholds.

186. Compare performance across AutoML frameworks programmatically.
     → Use scripts to run multiple frameworks on the same data and compare metrics.

187. Track effect of hyperparameter changes on pipeline.
     → Log how varying hyperparameters impacts model performance over runs.

188. Optimize pipeline for sparse datasets.
     → Choose models and preprocessing suitable for data with many zeros.

189. Optimize pipeline for high-dimensional datasets.
     → Use dimensionality reduction or regularization to handle large feature sets efficiently.

190. Export pipelines as reusable Python modules.
     → Package pipelines as Python scripts or modules for easy reuse.

191. Apply AutoML to very large datasets efficiently.
     → Use sampling, distributed training, or incremental learning for scalability.

192. Track convergence of optimization process.
     → Monitor AutoML search to ensure improvements plateau rather than continue endlessly.

193. Evaluate ensemble diversity.
     → Measure differences among models to understand ensemble strength.

194. Compare single best model vs ensemble.
     → Test if combining models outperforms the individually best model.

195. Apply AutoML to imbalanced classification tasks.
     → Use class weighting, resampling, or specialized metrics to handle imbalance.

196. Apply automated feature scaling and transformation.
     → Normalize, standardize, or apply transformations automatically during AutoML.

197. Document all experiments automatically.
     → Generate logs, reports, and metadata for every AutoML run.

198. Build reproducible end-to-end AutoML workflow.
     → Combine data preprocessing, AutoML, evaluation, and deployment into a scripted pipeline.

199. Automate prediction reporting.
     → Generate scheduled reports with model predictions and metrics automatically.

200. Build full end-to-end workflow: data prep → AutoML optimization → evaluation → interpretation → deployment → monitoring.
     → Create a complete pipeline covering data ingestion, modeling, evaluation, deployment, and ongoing monitoring for continuous ML operations.


---

# **spaCy Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, data handling, and basic NLP pipelines*

1. Install spaCy using pip.
   → `pip install spacy`

2. Check spaCy version.
   → `import spacy; spacy.__version__`

3. Download an English model (e.g., `en_core_web_sm`).
   → `python -m spacy download en_core_web_sm`

4. Load the English model using `spacy.load()`.
   → `nlp = spacy.load("en_core_web_sm")`

5. Load a larger English model (e.g., `en_core_web_md`).
   → `nlp = spacy.load("en_core_web_md")`

6. Create a blank spaCy model for English.
   → `nlp = spacy.blank("en")`

7. Process a simple sentence using `nlp()`.
   → `doc = nlp("This is a sentence.")`

8. Access tokens in a processed document.
   → `for token in doc: print(token)`

9. Access token text using `.text`.
   → `token.text`

10. Access token lemma using `.lemma_`.
    → `token.lemma_`

11. Access token part-of-speech using `.pos_`.
    → `token.pos_`

12. Access token detailed tag using `.tag_`.
    → `token.tag_`

13. Access token dependency using `.dep_`.
    → `token.dep_`

14. Access token shape using `.shape_`.
    → `token.shape_`

15. Access whether token is alpha using `.is_alpha`.
    → `token.is_alpha`

16. Access whether token is stopword using `.is_stop`.
    → `token.is_stop`

17. Iterate through tokens in a sentence.
    → `for token in doc: print(token.text)`

18. Print token text, lemma, POS, and dependency.
    → `for token in doc: print(token.text, token.lemma_, token.pos_, token.dep_)`

19. Access sentence spans using `.sents`.
    → `for sent in doc.sents: print(sent.text)`

20. Split text into sentences.
    → Ensure `nlp` has a sentencizer or parser; then use `doc.sents`.

21. Access named entities using `.ents`.
    → `doc.ents`

22. Print entity text and label.
    → `for ent in doc.ents: print(ent.text, ent.label_)`

23. Access entity start and end positions.
    → `ent.start`, `ent.end`

24. Access entity label string using `.label_`.
    → `ent.label_`

25. Visualize entities using `displacy.render()`.
    → `from spacy import displacy; displacy.render(doc, style="ent")`

26. Visualize entities in Jupyter notebook.
    → `displacy.render(doc, style="ent", jupyter=True)`

27. Visualize dependencies using `displacy.render()`.
    → `displacy.render(doc, style="dep")`

28. Change rendering style for dependencies.
    → `displacy.render(doc, style="dep", options={"compact": True, "color": "blue"})`

29. Access noun chunks using `.noun_chunks`.
    → `for chunk in doc.noun_chunks: print(chunk.text)`

30. Iterate through noun chunks and print text.
    → `for chunk in doc.noun_chunks: print(chunk.text)`

31. Use `Doc` object properties: `.text`, `.vector`, `.sentiment`.
    → `doc.text`, `doc.vector`, `doc.sentiment`

32. Access document vector for similarity tasks.
    → `doc.vector`

33. Compute similarity between two tokens.
    → `token1.similarity(token2)`

34. Compute similarity between two documents.
    → `doc1.similarity(doc2)`

35. Access sentence vectors (mean of token vectors).
    → `sent.vector`

36. Apply lowercasing to tokens.
    → `[token.text.lower() for token in doc]`

37. Apply tokenization to custom text.
    → `doc = nlp(custom_text)`

38. Access whitespace and punctuation tokens.
    → `token.is_space`, `token.is_punct`

39. Remove stopwords from a document.
    → `[token for token in doc if not token.is_stop]`

40. Count number of tokens, sentences, entities.
    → `len(doc)`, `len(list(doc.sents))`, `len(doc.ents)`

41. Filter tokens by POS.
    → `[token for token in doc if token.pos_=="NOUN"]`

42. Filter tokens by entity type.
    → `[ent for ent in doc.ents if ent.label_=="PERSON"]`

43. Add custom attributes to tokens using `Token.set_extension()`.
    → `Token.set_extension("is_custom", default=False)`

44. Add custom attributes to spans using `Span.set_extension()`.
    → `Span.set_extension("score", default=0.0)`

45. Add custom attributes to Doc using `Doc.set_extension()`.
    → `Doc.set_extension("source", default="")`

46. Process a batch of texts using `nlp.pipe()`.
    → `for doc in nlp.pipe(texts, batch_size=32): pass`

47. Disable unnecessary pipeline components for speed.
    → `with nlp.select_pipes(disable=["ner"]): doc = nlp(text)`

48. Measure processing speed with and without components.
    → Use `time` module and compare processing times

49. Save model to disk using `nlp.to_disk()`.
    → `nlp.to_disk("model_path")`

50. Load model from disk using `spacy.load()`.
    → `nlp = spacy.load("model_path")`


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Pipeline components, text preprocessing, training, and intermediate NLP tasks*

51. Access pipeline components using `nlp.pipe_names`.
    → `nlp.pipe_names`

52. Get named component using `nlp.get_pipe()`.
    → `nlp.get_pipe("component_name")`

53. Add custom pipeline component using `nlp.add_pipe()`.
    → `nlp.add_pipe(custom_component, name="my_component")`

54. Remove a pipeline component using `nlp.remove_pipe()`.
    → `nlp.remove_pipe("component_name")`

55. Move a pipeline component to a specific position.
    → `nlp.move_pipe("component_name", first=True)`

56. Access tokenizer configuration.
    → `nlp.tokenizer.rules` or `nlp.tokenizer.explain(text)`

57. Customize tokenization rules.
    → Modify `nlp.tokenizer` with custom `Tokenizer` or `infixes`, `prefixes`, `suffixes`.

58. Add special cases to tokenizer.
    → `nlp.tokenizer.add_special_case("U.S.A.", [{"ORTH": "U.S.A."}])`

59. Customize entity ruler patterns.
    → `ruler.add_patterns([{"label": "ORG", "pattern": "OpenAI"}])`

60. Add patterns to entity ruler dynamically.
    → Use `ruler.add_patterns(dynamic_pattern_list)`

61. Access existing rules in entity ruler.
    → `ruler.patterns`

62. Remove patterns from entity ruler.
    → `ruler.remove_patterns(["pattern_id_or_label"])`

63. Use Matcher to find token patterns.
    → Create `Matcher(nlp.vocab)` and add patterns, then call `matcher(doc)`

64. Create a Matcher object.
    → `matcher = Matcher(nlp.vocab)`

65. Add patterns to Matcher.
    → `matcher.add("GPE_PATTERN", [pattern1, pattern2])`

66. Apply Matcher to a Doc object.
    → `matches = matcher(doc)`

67. Extract matches from Matcher.
    → Iterate `for match_id, start, end in matches:`

68. Filter matches by label or pattern ID.
    → Compare `nlp.vocab.strings[match_id]` with desired label

69. Use PhraseMatcher for multi-word expressions.
    → `phrasematcher = PhraseMatcher(nlp.vocab); phrasematcher.add("LABEL", patterns)`

70. Load patterns from JSON for PhraseMatcher.
    → `patterns = [nlp(text) for text in json_patterns]; phrasematcher.add("LABEL", patterns)`

71. Apply PhraseMatcher to a batch of texts.
    → `for doc in nlp.pipe(texts): matches = phrasematcher(doc)`

72. Access span start and end indices from matches.
    → `for match_id, start, end in matches: doc[start:end]`

73. Apply regex patterns to token text.
    → Use Python `re` module on `[token.text for token in doc]`

74. Remove punctuation using token filtering.
    → `[token for token in doc if not token.is_punct]`

75. Remove numbers using token filtering.
    → `[token for token in doc if not token.like_num]`

76. Convert text to lowercase with spaCy.
    → `[token.text.lower() for token in doc]`

77. Lemmatize tokens programmatically.
    → `[token.lemma_ for token in doc]`

78. Use `.lemma_` to normalize text.
    → `token.lemma_` gives normalized form of each token

79. Extract noun phrases for information retrieval.
    → `[chunk.text for chunk in doc.noun_chunks]`

80. Extract verb phrases for analysis.
    → Use dependency parse: `[token.subtree for token in doc if token.pos_=="VERB"]`

81. Use dependency tree to find subjects and objects.
    → `[token for token in doc if token.dep_ in ("nsubj","dobj")]`

82. Visualize dependencies in Jupyter notebook.
    → `from spacy import displacy; displacy.render(doc, style="dep")`

83. Access token ancestors and children in dependency tree.
    → `list(token.ancestors)` and `list(token.children)`

84. Apply sentiment scoring with spaCy pipeline (if model supports).
    → Use `doc._.polarity` if model provides sentiment extension

85. Calculate token-level statistics (frequency, POS counts).
    → Use `Counter([token.pos_ for token in doc])`

86. Filter entities by label type (e.g., PERSON, ORG).
    → `[ent for ent in doc.ents if ent.label_=="PERSON"]`

87. Count entity occurrences in text.
    → `Counter([ent.text for ent in doc.ents])`

88. Apply entity linking (if model supports).
    → `doc.ents` linked to KB using `EntityLinker` component

89. Train custom named entity recognizer (NER).
    → Add `ner` to pipeline and update with labeled examples

90. Convert training data to spaCy format.
    → `TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]`

91. Split annotated data into train/test.
    → Use `train_test_split(TRAIN_DATA, test_size=0.2)`

92. Initialize blank NER model.
    → `nlp = spacy.blank("en"); ner = nlp.add_pipe("ner")`

93. Add NER to pipeline.
    → `ner = nlp.create_pipe("ner"); nlp.add_pipe(ner)`

94. Add labels to NER.
    → `ner.add_label("ORG")`

95. Update NER with training examples.
    → `nlp.update([text], [annotations])`

96. Apply minibatch training with `spacy.util.minibatch`.
    → `for batch in minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001)):`

97. Evaluate trained NER on test data.
    → Use `Scorer` or loop through test docs to compare predicted vs gold entities

98. Save custom NER model.
    → `nlp.to_disk("ner_model")`

99. Load custom NER model.
    → `nlp = spacy.load("ner_model")`

100. Apply matcher to find domain-specific patterns.
     → Use `matcher(doc)` with pre-defined domain-specific patterns


…*(questions 101–130 continue with medium-level: training text classifiers, text categorization, multi-label classification, word vectors, similarity computations, custom pipeline components, pipeline optimization, preprocessing pipelines, multi-language support, batch processing, model evaluation and visualization)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Custom pipelines, deep learning integration, production-ready models, advanced NLP tasks*

131. Add custom sentiment analysis component.
     → Create a custom pipeline component using `@Language.component` and add it via `nlp.add_pipe()`.

132. Integrate spaCy with TensorFlow/Keras for text classification.
     → Use `spacy-transformers` embeddings as input to a Keras model.

133. Integrate spaCy with PyTorch for sequence labeling.
     → Extract token features from spaCy and feed into a PyTorch `nn.Module`.

134. Train text classifier with `TextCategorizer`.
     → Add `TextCategorizer` to pipeline and use `nlp.begin_training()` on labeled data.

135. Train multi-label text classifier.
     → Set `exclusive_classes=False` in `TextCategorizer`.

136. Evaluate classifier with precision, recall, F1.
     → Use `spacy.scorer.Scorer` or sklearn metrics on predictions.

137. Apply early stopping during training.
     → Monitor validation loss and stop training if it doesn’t improve.

138. Fine-tune pre-trained transformer models in spaCy.
     → Use `spacy-transformers` pipeline and continue training with `nlp.update()`.

139. Use `spacy-transformers` pipeline for NER.
     → Load `en_core_web_trf` and train/extend with your labeled data.

140. Train custom transformer-based NER.
     → Replace default NER with `Transformer`-based component in pipeline and train.

141. Freeze transformer layers during training.
     → Set `requires_grad=False` for transformer weights in training config.

142. Apply dropout in custom components.
     → Use `nn.Dropout()` in PyTorch layers or `dropout` parameter in spaCy config.

143. Use GPU acceleration for training.
     → Call `spacy.require_gpu()` before training.

144. Measure GPU utilization during training.
     → Use `nvidia-smi` or `torch.cuda.memory_allocated()`.

145. Extract embeddings for words using `token.vector`.
     → Access via `token.vector` after processing text with `nlp(text)`.

146. Extract embeddings for sentences using `Doc.vector`.
     → Use `doc = nlp(sentence)` and `doc.vector`.

147. Apply similarity search using embeddings.
     → Compute cosine similarity between `Doc.vector` of different documents.

148. Cluster documents based on embeddings.
     → Use KMeans or HDBSCAN on `Doc.vector` representations.

149. Reduce dimensionality with PCA or UMAP.
     → Apply `PCA(n_components=2)` or `umap.UMAP(n_components=2)` to embeddings.

150. Visualize embeddings in 2D or 3D.
     → Use `matplotlib` or `plotly` after dimensionality reduction.

151. Apply topic modeling on spaCy tokens.
     → Convert tokenized docs to BoW/TF-IDF and use LDA/LSI models.

152. Use spaCy vectors in downstream ML tasks.
     → Feed `Doc.vector` as features to sklearn or PyTorch models.

153. Integrate spaCy embeddings with sklearn classifier.
     → Fit classifier on `Doc.vector` features and labels.

154. Train custom dependency parser.
     → Add `DependencyParser` to pipeline and train with `nlp.update()`.

155. Evaluate parser on standard metrics (UAS, LAS).
     → Use `spacy.scorer.Scorer().score(doc)` comparing predicted vs gold parses.

156. Apply multi-language models.
     → Load multilingual pipelines like `xx_ent_wiki_sm`.

157. Translate text and process with spaCy.
     → Translate externally (e.g., Google Translate API) and then pass to `nlp()`.

158. Apply custom stopword lists.
     → Extend `nlp.Defaults.stop_words` with domain-specific words.

159. Apply token filters for domain-specific tasks.
     → Use custom component to filter or modify tokens based on rules.

160. Optimize pipeline for processing speed.
     → Disable unused components and use `nlp.pipe()` for batch processing.

161. Parallelize text processing with `nlp.pipe()`.
     → `for doc in nlp.pipe(texts, batch_size=32, n_process=4):`

162. Apply streaming large datasets.
     → Process data in chunks via `nlp.pipe()` to avoid memory overload.

163. Handle extremely long documents efficiently.
     → Split text into segments before processing to reduce memory footprint.

164. Build named entity linking (NEL) component.
     → Add `EntityLinker` to pipeline and link entities to knowledge base.

165. Integrate spaCy with knowledge base (KB).
     → Create KB using `spacy.kb.KnowledgeBase` and connect to NEL component.

166. Evaluate NEL performance.
     → Compare predicted entity links vs gold-standard KB using precision/recall.

167. Apply rule-based and ML-based NER together.
     → Use `Matcher` or `PhraseMatcher` before or after NER in pipeline.

168. Combine matcher results with NER predictions.
     → Merge spans from matcher with `doc.ents` programmatically.

169. Save complete custom pipeline.
     → `nlp.to_disk('pipeline_path')`.

170. Load complete custom pipeline.
     → `nlp = spacy.load('pipeline_path')`.

171. Apply pipeline for batch predictions.
     → Use `nlp.pipe(texts)` for efficient processing.

172. Monitor pipeline performance over time.
     → Log metrics, timings, and errors during processing.

173. Deploy spaCy pipeline as REST API.
     → Wrap `nlp` calls in FastAPI or Flask endpoints.

174. Deploy spaCy pipeline with FastAPI or Flask.
     → Define `/predict` endpoint returning `doc.ents` or other outputs.

175. Deploy spaCy pipeline in production for streaming data.
     → Use queues or streaming APIs with batch processing via `nlp.pipe()`.

176. Automate retraining of models.
     → Schedule periodic retraining scripts with updated labeled data.

177. Track changes in language usage over time.
     → Analyze embeddings, token frequencies, or topic distributions periodically.

178. Apply domain-specific NER for medical, legal, or financial texts.
     → Train NER with annotated domain-specific corpus.

179. Evaluate domain-specific model performance.
     → Use precision, recall, F1 on held-out domain dataset.

180. Integrate spaCy with text summarization.
     → Extract sentences via `Doc.sents` and apply scoring/ranking for summaries.

181. Extract keyphrases and keywords from documents.
     → Use noun chunks, entity spans, or TF-IDF weighting.

182. Apply coreference resolution in custom pipeline.
     → Integrate libraries like `neuralcoref` or HuggingFace models.

183. Integrate with transformer-based models for QA.
     → Combine `nlp` preprocessing with transformer QA pipeline from HuggingFace.

184. Evaluate pipeline performance with standard NLP benchmarks.
     → Use datasets like CoNLL, GLUE, or custom annotated corpora.

185. Use spaCy with custom embeddings.
     → Replace `Tok2Vec` weights with pretrained embeddings in config.

186. Apply custom token embeddings for domain adaptation.
     → Fine-tune embeddings on domain-specific corpus using `Tok2Vec`.

187. Fine-tune word vectors in spaCy.
     → Train `Tok2Vec` or `Transformer` layers on new data.

188. Evaluate semantic similarity between documents.
     → Compute cosine similarity between `Doc.vector` representations.

189. Apply clustering on named entities.
     → Cluster `Span.vector` embeddings using KMeans or HDBSCAN.

190. Visualize entity relationships.
     → Create network graph with `networkx` or `pyvis`.

191. Create dependency-based search queries.
     → Query `doc` using `token.dep_` and `token.head` relationships.

192. Apply search in knowledge graphs.
     → Match entities in `doc.ents` to nodes in a graph database.

193. Integrate spaCy with FLAML or other AutoML for NLP tasks.
     → Use embeddings/features from spaCy as input to AutoML pipelines.

194. Combine rule-based and ML-based classification.
     → Merge outputs of matcher rules and classifier predictions.

195. Evaluate pipeline robustness across datasets.
     → Test on multiple corpora and compare performance metrics.

196. Track model versioning for deployed pipelines.
     → Use version tags or tools like MLflow for tracking.

197. Build end-to-end NLP workflow: preprocessing → pipeline → evaluation → deployment.
     → Chain all steps: clean text → train → evaluate → save → deploy.

198. Automate retraining based on new incoming data.
     → Detect new labeled data and trigger scheduled retraining scripts.

199. Generate production-ready documentation for pipeline.
     → Use markdown or Sphinx to document components, configs, and metrics.

200. Build full spaCy NLP system: data prep → tokenization → training → evaluation → deployment → monitoring.
     → Implement complete workflow integrating all previous steps with logging, monitoring, and retraining.


---

# **Gensim Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, corpora handling, and basic preprocessing*

1. Install Gensim using pip.
   → `pip install gensim`

2. Check Gensim version.
   → `import gensim; gensim.__version__`

3. Import core modules (`corpora`, `models`, `similarities`).
   → `from gensim import corpora, models, similarities`

4. Load a sample dataset (e.g., Reuters or any text corpus).
   → Use `nltk.corpus.reuters` or `gensim.downloader.load('text8')`.

5. Convert dataset to a list of strings.
   → `[reuters.raw(fileid) for fileid in reuters.fileids()]`

6. Tokenize text into words.
   → Use `word_tokenize(text)` from NLTK.

7. Lowercase all tokens.
   → `[token.lower() for token in tokens]`

8. Remove punctuation from tokens.
   → `[token for token in tokens if token.isalpha()]`

9. Remove stopwords using custom list.
   → `[token for token in tokens if token not in custom_stopwords]`

10. Remove stopwords using NLTK or spaCy.
    → `[token for token in tokens if token not in stopwords.words('english')]`

11. Remove numbers from tokens.
    → `[token for token in tokens if not token.isdigit()]`

12. Apply simple stemming using NLTK.
    → `PorterStemmer().stem(token)`

13. Apply lemmatization using spaCy.
    → `nlp(token).lemma_`

14. Create a dictionary from tokenized documents using `corpora.Dictionary()`.
    → `dictionary = corpora.Dictionary(tokenized_docs)`

15. Inspect dictionary token to ID mapping.
    → `dictionary.token2id`

16. Filter out extreme tokens with `filter_extremes()`.
    → `dictionary.filter_extremes(no_below=5, no_above=0.5)`

17. Convert documents into Bag-of-Words (BoW) format.
    → `[dictionary.doc2bow(doc) for doc in tokenized_docs]`

18. Inspect BoW of a sample document.
    → `bow_corpus[0]`

19. Count total unique tokens.
    → `len(dictionary)`

20. Count total number of documents.
    → `len(tokenized_docs)`

21. Convert tokenized corpus into a TF-IDF model.
    → `tfidf_model = models.TfidfModel(bow_corpus)`

22. Inspect TF-IDF weights for a sample document.
    → `tfidf_model[bow_corpus[0]]`

23. Save dictionary to disk.
    → `dictionary.save('dictionary.dict')`

24. Load dictionary from disk.
    → `dictionary = corpora.Dictionary.load('dictionary.dict')`

25. Save corpus in BoW format.
    → `corpora.MmCorpus.serialize('corpus.mm', bow_corpus)`

26. Load corpus from disk.
    → `corpus = corpora.MmCorpus('corpus.mm')`

27. Iterate through corpus to view token IDs and frequencies.
    → `for doc in corpus: print(doc)`

28. Handle empty documents in corpus.
    → Filter with `[doc for doc in corpus if len(doc) > 0]`

29. Remove rare tokens programmatically.
    → Filter tokens with `dictionary.filter_extremes(no_below=5)`

30. Remove overly common tokens programmatically.
    → Filter tokens with `dictionary.filter_extremes(no_above=0.5)`

31. Apply simple bigram detection using `Phrases`.
    → `bigram = Phrases(tokenized_docs, min_count=5, threshold=10)`

32. Apply trigram detection using `Phrases`.
    → `trigram = Phrases(bigram[tokenized_docs], threshold=10)`

33. Inspect bigrams and trigrams detected.
    → `list(bigram[tokenized_docs[0]])`

34. Add detected bigrams to tokenized corpus.
    → `bigram_mod = Phraser(bigram); bigram_mod[doc]`

35. Add detected trigrams to tokenized corpus.
    → `trigram_mod = Phraser(trigram); trigram_mod[bigram_mod[doc]]`

36. Convert corpus to bag-of-ngrams representation.
    → `[dictionary.doc2bow(doc) for doc in tokenized_docs_with_ngrams]`

37. Count number of bigrams/trigrams per document.
    → `sum(1 for token in doc if '_' in token)`

38. Create a dictionary including n-grams.
    → `corpora.Dictionary(tokenized_docs_with_ngrams)`

39. Save preprocessed corpus for future use.
    → `corpora.MmCorpus.serialize('preprocessed_corpus.mm', bow_corpus)`

40. Load preprocessed corpus for analysis.
    → `corpus = corpora.MmCorpus('preprocessed_corpus.mm')`

41. Access token IDs from dictionary.
    → `dictionary.token2id`

42. Access token counts across corpus.
    → `dictionary.cfs`

43. Remove tokens below minimum frequency threshold.
    → `dictionary.filter_extremes(no_below=5)`

44. Remove tokens above maximum frequency threshold.
    → `dictionary.filter_extremes(no_above=0.5)`

45. Apply custom token filters.
    → `[token for token in doc if custom_filter(token)]`

46. Track preprocessing steps programmatically.
    → Keep a list or log of applied transformations.

47. Split dataset into training and testing sets.
    → Use `train_test_split(documents, test_size=0.2)`

48. Prepare corpus for topic modeling.
    → Tokenize, clean, convert to BoW or TF-IDF, and create dictionary.

49. Inspect first 10 documents after preprocessing.
    → `tokenized_docs[:10]`

50. Visualize token frequency distribution.
    → Use `matplotlib` or `collections.Counter` to plot token counts.


---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Topic modeling, similarity queries, vectorization, and intermediate NLP tasks*

51. Create a TF-IDF model from corpus.
    → Use `TfidfModel(corpus)` from gensim on your BoW corpus.

52. Transform BoW corpus into TF-IDF corpus.
    → Apply `tfidf_model[bow_corpus]` to get TF-IDF weights.

53. Train a Latent Semantic Indexing (LSI) model.
    → Use `LsiModel(corpus_tfidf, num_topics=N, id2word=dictionary)`.

54. Inspect topics generated by LSI.
    → Call `lsi_model.print_topics(num_topics=N)`.

55. Evaluate similarity of documents using LSI.
    → Use `MatrixSimilarity(lsi_model[corpus_tfidf])` and query vectors.

56. Train a Latent Dirichlet Allocation (LDA) model.
    → Use `LdaModel(corpus=corpus, id2word=dictionary, num_topics=N)`.

57. Inspect top words per topic in LDA.
    → Call `lda_model.print_topics(num_words=10)`.

58. Visualize topic distributions per document.
    → Use pyLDAvis: `pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)`.

59. Assign dominant topic to each document.
    → Extract the topic with the highest probability from `lda_model[doc]`.

60. Calculate coherence score for LDA model.
    → Use `CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()`.

61. Tune number of topics for LDA.
    → Train multiple LDA models with different `num_topics` and compare coherence.

62. Tune number of passes/iterations for LDA.
    → Adjust `passes` and `iterations` in `LdaModel` and monitor convergence.

63. Apply online training for large corpora.
    → Set `update_every=1` in `LdaModel` for incremental updates.

64. Save trained LDA model.
    → Use `lda_model.save('model_path')`.

65. Load trained LDA model.
    → Use `LdaModel.load('model_path')`.

66. Apply Hierarchical Dirichlet Process (HDP) modeling.
    → Use `HdpModel(corpus=corpus, id2word=dictionary)`.

67. Compare LDA vs HDP topic quality.
    → Compare coherence scores and interpretability of topics.

68. Filter corpus for rare topics.
    → Remove documents with low topic probability or rare word frequency.

69. Apply LSI similarity index.
    → Use `similarity_index = MatrixSimilarity(lsi_model[corpus_tfidf])`.

70. Apply LDA similarity index.
    → Convert documents to LDA space and use `MatrixSimilarity(lda_corpus)`.

71. Compute similarity between documents.
    → Calculate cosine similarity between vector representations.

72. Retrieve top N most similar documents to a query.
    → Sort similarity scores from the similarity matrix and pick top N.

73. Preprocess query for similarity search.
    → Tokenize, remove stopwords, and convert to BoW or TF-IDF.

74. Convert query into BoW or TF-IDF format.
    → Use `dictionary.doc2bow(query_tokens)` and optionally `tfidf_model[query_bow]`.

75. Use similarity matrix for fast retrieval.
    → Apply `similarity_index[query_vector]` for quick scoring.

76. Build and persist similarity index.
    → Use `similarity_index.save('index_path')`.

77. Load saved similarity index.
    → Use `MatrixSimilarity.load('index_path')`.

78. Apply streaming similarity search for large corpus.
    → Process documents in batches and update similarity index incrementally.

79. Filter similarity results by threshold.
    → Select scores above a chosen similarity value.

80. Evaluate retrieval performance (precision/recall).
    → Compare retrieved documents with ground truth and calculate metrics.

81. Apply gensim’s `Matutils.cossim` for vector similarity.
    → Use `gensim.matutils.cossim(vec1, vec2)`.

82. Apply weighted TF-IDF similarity.
    → Multiply term vectors by TF-IDF weights before similarity calculation.

83. Create custom similarity functions.
    → Implement a function combining cosine, Jaccard, or other metrics.

84. Apply online TF-IDF updates for streaming documents.
    → Use `TfidfModel.update(new_corpus)` incrementally.

85. Combine bigrams/trigrams in vector space.
    → Use `Phrases` and `Phraser` from gensim to detect n-grams.

86. Visualize topic distribution using pyLDAvis.
    → `pyLDAvis.show(prepared_data)` after preparing LDA and corpus.

87. Install pyLDAvis for visualization.
    → Run `pip install pyLDAvis`.

88. Prepare LDA model and corpus for pyLDAvis.
    → `pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)`.

89. Explore inter-topic distance map.
    → Visualize topic distances in pyLDAvis bubble chart.

90. Interpret top words per topic in visualization.
    → Hover over bubbles in pyLDAvis to see top words and relevance.

91. Track word probability distribution per topic.
    → Access `lda_model.get_topics()` to see per-topic word probabilities.

92. Compare multiple LDA models programmatically.
    → Use coherence scores and perplexity to select the best model.

93. Automate hyperparameter tuning for LDA (number of topics, alpha, eta).
    → Iterate over parameters and evaluate coherence to find optimal set.

94. Evaluate coherence for multiple LDA models.
    → Use `CoherenceModel` on each model and compare results.

95. Apply LDA to unseen documents.
    → Convert to BoW and call `lda_model[doc_bow]`.

96. Transform unseen document into topic distribution.
    → Use `lda_model[doc_bow]` to get topic probabilities.

97. Retrieve dominant topic for new document.
    → Select the topic with the highest probability from the LDA output.

98. Update dictionary with new vocabulary.
    → Use `dictionary.add_documents(new_texts)`.

99. Update LDA model with new documents incrementally.
    → Use `lda_model.update(new_corpus)`.

100. Track model changes over incremental updates.
     → Compare topic distributions before and after `update()` using similarity or coherence.


…*(questions 101–130 continue with medium-level: Word2Vec embeddings, Doc2Vec models, similarity search using embeddings, vector arithmetic, training embeddings on large corpus, incremental updates, model evaluation, batch processing, exporting/loading models, applying embeddings to downstream ML tasks, multi-word expressions integration, visualization of embeddings with t-SNE/UMAP, word analogy queries, nearest neighbors search, optimizing vector dimensionality, negative sampling, hierarchical softmax)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Deep learning integration, large-scale pipelines, domain adaptation, deployment, and automation*

131. Train Word2Vec embeddings on corpus.
     → `from gensim.models import Word2Vec; model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)`

132. Train CBOW vs Skip-gram models.
     → `Word2Vec(sentences, sg=0)` for CBOW, `sg=1` for Skip-gram

133. Set vector dimensionality, window size, and min count.
     → `Word2Vec(sentences, vector_size=200, window=5, min_count=2)`

134. Apply negative sampling for training.
     → `Word2Vec(sentences, negative=5)`

135. Apply hierarchical softmax for training.
     → `Word2Vec(sentences, hs=1, negative=0)`

136. Save trained Word2Vec model.
     → `model.save("word2vec.model")`

137. Load Word2Vec model from disk.
     → `from gensim.models import Word2Vec; model = Word2Vec.load("word2vec.model")`

138. Retrieve most similar words to a query word.
     → `model.wv.most_similar("king")`

139. Compute similarity between two words.
     → `model.wv.similarity("king", "queen")`

140. Compute similarity between two documents.
     → Average word vectors per document, then `cosine_similarity(doc1_vec, doc2_vec)`

141. Train Doc2Vec model on corpus.
     → `from gensim.models import Doc2Vec; model = Doc2Vec(tagged_docs, vector_size=100, window=5, min_count=1)`

142. Tag documents for Doc2Vec training.
     → `from gensim.models.doc2vec import TaggedDocument; tagged_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(docs)]`

143. Infer vector for unseen document using Doc2Vec.
     → `model.infer_vector(new_doc_words)`

144. Use embeddings in downstream ML tasks.
     → Extract vectors: `X = [model.wv[word] for word in sentence]` or Doc2Vec vectors for classification/regression

145. Cluster documents using embeddings.
     → `from sklearn.cluster import KMeans; KMeans(n_clusters=5).fit(doc_vectors)`

146. Visualize embeddings with t-SNE.
     → `from sklearn.manifold import TSNE; tsne = TSNE(); tsne_embeddings = tsne.fit_transform(vectors)`

147. Visualize embeddings with UMAP.
     → `import umap; reducer = umap.UMAP(); umap_embeddings = reducer.fit_transform(vectors)`

148. Train FastText embeddings.
     → `from gensim.models import FastText; model = FastText(sentences, vector_size=100, window=5, min_count=1)`

149. Evaluate OOV (out-of-vocabulary) word handling in FastText.
     → `model.wv['newword']` works via subword embeddings

150. Apply word analogy queries with embeddings.
     → `model.wv.most_similar(positive=['king','woman'], negative=['man'])`

151. Compute nearest neighbors for embedding vectors.
     → `model.wv.most_similar("apple", topn=10)`

152. Train embeddings on large corpus with streaming.
     → Use `gensim.models.word2vec.LineSentence(file_path)`

153. Apply multiprocessing for faster embedding training.
     → `Word2Vec(sentences, workers=4)`

154. Save embeddings in binary and text format.
     → `model.wv.save_word2vec_format('vectors.bin', binary=True)`

155. Load embeddings from binary and text format.
     → `from gensim.models import KeyedVectors; KeyedVectors.load_word2vec_format('vectors.bin', binary=True)`

156. Integrate embeddings with sklearn classifier/regressor.
     → Use averaged word vectors or Doc2Vec vectors as features for sklearn models.

157. Fine-tune embeddings for domain-specific tasks.
     → Continue training on domain corpus: `model.train(domain_sentences)`

158. Apply embeddings to topic modeling as features.
     → Replace BoW vectors with averaged embeddings for LDA/LSI input.

159. Apply embeddings to document similarity search.
     → Compute cosine similarity between document vectors.

160. Use embeddings for semantic search.
     → Compute similarity between query vector and document vectors, rank results.

161. Build retrieval system using embeddings.
     → Index embeddings in FAISS or Annoy for fast nearest neighbor search.

162. Evaluate retrieval system performance.
     → Metrics: Precision@k, Recall@k, MRR.

163. Apply embeddings for clustering (KMeans, Agglomerative).
     → `KMeans(n_clusters=10).fit(doc_vectors)` or `AgglomerativeClustering().fit(doc_vectors)`

164. Track convergence of clustering with embeddings.
     → Monitor inertia (KMeans) or dendrogram for hierarchical clustering.

165. Optimize dimensionality for embeddings.
     → Tune `vector_size` parameter and evaluate downstream performance.

166. Train embeddings with custom tokenization.
     → Pre-tokenize text before passing to Word2Vec/FastText.

167. Remove stopwords before embedding training.
     → Preprocess: remove stopwords from corpus.

168. Apply bigram/trigram detection before embedding training.
     → `from gensim.models import Phrases; bigram = Phrases(sentences); trigram = Phrases(bigram[sentences])`

169. Use embeddings to compute sentence similarity.
     → Average word vectors per sentence, then `cosine_similarity`.

170. Use embeddings in recommendation systems.
     → Compute item/item or user/item similarity with embeddings.

171. Track model performance over incremental updates.
     → Evaluate similarity or downstream task after each training increment.

172. Automate retraining with new corpus data.
     → Append new sentences and continue training with `model.train()`.

173. Deploy embeddings for real-time search.
     → Wrap vector lookup and similarity computation in API.

174. Deploy Doc2Vec/Word2Vec as REST API.
     → Use Flask/FastAPI endpoint to return vector or similarity scores.

175. Apply embeddings in chatbots/NLP pipelines.
     → Use vectors as input to intent detection or response ranking.

176. Apply embeddings in semantic clustering.
     → Cluster document vectors for topic grouping.

177. Use embeddings for document summarization tasks.
     → Compute vector representation of sentences, rank by similarity to document vector.

178. Integrate embeddings with deep learning models.
     → Feed pre-trained vectors to neural networks as input embeddings.

179. Combine embeddings with transformer models.
     → Concatenate Word2Vec/FastText vectors with BERT/Transformer embeddings.

180. Apply embeddings for entity linking.
     → Compute similarity between entity mention vectors and candidate entity vectors.

181. Fine-tune embeddings for multi-language corpora.
     → Train on multilingual corpus or use aligned embeddings.

182. Handle extremely large vocabulary efficiently.
     → Use `min_count` to filter rare words; subword embeddings with FastText.

183. Optimize training for GPU acceleration.
     → Use `gensim.models.word2vec.Word2Vec(sentences, vector_size=100, workers=4, compute_loss=True)` with GPU-compatible libraries (e.g., GPU-accelerated libraries or PyTorch).

184. Reduce memory usage for large embedding models.
     → Use `model.wv.vectors_norm` or limit `max_vocab_size`.

185. Apply incremental updates to embeddings.
     → `model.build_vocab(new_sentences, update=True); model.train(new_sentences)`

186. Track embedding drift over time.
     → Compare cosine similarity distributions of old vs new embeddings.

187. Save entire embedding pipeline for deployment.
     → Save pre-processing, tokenization, and trained embeddings together with `pickle`.

188. Load embedding pipeline in production.
     → Unpickle pipeline, use for preprocessing and vector retrieval.

189. Automate batch similarity search.
     → Use FAISS/Annoy indices to retrieve nearest neighbors in batches.

190. Apply embeddings to topic coherence scoring.
     → Compute average pairwise similarity between top words in each topic.

191. Compare embeddings across multiple corpora.
     → Train separate embeddings per corpus and compute vector similarities.

192. Evaluate embeddings for downstream NLP classification.
     → Use embeddings as features and assess model accuracy/F1.

193. Integrate embeddings with gensim LDA/LSI pipelines.
     → Replace BoW with averaged embeddings or use embeddings for topic labeling.

194. Visualize semantic clusters using embeddings.
     → Apply t-SNE/UMAP on embeddings and plot clusters.

195. Generate nearest neighbor reports.
     → Extract top-k most similar words or documents per query.

196. Track vector arithmetic operations (king - man + woman).
     → `model.wv.most_similar(positive=['king','woman'], negative=['man'])`

197. Build end-to-end embedding-based NLP workflow.
     → Preprocess → Tokenize → Train embeddings → Apply similarity, clustering, downstream tasks.

198. Automate retraining workflow for embeddings.
     → Schedule incremental updates on new corpus and re-save model.

199. Monitor deployed embedding system performance.
     → Track query response time, retrieval accuracy, and embedding drift metrics.

200. Build full end-to-end Gensim NLP system: preprocessing → dictionary/corpus → embeddings → topic modeling → similarity → deployment → monitoring.
     → Chain preprocessing → build dictionary/corpus → train Word2Vec/FastText/Doc2Vec → compute embeddings → topic modeling (LDA/LSI) → document similarity → API deployment → monitor performance.


---