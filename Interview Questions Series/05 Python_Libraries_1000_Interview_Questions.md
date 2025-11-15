To **master Matplotlib in a strictly practical way**, the key is to **start from tiny, focused exercises and gradually increase complexity**. I can structure 200 hands-on questions for you, progressing from **very basics → medium → advanced**, fully aligned with the hierarchy and plotting elements you’ve outlined. Here's how it could look:

---

## **Phase 1: Basics (Questions 1–50)**

Focus: Understanding Figure, Axes, Axis, and simple plotting functions.

**Sample questions:**

1. Create a simple line plot of `[1,2,3,4,5]` vs `[10,20,25,30,35]`.
2. Plot two lines on the same figure with different colors.
3. Change the line style to dashed and color to red.
4. Add a title “My First Plot” to your figure.
5. Label the x-axis as “Time” and y-axis as “Value”.
6. Add a legend to differentiate two lines.
7. Create a figure with **size 10x6 inches**.
8. Save a plot as `my_plot.png`.
9. Create a scatter plot for two lists `[1,2,3]` and `[4,5,6]`.
10. Plot multiple subplots (2x1) in a single figure.
11. Use `plt.subplots()` to create 2x2 Axes and plot different data in each.
12. Set the x-axis limits to 0–10 and y-axis limits to 0–100.
13. Add gridlines to your plot.
14. Annotate a point `(2,20)` with text “Important Point”.
15. Change marker style to `o` and size to `10`.
16. Use `plt.style.use('ggplot')` and observe changes.
17. Explore default colormaps with `plt.cm.viridis`.
18. Plot a horizontal line at `y=15`.
19. Plot a vertical line at `x=3`.
20. Create a bar chart with categories `['A','B','C']` and values `[10,20,30]`.

…and so on until **Question 50**, gradually including **patches** (Rectangle, Circle), basic **histograms**, and **simple customizations**.

---

## **Phase 2: Medium (Questions 51–130)**

Focus: Deepening control over the Artist hierarchy, multiple Axes, and styling.

**Sample questions:**
51. Plot a sine wave from 0 to 2π.
52. Add a cosine wave to the same plot.
53. Adjust line width and alpha (transparency).
54. Add multiple legends for different data series.
55. Create a stacked bar chart.
56. Plot a histogram with 20 bins.
57. Customize tick labels to show percentages.
58. Rotate x-axis labels by 45 degrees.
59. Place legend outside the Axes on the right.
60. Use `ax.annotate()` to add arrows pointing to a peak.
61. Combine a bar chart and line chart in one Axes.
62. Change the figure background color.
63. Change the Axes background color.
64. Add a subplot that spans multiple columns.
65. Share x-axis between multiple subplots.
66. Create a log-scale plot on the y-axis.
67. Plot error bars using `plt.errorbar()`.
68. Draw a pie chart with labels and explode effect.
69. Customize tick frequency using `MultipleLocator`.
70. Use `tight_layout()` to avoid overlapping labels.

…and continue building on **ticks, grids, color maps, advanced annotations, and composite figures**.

---

## **Phase 3: Advanced (Questions 131–200)**

Focus: Fully mastering Artist hierarchy, custom objects, interactive plots, and real-world-like data visualization.

**Sample questions:**
131. Manually create a Line2D object and add it to Axes.
132. Manually create a Text object and place it in figure coordinates.
133. Create custom patch objects (Rectangle, Circle) on a plot.
134. Overlay multiple Axes in the same Figure with different scales.
135. Use `GridSpec` to create complex subplot arrangements.
136. Create a 3D plot with `Axes3D`.
137. Plot a 3D surface using `plot_surface`.
138. Use `imshow()` to display a 2D array as an image with a colormap.
139. Add a colorbar to a heatmap.
140. Create a scatter plot with size and color mapped to data values.
141. Animate a sine wave over time using `FuncAnimation`.
142. Plot multiple time series with shared x-axis.
143. Customize major and minor ticks differently.
144. Use `transforms` to position an annotation relative to Axes, not data.
145. Combine multiple figures in one figure canvas using `Figure.add_axes()`.
146. Export a plot as PDF, SVG, and PNG.
147. Create a complex dashboard-like figure with inset plots.
148. Plot data from a CSV file using pandas and Matplotlib.
149. Create custom colormaps for heatmaps.
150. Create a twin y-axis for dual-scale plots.

---

# **Matplotlib Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Figure, Axes, Axis, basic plotting, labels, legends, styles*

1. Import `matplotlib.pyplot` as `plt` and plot `[1,2,3]` vs `[4,5,6]`.
2. Create a line plot with a green dashed line.
3. Plot two lines on the same figure with different colors.
4. Add a title “Basic Line Plot”.
5. Label the x-axis as “Time” and y-axis as “Value”.
6. Add a legend to differentiate two lines.
7. Change line width to 3.
8. Set marker style to `o` with size 8.
9. Create a figure of size 8x6 inches.
10. Save the figure as `plot1.png`.
11. Create a scatter plot of `[1,2,3]` vs `[4,5,6]`.
12. Change scatter marker color to red.
13. Change scatter marker to `^` and size to 100.
14. Plot multiple subplots (2x1) with different data.
15. Use `plt.subplots()` to create 2x2 Axes and plot in each.
16. Set x-axis limits from 0 to 10 and y-axis from 0 to 50.
17. Add gridlines to the plot.
18. Rotate x-axis labels by 45 degrees.
19. Annotate point `(2,20)` with “Important Point”.
20. Add horizontal line at y=15.
21. Add vertical line at x=3.
22. Create a bar chart with categories `['A','B','C']` and values `[10,20,30]`.
23. Change bar colors to blue.
24. Change bar width to 0.5.
25. Plot a histogram of `[1,2,2,3,3,3,4,4,5]` with 5 bins.
26. Add title and labels to histogram.
27. Use `plt.style.use('ggplot')` for a different look.
28. Explore `plt.style.available` to list all styles.
29. Create a figure with two y-axes using `twinx()`.
30. Plot sine and cosine on the same axes.
31. Change figure background color.
32. Change Axes background color.
33. Customize tick frequency on x-axis using `MultipleLocator`.
34. Add minor ticks on y-axis.
35. Add a legend outside the plot.
36. Combine a bar chart and line chart in one Axes.
37. Change font size of title and labels.
38. Change font family to `serif`.
39. Create a figure with multiple rows and columns of subplots.
40. Share x-axis between multiple subplots.
41. Share y-axis between multiple subplots.
42. Add space between subplots using `plt.tight_layout()`.
43. Create a pie chart with labels `['A','B','C']` and values `[10,20,30]`.
44. Explode the first slice of the pie chart.
45. Show percentage values on pie chart.
46. Plot a horizontal bar chart.
47. Change orientation of tick labels on horizontal bar chart.
48. Create a stacked bar chart.
49. Plot a simple area chart using `fill_between()`.
50. Use `plt.show()` to display the figure.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Deep control of Artist hierarchy, multiple Axes, styles, annotations*

51. Plot a sine wave from 0 to 2π with 100 points.
52. Add a cosine wave to the same plot.
53. Change alpha (transparency) to 0.5 for one line.
54. Add multiple legends for different data series.
55. Plot a bar chart with error bars.
56. Plot a histogram with 20 bins and normalized frequency.
57. Change tick labels to show percentages.
58. Rotate tick labels by 90 degrees.
59. Place legend outside the axes on the right.
60. Add an arrow annotation pointing to the maximum value.
61. Combine bar chart and line plot in same Axes.
62. Use `plt.subplots_adjust()` to control spacing between plots.
63. Create a log-scale plot on y-axis.
64. Create a log-scale plot on x-axis.
65. Plot a scatter plot with size mapped to data.
66. Plot a scatter plot with color mapped to data.
67. Use `plt.cm.viridis` colormap for scatter plot.
68. Plot multiple subplots with shared x-axis.
69. Plot multiple subplots with shared y-axis.
70. Use `ax.annotate()` to label a local minimum.
71. Create a stacked area chart.
72. Add minor gridlines to the plot.
73. Change linestyle to dotted for one line.
74. Plot multiple lines with different line styles.
75. Add text annotation using figure coordinates.
76. Change legend font size and frame.
77. Change tick direction to `in` or `out`.
78. Change tick length and width.
79. Create a bar chart with error bars.
80. Customize colors of a histogram.
81. Plot a cumulative histogram.
82. Create a scatter plot with categorical x-axis.
83. Plot data using pandas DataFrame.
84. Change x-axis and y-axis limits dynamically using data.
85. Use `ax.set_xticks()` to set custom ticks.
86. Use `ax.set_xticklabels()` to set custom labels.
87. Use `ax.set_yticks()` and `ax.set_yticklabels()`.
88. Create a horizontal bar chart with colors based on values.
89. Create a figure with inset Axes.
90. Plot two lines and fill the area between them.
91. Use `ax.fill_between()` with different alpha values.
92. Plot a bar chart with positive and negative values.
93. Plot multiple lines with different markers.
94. Customize marker edge color and width.
95. Create a polar plot.
96. Plot a sine wave in polar coordinates.
97. Customize polar plot gridlines and labels.
98. Plot a histogram on a logarithmic scale.
99. Plot multiple histograms on the same Axes.
100. Customize histogram bin edges.

…*(continue questions 101–130, covering error bars, color maps, twin axes, GridSpec, minor/major ticks, styles, font customization, pie charts, horizontal/stacked bar charts, legends, annotation placement, figure size adjustments, combining multiple plotting types, subplot arrangements)*.

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Fully mastering Artist hierarchy, custom objects, 3D plots, interactive plots, animations*

131. Manually create a `Line2D` object and add it to Axes.
132. Manually create a `Text` object at figure coordinates.
133. Create custom patch objects (Rectangle, Circle) on a plot.
134. Overlay multiple Axes in the same Figure with different scales.
135. Use `GridSpec` to create complex subplot arrangements.
136. Create a 3D plot using `Axes3D`.
137. Plot a 3D surface using `plot_surface`.
138. Plot a 3D wireframe.
139. Plot a 3D scatter plot.
140. Plot 3D contour plot.
141. Display a 2D array as an image using `imshow()`.
142. Add a colorbar to the image.
143. Customize colormap of `imshow()`.
144. Use logarithmic colormap normalization.
145. Plot a scatter plot with size and color mapped to two separate data columns.
146. Add a twin y-axis in a 3D plot.
147. Annotate a point in a 3D plot.
148. Use `transforms` to position annotations relative to axes.
149. Combine multiple figures in one canvas using `Figure.add_axes()`.
150. Export a figure as PDF, SVG, and PNG.
151. Animate a sine wave using `FuncAnimation`.
152. Animate multiple lines with different speeds.
153. Create a time series plot from CSV data.
154. Plot multiple time series with shared x-axis.
155. Customize major and minor ticks differently.
156. Plot multiple y-axis scales in one figure.
157. Plot multiple heatmaps side by side.
158. Use `pcolormesh()` for irregular grids.
159. Plot a hexbin chart.
160. Plot a violin plot.
161. Customize violin plot colors and widths.
162. Plot a boxplot with custom colors.
163. Overlay scatter points on boxplot.
164. Create a swarmplot using matplotlib only.
165. Plot correlation heatmap from pandas DataFrame.
166. Use masks to hide parts of heatmap.
167. Plot time series with rolling mean overlay.
168. Plot time series with shaded error region.
169. Use `step()` to create step plots.
170. Plot a dual-axis plot with different scales and line styles.
171. Create inset axes for zoomed-in plot.
172. Add custom gridlines for inset axes.
173. Add annotation with arrow pointing to inset region.
174. Combine polar plot and Cartesian plot in one figure.
175. Plot multiple pie charts in subplots.
176. Plot nested pie chart.
177. Plot a donut chart.
178. Use `broken_barh` to create timeline plots.
179. Create a Gantt chart.
180. Plot calendar heatmap.
181. Create a custom colormap from scratch.
182. Use diverging colormap for positive and negative values.
183. Plot a contour plot with labeled contours.
184. Fill contours with colors.
185. Overlay scatter on contour plot.
186. Plot quiver (vector field) plot.
187. Plot streamplot for fluid dynamics visualization.
188. Create custom legend handles.
189. Use proxy artists for complex legend entries.
190. Add annotations with formatted numbers (scientific, percentage).
191. Customize all fonts in the figure globally.
192. Plot multiple figures in a loop efficiently.
193. Plot multiple datasets with consistent styling.
194. Create reusable plotting function for custom style.
195. Use `rcParams` to change default plot settings.
196. Create figure with interactive widgets (sliders) using matplotlib.
197. Use `mpl_toolkits.axes_grid1` for colorbar alignment.
198. Plot a 3D animation.
199. Combine Matplotlib with PIL to annotate images.
200. Create a fully customized dashboard-like figure with multiple plots, annotations, legends, inset axes, and colorbars.

---

Here’s the structured draft:

---

# **NumPy Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Arrays, creation, indexing, data types, basic operations*

1. Import NumPy as `np` and print its version.
2. Create a 1D array `[1,2,3,4,5]`.
3. Create a 2D array `[[1,2,3],[4,5,6]]`.
4. Check the type and shape of an array.
5. Create an array of zeros with shape (3,4).
6. Create an array of ones with shape (2,5).
7. Create an array filled with a constant value, e.g., 7, shape (3,3).
8. Create an array using `arange(0,10,2)`.
9. Create an array using `linspace(0,1,5)`.
10. Create a 3x3 identity matrix using `eye()`.
11. Create a 2x3 random array using `np.random.rand()`.
12. Create a 2x3 normal-distributed array using `np.random.randn()`.
13. Check data type of array elements using `dtype`.
14. Change data type using `astype()`.
15. Access a specific element in 1D array.
16. Access a specific element in 2D array.
17. Slice a 1D array `[2:5]`.
18. Slice a 2D array to get submatrix.
19. Access last element using negative index.
20. Access last row in a 2D array.
21. Access last column in a 2D array.
22. Use boolean indexing to select elements >3.
23. Use boolean indexing to select even numbers.
24. Use fancy indexing to select specific rows/columns.
25. Modify a single element in a 1D array.
26. Modify an entire row in a 2D array.
27. Modify an entire column in a 2D array.
28. Add two arrays element-wise.
29. Subtract two arrays element-wise.
30. Multiply two arrays element-wise.
31. Divide two arrays element-wise.
32. Perform matrix multiplication using `@` or `dot()`.
33. Square each element of an array.
34. Take the square root of each element.
35. Compute sum of all elements.
36. Compute sum along axis 0 and axis 1.
37. Compute mean and median of an array.
38. Compute standard deviation and variance.
39. Find minimum and maximum values.
40. Find indices of minimum and maximum using `argmin` and `argmax`.
41. Use `np.unique()` to get unique elements.
42. Use `np.sort()` to sort an array.
43. Use `np.argsort()` to get sorting indices.
44. Flatten a 2D array to 1D.
45. Reshape a 1D array to 2D.
46. Transpose a 2D array.
47. Concatenate two arrays along axis 0.
48. Concatenate two arrays along axis 1.
49. Split an array into two parts.
50. Split a 2D array vertically and horizontally.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced indexing, broadcasting, linear algebra, statistics, I/O*

51. Create a 5x5 array of random integers between 0 and 10.
52. Mask elements greater than 5 using boolean indexing.
53. Replace elements divisible by 2 with -1.
54. Select every second element from a 1D array.
55. Select every second row from a 2D array.
56. Reverse a 1D array using slicing.
57. Reverse rows and columns in a 2D array.
58. Broadcast addition of a 1D array to 2D array.
59. Multiply a 2D array by a 1D array using broadcasting.
60. Add scalar value to entire array.
61. Compute element-wise exponentials.
62. Compute natural logarithm of array elements.
63. Compute sine and cosine of an array.
64. Compute dot product of two vectors.
65. Compute cross product of two vectors.
66. Compute determinant of a 2x2 matrix.
67. Compute determinant of a 3x3 matrix.
68. Compute matrix inverse.
69. Compute eigenvalues and eigenvectors.
70. Compute singular value decomposition (SVD).
71. Solve a system of linear equations using `np.linalg.solve()`.
72. Compute rank of a matrix.
73. Compute trace of a matrix.
74. Compute covariance matrix using `np.cov()`.
75. Compute correlation coefficient using `np.corrcoef()`.
76. Compute histogram of an array using `np.histogram()`.
77. Compute cumulative sum using `cumsum()`.
78. Compute cumulative product using `cumprod()`.
79. Round elements to nearest integer using `round()`.
80. Floor and ceil elements using `floor()` and `ceil()`.
81. Clip array elements between two values.
82. Find where elements satisfy a condition using `np.where()`.
83. Use `np.take()` to select elements by index.
84. Use `np.put()` to replace elements at indices.
85. Use `np.choose()` for conditional selection.
86. Generate random integers with specific seed.
87. Shuffle elements of an array randomly.
88. Repeat elements of an array.
89. Tile an array to repeat along axes.
90. Stack arrays vertically using `vstack()`.
91. Stack arrays horizontally using `hstack()`.
92. Stack arrays along a new axis using `stack()`.
93. Split arrays using `hsplit()` and `vsplit()`.
94. Load array from text file using `np.loadtxt()`.
95. Save array to text file using `np.savetxt()`.
96. Load array from binary file using `np.load()`.
97. Save array to binary file using `np.save()`.
98. Use structured arrays to store mixed data types.
99. Access fields of structured array.
100. Convert list of lists to NumPy array.

…*(questions 101–130 include advanced slicing, advanced broadcasting, linear algebra with large matrices, generating random numbers, cumulative and windowed operations, and statistics functions like percentile, quantile, masked arrays, and structured arrays)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Advanced manipulations, broadcasting tricks, memory efficiency, performance, complex operations*

131. Create a 3D array and index slices along each axis.
132. Swap axes of a 3D array.
133. Reshape 3D array to 2D.
134. Flatten 3D array to 1D.
135. Use `np.broadcast_arrays()` to align shapes.
136. Use `np.meshgrid()` to create coordinate grids.
137. Compute function values over meshgrid.
138. Use `np.vectorize()` for element-wise operations on arrays.
139. Perform element-wise comparison between arrays.
140. Use `np.all()` and `np.any()` for logical tests.
141. Use `np.isclose()` to compare float arrays.
142. Compute Manhattan distance between two vectors.
143. Compute Euclidean distance between two arrays.
144. Implement linear regression using matrix operations.
145. Compute polynomial features of a vector.
146. Perform FFT using `np.fft.fft()`.
147. Perform inverse FFT using `np.fft.ifft()`.
148. Compute correlation of two 1D arrays.
149. Compute moving average using convolution.
150. Compute weighted average using `np.average()` with weights.
151. Use `np.lib.stride_tricks` to create sliding window view.
152. Perform block-wise operations using reshaping.
153. Normalize an array to 0–1 range.
154. Standardize an array to mean=0, std=1.
155. Replace NaN values with mean of array.
156. Mask NaN values and perform computations.
157. Compute cumulative product along axis in 2D array.
158. Compute rank of elements along axis.
159. Sort array along specified axis.
160. Perform argsort along axis.
161. Compute percentile along axis.
162. Compute quantiles.
163. Apply function along axis using `np.apply_along_axis()`.
164. Compute pairwise distances between rows of a matrix.
165. Generate random samples from uniform distribution.
166. Generate random samples from normal distribution with mean/std.
167. Generate multivariate normal samples.
168. Perform linear algebra eigen decomposition for symmetric matrices.
169. Compute pseudo-inverse of a matrix.
170. Use Kronecker product.
171. Perform Hadamard (element-wise) product.
172. Reshape array without copying data using `reshape(-1)`.
173. Check memory layout of an array (`C` vs `F`).
174. Convert array to Fortran-contiguous layout.
175. Use `np.mgrid` for dense coordinate grids.
176. Use `np.ogrid` for memory-efficient grids.
177. Compute outer product of vectors.
178. Compute inner product of vectors.
179. Broadcast scalar operations to higher-dim array.
180. Create 3D boolean mask and apply to array.
181. Use `np.take_along_axis()` for advanced indexing.
182. Use `np.put_along_axis()` to modify elements.
183. Use `np.choose()` for multi-condition selection.
184. Implement fast element-wise conditional using `np.where()`.
185. Perform block matrix multiplication with einsum.
186. Perform trace of a batched 3D matrix using `einsum`.
187. Use `einsum` for outer product.
188. Compute covariance matrix using `einsum`.
189. Compute Gram matrix using `einsum`.
190. Compute pairwise Euclidean distance using `einsum`.
191. Vectorize a Python function for arrays.
192. Optimize array computations using in-place operations.
193. Minimize memory allocation with `out` parameter in ufuncs.
194. Compare performance of list vs NumPy array operations.
195. Time NumPy operations with `%timeit`.
196. Use structured arrays for tabular datasets.
197. Load CSV into structured array and manipulate fields.
198. Perform cumulative operations along multiple axes.
199. Perform masked array operations.
200. Combine multiple advanced techniques to implement small data pipeline using NumPy (e.g., generate, normalize, compute statistics, filter, and aggregate).

---

# **Plotly Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Plotly Express, simple plots, figure creation, basic customization*

1. Import `plotly.express` as `px` and print version.
2. Create a simple line plot of `[1,2,3,4]` vs `[10,15,13,17]`.
3. Create a scatter plot of `[1,2,3,4]` vs `[10,15,13,17]`.
4. Add a title to a plot.
5. Label x-axis and y-axis.
6. Change color of markers in scatter plot.
7. Change size of markers in scatter plot.
8. Use Plotly Express to create a bar chart.
9. Change bar colors in a bar chart.
10. Create a histogram using Plotly Express.
11. Adjust number of bins in histogram.
12. Create a box plot.
13. Customize box plot color.
14. Create a violin plot.
15. Customize violin plot color.
16. Create an area chart using `line()` with `fill='tozeroy'`.
17. Create a simple Pie chart.
18. Add labels and values to Pie chart.
19. Customize Pie chart colors.
20. Pull out a slice in Pie chart using `pull`.
21. Create a sunburst chart with hierarchical data.
22. Create a treemap.
23. Create a scatter plot with color mapped to a third variable.
24. Create a scatter plot with size mapped to a variable.
25. Combine color and size mapping in scatter plot.
26. Create a scatter plot with categorical color.
27. Create a bar chart with grouped bars using `barmode='group'`.
28. Create stacked bar chart using `barmode='stack'`.
29. Reverse y-axis in a bar chart.
30. Reverse x-axis in a bar chart.
31. Add hover text to scatter plot.
32. Customize hover template.
33. Display multiple traces in one figure using `px.scatter()` with `color` argument.
34. Display multiple traces using `px.line()` with `line_dash` argument.
35. Customize marker symbols in scatter plot.
36. Change line style in line plot.
37. Change line width in line plot.
38. Customize figure size using `width` and `height`.
39. Update layout title font size and family.
40. Update axis titles font and color.
41. Update axes range.
42. Add gridlines to a plot.
43. Remove gridlines.
44. Show minor ticks.
45. Update legend position.
46. Hide legend.
47. Update legend title.
48. Customize legend marker size and symbol.
49. Export figure as HTML.
50. Export figure as static PNG.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Plotly Graph Objects, multiple traces, subplots, interactive features*

51. Import `plotly.graph_objects` as `go`.
52. Create a line trace using `go.Scatter`.
53. Create a scatter trace using `go.Scatter`.
54. Create a bar trace using `go.Bar`.
55. Combine multiple traces in one figure.
56. Create a figure using `go.Figure()`.
57. Add traces to figure using `add_trace()`.
58. Add a layout title using `update_layout()`.
59. Add x-axis and y-axis labels using `update_layout()`.
60. Customize x-axis tick format.
61. Customize y-axis tick format.
62. Change marker color for one trace.
63. Change line color for one trace.
64. Change marker size for one trace.
65. Change line width for one trace.
66. Change line dash style.
67. Add hover info to trace.
68. Customize hover template for one trace.
69. Update legend position in figure layout.
70. Customize legend font and marker size.
71. Add annotation to figure.
72. Add multiple annotations.
73. Use arrows in annotations.
74. Update figure background color.
75. Update plot area background color.
76. Create multiple subplots using `make_subplots()`.
77. Add traces to specific subplot using `row` and `col`.
78. Update subplot titles.
79. Share x-axis among subplots.
80. Share y-axis among subplots.
81. Update subplot layout padding using `update_layout()`.
82. Combine line and scatter traces in one subplot.
83. Combine bar and line traces in one subplot.
84. Use secondary y-axis in subplot.
85. Add multiple traces to secondary y-axis.
86. Use `update_xaxes()` to customize one subplot’s x-axis.
87. Use `update_yaxes()` to customize one subplot’s y-axis.
88. Reverse x-axis in one subplot.
89. Reverse y-axis in one subplot.
90. Add vertical line using `go.layout.Shape(type='line')`.
91. Add horizontal line using `go.layout.Shape`.
92. Add rectangle shape for highlighting region.
93. Add circle shape to highlight point.
94. Add ellipse shape.
95. Add polygon shape.
96. Update shape color and opacity.
97. Add interactive buttons using `updatemenus`.
98. Add dropdown menu to switch traces.
99. Add slider to animate plot.
100. Animate line plot over time using `frame` argument.

…*(questions 101–130 continue with medium-level interactions: choropleth maps, scatter_3d, line_3d, updating layout dynamically, linking traces, hoverlabel customization, multi-axis annotations, shapes and layers, grouped/stacked bar interactivity, subplots with different types, combined charts, legend customization, responsive layouts)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: 3D plots, maps, animations, advanced interactions, dashboards*

131. Create a 3D scatter plot using `px.scatter_3d()`.
132. Customize marker size and color in 3D scatter.
133. Add animation frames to 3D scatter.
134. Create a 3D line plot using `go.Scatter3d`.
135. Add multiple 3D traces to one figure.
136. Customize 3D axes titles and ranges.
137. Rotate camera view in 3D plot.
138. Update camera projection type.
139. Create 3D surface plot using `go.Surface`.
140. Customize surface colors using colormap.
141. Add contour to 3D surface plot.
142. Customize lighting on 3D surface.
143. Add text annotations to 3D plot.
144. Create choropleth map using Plotly Express.
145. Customize colorscale for choropleth.
146. Add hover data to map.
147. Add map projection options.
148. Create scatter geo map.
149. Customize marker size and color on map.
150. Add multiple traces to geo map.
151. Animate map over time.
152. Create density map using `px.density_mapbox`.
153. Customize mapbox style.
154. Add custom mapbox token.
155. Create bubble map with multiple sizes.
156. Create parallel coordinates plot using `px.parallel_coordinates`.
157. Customize line color in parallel coordinates.
158. Add categorical coloring.
159. Create sankey diagram using `go.Sankey`.
160. Customize link colors in sankey.
161. Add node labels and colors.
162. Create sunburst plot using `px.sunburst`.
163. Customize color scale in sunburst.
164. Add hover data to sunburst.
165. Create treemap using `px.treemap`.
166. Combine sunburst and treemap in dashboard layout.
167. Create radar chart (polar plot) using `go.Scatterpolar`.
168. Customize radial axes and angular axes.
169. Fill area under polar plot.
170. Add multiple polar traces.
171. Animate polar plot over time.
172. Create funnel chart using `px.funnel`.
173. Customize funnel colors.
174. Add multiple funnel traces.
175. Create indicator chart using `go.Indicator`.
176. Update value and delta properties of indicator.
177. Combine multiple indicators in one figure.
178. Use subplot for indicators and charts together.
179. Create timeline visualization using Gantt chart (`px.timeline`).
180. Update start and end dates in timeline chart.
181. Color-code timeline bars by category.
182. Animate timeline chart.
183. Add custom hover info to timeline chart.
184. Create waterfall chart using `px.waterfall`.
185. Customize measure type in waterfall.
186. Combine waterfall chart with bar chart in subplot.
187. Create table visualization using `go.Table`.
188. Customize table header colors and fonts.
189. Add multiple tables to one figure.
190. Combine table with chart in dashboard.
191. Use `dash` to embed Plotly figure in interactive app.
192. Add callbacks to update figure based on input.
193. Update figure layout dynamically using dropdowns.
194. Update traces dynamically using sliders.
195. Use buttons to toggle traces visibility.
196. Animate multiple traces over time.
197. Combine choropleth, scatter, and line plots in dashboard.
198. Add annotations and shapes to dashboard layout.
199. Export fully interactive dashboard to HTML.
200. Build complete Plotly dashboard with multiple charts, maps, 3D plots, interactive filters, and annotations.

---

# **Seaborn Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, data loading, simple plots, styling, basic customization*

1. Import `seaborn` as `sns` and print version.
2. Load built-in dataset `tips` using `sns.load_dataset()`.
3. Display the first 5 rows of `tips`.
4. Create a simple scatter plot with `sns.scatterplot()`.
5. Create a simple line plot with `sns.lineplot()`.
6. Create a simple bar plot with `sns.barplot()`.
7. Create a count plot using `sns.countplot()`.
8. Create a box plot with `sns.boxplot()`.
9. Create a violin plot with `sns.violinplot()`.
10. Create a strip plot with `sns.stripplot()`.
11. Create a swarm plot with `sns.swarmplot()`.
12. Add `hue` to scatter plot to color by category.
13. Add `style` to scatter plot to differentiate markers.
14. Add `size` to scatter plot for numeric variable.
15. Add a title to a Seaborn plot using `plt.title()`.
16. Change x-axis and y-axis labels using `plt.xlabel()` and `plt.ylabel()`.
17. Change figure size using `plt.figure(figsize=(width,height))`.
18. Set Seaborn style to `'darkgrid'`.
19. Set Seaborn style to `'whitegrid'`.
20. Set Seaborn style to `'ticks'`.
21. Remove gridlines using `'white'` style.
22. Set color palette to `'deep'`.
23. Set color palette to `'muted'`.
24. Set color palette to `'bright'`.
25. Use `sns.set_context('talk')` to adjust figure context.
26. Use `sns.set_context('notebook')`.
27. Use `sns.set_context('paper')`.
28. Change marker size in scatter plot.
29. Change line width in line plot.
30. Change box plot colors using `palette`.
31. Change violin plot colors using `palette`.
32. Use `hue` in bar plot to show subcategories.
33. Display numerical aggregation (mean) in bar plot.
34. Change estimator function in bar plot to `np.sum`.
35. Create horizontal bar plot by swapping `x` and `y`.
36. Rotate x-axis tick labels.
37. Adjust y-axis tick labels font size.
38. Change figure background color.
39. Change axes background color.
40. Remove axes spines using `sns.despine()`.
41. Remove top and right spines only.
42. Keep left spine only.
43. Show gridlines while removing spines.
44. Add jitter to strip plot.
45. Adjust jitter size in strip plot.
46. Control marker transparency in scatter plot using `alpha`.
47. Display multiple scatter plots in one figure using `plt.subplot()`.
48. Save Seaborn figure as PNG.
49. Save figure as SVG.
50. Show figure using `plt.show()`.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Multi-variable visualizations, aggregation, categorical plots, regression*

51. Create a scatter plot with regression line using `sns.regplot()`.
52. Fit polynomial regression using `order=2` in `regplot`.
53. Fit lowess regression using `lowess=True`.
54. Display confidence interval in regression using `ci=95`.
55. Remove confidence interval in regression using `ci=None`.
56. Create residual plot using `sns.residplot()`.
57. Create joint plot using `sns.jointplot()` with scatter kind.
58. Use `kind='hex'` in joint plot.
59. Use `kind='kde'` in joint plot.
60. Use `kind='reg'` in joint plot.
61. Create pair plot using `sns.pairplot()`.
62. Add `hue` to pair plot.
63. Choose subset of variables in pair plot using `vars`.
64. Drop upper triangle in pair plot.
65. Create categorical box plot using `x` and `y`.
66. Use `hue` in categorical box plot.
67. Show swarm plot over box plot for distribution.
68. Show strip plot over violin plot for distribution.
69. Use `inner='quartile'` in violin plot.
70. Use `inner='stick'` in violin plot.
71. Use `split=True` in violin plot for two categories.
72. Aggregate categorical data using `sns.barplot()`.
73. Customize estimator function to `np.median`.
74. Create count plot with hue for subcategories.
75. Change palette in count plot.
76. Create factor plot using `sns.catplot()`.
77. Use `kind='strip'` in catplot.
78. Use `kind='swarm'` in catplot.
79. Use `kind='box'` in catplot.
80. Use `kind='violin'` in catplot.
81. Use `kind='bar'` in catplot.
82. Create facet grid using `sns.FacetGrid()`.
83. Map scatter plot to facet grid using `map()`.
84. Map histogram to facet grid using `map()`.
85. Map KDE plot to facet grid using `map()`.
86. Use `col` in facet grid for column facets.
87. Use `row` in facet grid for row facets.
88. Use `hue` in facet grid.
89. Control figure size in facet grid.
90. Adjust spacing between facets.
91. Add titles to facet grid.
92. Create a heatmap using `sns.heatmap()`.
93. Customize heatmap colors using `cmap`.
94. Annotate heatmap with values using `annot=True`.
95. Format annotations in heatmap.
96. Mask upper triangle in heatmap.
97. Mask lower triangle in heatmap.
98. Control line widths in heatmap.
99. Add color bar to heatmap.
100. Remove color bar from heatmap.

…*(questions 101–130 continue with medium-level: correlation heatmaps, clustered heatmaps, kde plots, rug plots, multiple regression overlays, facet grids with multiple plot types, swarm/strip overlays, categorical ordering, palette management, axis scaling, normalization, and data transformations)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Complex multi-plot arrangements, advanced customizations, statistical modeling, dashboards*

131. Create clustermap using `sns.clustermap()`.
132. Customize row and column clustering.
133. Normalize data in clustermap.
134. Annotate values in clustermap.
135. Change color palette in clustermap.
136. Adjust dendrogram line width.
137. Create kde plot using `sns.kdeplot()`.
138. Fill area under KDE curve.
139. Overlay multiple KDE plots.
140. Use cumulative KDE.
141. Change bandwidth in KDE plot.
142. Use rug plot to show individual points.
143. Overlay KDE and rug plot.
144. Create 2D KDE plot using `sns.kdeplot()`.
145. Add contour lines to 2D KDE plot.
146. Fill contours in 2D KDE plot.
147. Use scatter plot overlay on 2D KDE.
148. Create joint plot with 2D KDE kind.
149. Add marginal histograms to joint plot.
150. Add marginal box plots to joint plot.
151. Create multiple subplots with seaborn in one figure.
152. Combine different plot types in one figure (e.g., box + strip).
153. Control figure size using `plt.figure()`.
154. Share x-axis across multiple seaborn plots.
155. Share y-axis across multiple seaborn plots.
156. Rotate tick labels in complex figure.
157. Adjust font sizes globally using `sns.set_context()`.
158. Use custom palette across multiple plots.
159. Save multiple seaborn figures programmatically.
160. Annotate multiple points on scatter plot.
161. Add regression line with multiple categories.
162. Customize confidence intervals for multiple categories.
163. Plot interaction effects using `catplot()`.
164. Plot violin + swarm overlay for multiple subplots.
165. Plot box + strip overlay with multiple subplots.
166. Customize figure titles for multi-plot figures.
167. Use seaborn with pandas categorical dtype for ordering.
168. Sort categories in plots manually.
169. Reverse category order.
170. Apply logarithmic scaling to y-axis.
171. Apply log scaling to x-axis.
172. Create facet grid with custom row and column ordering.
173. Adjust spacing between facet grid plots.
174. Annotate facet grid with titles.
175. Overlay multiple plots with hue in facet grid.
176. Map different plot types to different facets.
177. Use seaborn with large datasets efficiently.
178. Reduce figure complexity with sample data.
179. Apply smoothing to line plots.
180. Plot confidence intervals with multiple categories.
181. Combine line plot + bar plot in one figure.
182. Create heatmap with hierarchical clustering.
183. Apply masks to heatmaps for selective visualization.
184. Customize annotations and formatting in heatmaps.
185. Use diverging color palettes for heatmaps.
186. Create multi-index heatmap from pivot table.
187. Combine seaborn plots with matplotlib annotations.
188. Add custom text to figure using `plt.text()`.
189. Overlay shapes on seaborn plots using `plt.axvline()` or `axhline()`.
190. Highlight regions in seaborn plots.
191. Customize legend across multiple seaborn plots.
192. Move legend to custom position.
193. Update legend font and style.
194. Combine multiple datasets in one plot using seaborn.
195. Handle missing data in seaborn plots.
196. Apply log transformation to numeric variables.
197. Normalize numeric variables before plotting.
198. Plot multiple regression lines with `lmplot()`.
199. Use `col` and `row` in `lmplot()` for faceting.
200. Build complete dashboard-like figure using seaborn: multiple plots, facets, overlays, heatmaps, and custom annotations.

---

# **NLTK Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, corpora, basic text handling, tokenization*

1. Install NLTK and import it using `import nltk`.
2. Check NLTK version.
3. Download the `punkt` tokenizer.
4. Download the `stopwords` corpus.
5. Download the `wordnet` corpus.
6. Load a built-in text corpus, e.g., `nltk.corpus.gutenberg`.
7. Display file IDs in `gutenberg`.
8. Read raw text from one file.
9. Tokenize text into sentences using `sent_tokenize()`.
10. Tokenize text into words using `word_tokenize()`.
11. Convert all words to lowercase.
12. Remove punctuation from tokenized words.
13. Count the number of words in a text.
14. Count the number of sentences in a text.
15. Get the first 10 words of a text.
16. Get the last 10 words of a text.
17. Slice tokens from position 10 to 20.
18. Create a frequency distribution of words using `FreqDist`.
19. Find the 10 most common words.
20. Plot the frequency distribution.
21. Remove stopwords using `stopwords.words('english')`.
22. Compute frequency distribution after removing stopwords.
23. Filter out words shorter than 3 characters.
24. Stem words using `PorterStemmer`.
25. Stem words using `LancasterStemmer`.
26. Lemmatize words using `WordNetLemmatizer`.
27. Compare results of stemming vs lemmatization.
28. Find synonyms of a word using WordNet.
29. Find antonyms of a word using WordNet.
30. Get definitions of a word using WordNet.
31. Get part of speech of a word using WordNet.
32. Tokenize a paragraph into sentences, then into words.
33. Identify vocabulary size of a text (unique words).
34. Compute lexical diversity (unique/total words).
35. Find all words starting with a specific letter.
36. Find all words ending with a specific suffix.
37. Find all words containing a substring.
38. Generate bigrams from a tokenized text.
39. Generate trigrams from a tokenized text.
40. Generate n-grams for n=4.
41. Count frequency of bigrams.
42. Count frequency of trigrams.
43. Create conditional frequency distribution (words by category).
44. Plot conditional frequency distribution.
45. Access raw text from the `inaugural` corpus.
46. Access words from `inaugural` corpus.
47. Access sentences from `inaugural` corpus.
48. Explore file IDs in `movie_reviews` corpus.
49. Access words and categories in `movie_reviews`.
50. Compute most frequent words per category in `movie_reviews`.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Tagging, parsing, collocations, concordance, lexical analysis*

51. Tokenize a new text using `word_tokenize()`.
52. Apply `nltk.pos_tag()` to the tokenized words.
53. Identify nouns and verbs in a tagged text.
54. Count frequency of each POS tag.
55. Plot frequency distribution of POS tags.
56. Extract proper nouns (NNP) from a text.
57. Extract adjectives (JJ) from a text.
58. Extract adverbs (RB) from a text.
59. Create a named entity tree using `nltk.ne_chunk()`.
60. Identify named entities in a text.
61. Convert named entity tree to list of entities.
62. Chunk a text using a custom grammar.
63. Define a grammar to extract noun phrases.
64. Apply `RegexpParser` with custom grammar.
65. Extract all noun phrases from parsed text.
66. Extract verb phrases using chunking.
67. Perform shallow parsing on a paragraph.
68. Use `ConcordanceIndex` to find occurrences of a word.
69. Display concordance for a word in a corpus.
70. Find similar words using `text.similar()`.
71. Find common contexts of words using `text.common_contexts()`.
72. Compute dispersion plot of words in a text.
73. Use `Collocations` to find common bigrams.
74. Find collocations in a text after filtering stopwords.
75. Compute PMI (Pointwise Mutual Information) for bigrams.
76. Generate trigrams and compute their frequencies.
77. Identify hapaxes (words occurring once).
78. Compute average word length in a text.
79. Compute average sentence length.
80. Compute readability metrics (basic: words per sentence).
81. Normalize text (lowercase, remove punctuation).
82. Create a custom tokenizer using regex.
83. Apply regex tokenizer to extract email addresses.
84. Extract hashtags from a social media text.
85. Extract URLs from text using regex tokenizer.
86. Apply tokenization to multiple documents in a loop.
87. Build a vocabulary from multiple texts.
88. Create a frequency distribution across multiple texts.
89. Filter tokens by frequency threshold.
90. Apply lemmatization to filtered tokens.
91. Create a POS-tagged frequency distribution.
92. Compute frequency of verbs across a corpus.
93. Compute frequency of nouns across a corpus.
94. Compare lexical diversity of two corpora.
95. Find common words across multiple corpora.
96. Find words unique to one corpus.
97. Extract context windows around a target word.
98. Compute bigram frequency distribution per category.
99. Compute trigram frequency distribution per category.
100. Display top 10 collocations per category.

…*(questions 101–130 continue with medium-level NLP: custom tagging, regular expression tagging, backoff tagging, training unigram/bigram POS taggers, simple parsing, dependency extraction, chunking with custom grammars, named entity recognition from different corpora, and frequency comparisons across categories)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: NLP pipelines, classifiers, text normalization, advanced corpus handling, feature extraction*

131. Train a unigram POS tagger using `nltk.UnigramTagger()`.
132. Train a bigram POS tagger with backoff to unigram.
133. Evaluate POS tagger accuracy on test corpus.
134. Tag unknown words and handle unknowns with backoff tagger.
135. Create a custom tokenizer for multi-word expressions.
136. Tokenize a text preserving multi-word expressions.
137. Build a simple sentiment analysis dataset.
138. Extract features for classification (word presence).
139. Extract POS features for classification.
140. Train a Naive Bayes classifier on text data.
141. Evaluate classifier accuracy.
142. Extract n-grams as features for classification.
143. Apply feature selection to reduce vocabulary.
144. Train classifier using bigram features.
145. Train classifier using TF-IDF features (with NLTK + scikit-learn).
146. Classify new sentences using trained model.
147. Build a confusion matrix for classifier.
148. Plot confusion matrix.
149. Apply stemming as preprocessing before classification.
150. Apply lemmatization before classification.
151. Remove stopwords before classification.
152. Normalize text before classification.
153. Handle punctuation in text preprocessing.
154. Handle numbers in text preprocessing.
155. Convert text to lowercase.
156. Apply regex for custom text cleaning.
157. Build corpus from multiple text files.
158. Build labeled dataset from folder structure.
159. Serialize preprocessed text to disk.
160. Load serialized corpus.
161. Compute TF-IDF manually using NLTK functions.
162. Find top TF-IDF words per document.
163. Compute document similarity using cosine similarity.
164. Identify most similar documents in a corpus.
165. Build word co-occurrence matrix.
166. Compute PMI for word pairs across corpus.
167. Build word network graph from co-occurrence.
168. Apply collocation measures (PMI, chi-squared) to find strong associations.
169. Train a simple bigram language model.
170. Train a trigram language model.
171. Generate text using trained n-gram model.
172. Compute perplexity of a trained n-gram model.
173. Use `ConditionalFreqDist` for word prediction.
174. Compute conditional probabilities for next-word prediction.
175. Build a simple chatbot using NLTK.
176. Use regex patterns for chatbot responses.
177. Extract keywords from user input for chatbot.
178. Build context-aware responses using POS tagging.
179. Build rule-based named entity extraction.
180. Extract dates from text using regex and NER.
181. Extract locations using NER.
182. Extract organizations using NER.
183. Merge multiple corpora into one for analysis.
184. Compare word frequency distributions across corpora.
185. Compare lexical diversity across corpora.
186. Visualize frequency distributions using `matplotlib` + NLTK.
187. Visualize conditional frequency distributions.
188. Build cumulative frequency plots.
189. Plot Zipf’s law for a corpus.
190. Compute word length distributions.
191. Compute sentence length distributions.
192. Compute readability metrics for multiple texts.
193. Perform topic modeling using NLTK + Gensim (basic integration).
194. Preprocess text for topic modeling (tokenize, remove stopwords, lemmatize).
195. Build dictionary and corpus for topic modeling.
196. Visualize top words per topic.
197. Integrate NLTK preprocessing with scikit-learn vectorizers.
198. Build end-to-end text classification pipeline.
199. Apply pipeline to multiple categories and evaluate performance.
200. Build full NLP workflow: text preprocessing, tokenization, tagging, feature extraction, classification, evaluation, and visualization.

---

# **Statsmodels Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, datasets, model basics, OLS regression*

1. Install Statsmodels and import it using `import statsmodels.api as sm`.
2. Check Statsmodels version.
3. Import `statsmodels.formula.api` as `smf`.
4. Load built-in dataset `mtcars` (or `dataset = sm.datasets.get_rdataset('mtcars').data`).
5. Display first 5 rows of dataset.
6. Display dataset summary using `.describe()`.
7. Check dataset column names.
8. Check data types of columns.
9. Plot scatter of `mpg` vs `wt` using matplotlib.
10. Plot scatter of `mpg` vs `hp`.
11. Fit simple OLS regression: `mpg ~ wt`.
12. Print regression summary using `.summary()`.
13. Extract coefficients of the model.
14. Extract R-squared of the model.
15. Extract adjusted R-squared.
16. Extract p-values of coefficients.
17. Extract standard errors of coefficients.
18. Predict new values using `.predict()`.
19. Plot fitted line on scatter plot.
20. Fit multiple linear regression: `mpg ~ wt + hp`.
21. Add interaction term: `mpg ~ wt * hp`.
22. Fit model using categorical variable: `mpg ~ factor(cyl)`.
23. Fit model using log transformation: `mpg ~ np.log(wt)`.
24. Fit model using polynomial term: `mpg ~ I(wt**2)`.
25. Fit model using formula interface with multiple predictors.
26. Extract confidence intervals of coefficients.
27. Extract residuals of model.
28. Extract fitted values.
29. Plot residuals vs fitted values.
30. Plot histogram of residuals.
31. Plot Q-Q plot of residuals using `sm.qqplot()`.
32. Check assumptions: linearity by residual plot.
33. Check assumptions: normality by Q-Q plot.
34. Check assumptions: homoscedasticity visually.
35. Calculate variance inflation factor (VIF) manually.
36. Identify multicollinearity using VIF > 10.
37. Drop predictor and refit model to reduce multicollinearity.
38. Fit weighted least squares (WLS) regression.
39. Fit robust regression using `RLM`.
40. Fit regression with missing data using `dropna()`.
41. Fit regression with missing data using `fillna()`.
42. Use `add_constant()` to include intercept.
43. Compare models using AIC.
44. Compare models using BIC.
45. Predict confidence intervals for new data.
46. Predict prediction intervals for new data.
47. Create model formula dynamically using string variables.
48. Extract influence measures using `get_influence()`.
49. Plot leverage vs residuals.
50. Identify high leverage points using `hat_matrix_diag`.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: ANOVA, GLM, logistic regression, model diagnostics, categorical data*

51. Fit logistic regression using `smf.logit()`.
52. Fit logistic regression with multiple predictors.
53. Extract predicted probabilities.
54. Create classification table from logistic model.
55. Compute confusion matrix.
56. Compute accuracy, precision, recall.
57. Plot ROC curve using statsmodels or sklearn.
58. Compute AUC for logistic regression.
59. Fit Probit model.
60. Compare logit vs probit coefficients.
61. Fit GLM with Gaussian family.
62. Fit GLM with Binomial family.
63. Fit GLM with Poisson family.
64. Fit GLM with Negative Binomial family.
65. Specify link functions (logit, probit, log).
66. Extract deviance of GLM.
67. Extract Pearson chi-squared statistic.
68. Fit GLM with robust covariance.
69. Conduct likelihood ratio test for nested models.
70. Perform Wald test for coefficients.
71. Conduct t-test for a single coefficient.
72. Conduct F-test for multiple coefficients.
73. Fit ANOVA model using formula: `y ~ C(group)`.
74. Perform one-way ANOVA.
75. Perform two-way ANOVA.
76. Extract sum of squares (SS) from ANOVA table.
77. Extract mean squares (MS) from ANOVA table.
78. Extract F-statistic and p-value from ANOVA table.
79. Perform post-hoc tests manually using pairwise comparisons.
80. Use Bonferroni correction for multiple comparisons.
81. Fit repeated measures ANOVA.
82. Fit ANCOVA using covariates.
83. Check assumptions of ANOVA: normality of residuals.
84. Check homogeneity of variances.
85. Plot boxplots for group comparison.
86. Fit ordinal regression model.
87. Fit multinomial logistic regression.
88. Predict probabilities for multiple outcome classes.
89. Fit count data regression using Poisson.
90. Handle overdispersion using Negative Binomial.
91. Fit zero-inflated Poisson model.
92. Fit zero-inflated Negative Binomial model.
93. Compare nested models using likelihood ratio test.
94. Fit mixed effects model using `MixedLM`.
95. Specify random intercept in mixed model.
96. Specify random slope in mixed model.
97. Extract variance components from mixed model.
98. Compute ICC (intra-class correlation).
99. Fit generalized mixed model using GLM family.
100. Fit repeated measures mixed model.

…*(questions 101–130 continue with medium-level: model selection, stepwise regression, robust covariance estimators, heteroscedasticity-consistent standard errors, influence measures, Cook’s distance, DFBetas, leverage, correlation matrices, residual diagnostics, partial regression plots, interaction effects, polynomial regression, categorical coding methods, dummy variables, multicollinearity handling)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Time series, ARIMA, VAR, forecasting, advanced statistical models, simulations*

131. Load `AirPassengers` or other time series dataset.
132. Plot time series.
133. Decompose time series into trend, seasonal, residual.
134. Fit AR model using `AR` or `AutoReg`.
135. Fit MA model using `ARIMA` with AR=0.
136. Fit ARMA model using `ARIMA`.
137. Fit ARIMA model with p,d,q parameters.
138. Perform grid search for optimal ARIMA parameters.
139. Fit seasonal ARIMA (SARIMA).
140. Forecast future values using ARIMA model.
141. Plot forecast with confidence intervals.
142. Compute forecast error metrics: MAE, MSE, RMSE.
143. Perform stationarity test using Augmented Dickey-Fuller.
144. Difference non-stationary time series.
145. Plot ACF and PACF.
146. Fit VAR (Vector Autoregression) for multivariate time series.
147. Forecast using VAR model.
148. Compute impulse response functions.
149. Compute forecast error variance decomposition.
150. Fit ARIMA with exogenous variables (ARIMAX).
151. Fit GLS (Generalized Least Squares) model.
152. Fit GLS with autocorrelation structure.
153. Fit WLS with heteroscedastic weights.
154. Fit robust regression using Huber’s T.
155. Fit quantile regression.
156. Extract conditional quantiles from model.
157. Fit survival regression (Cox Proportional Hazards).
158. Plot survival function from model.
159. Fit duration models (Weibull, Exponential).
160. Fit GARCH model for volatility (via statsmodels.tsa).
161. Perform Ljung-Box test for autocorrelation.
162. Compute Durbin-Watson statistic for residuals.
163. Perform Breusch-Pagan test for heteroscedasticity.
164. Perform White’s test for heteroscedasticity.
165. Perform Jarque-Bera test for normality of residuals.
166. Perform Shapiro-Wilk test for residual normality.
167. Fit logistic regression with interaction terms.
168. Fit multinomial logistic regression for multi-class outcomes.
169. Fit Probit mixed effects model.
170. Fit Poisson mixed effects model.
171. Fit Negative Binomial mixed effects model.
172. Simulate data for regression analysis.
173. Fit regression on simulated data and verify coefficients.
174. Perform Monte Carlo simulation for model parameters.
175. Use bootstrap to compute confidence intervals.
176. Fit panel data regression using `PanelOLS`.
177. Fit fixed-effects model.
178. Fit random-effects model.
179. Compare fixed vs random effects using Hausman test.
180. Fit instrumental variable regression using `IV2SLS`.
181. Fit two-stage least squares manually.
182. Compute robust standard errors for IV regression.
183. Conduct hypothesis testing for coefficient equality.
184. Compare nested models using F-test.
185. Compare non-nested models using AIC/BIC.
186. Use Wald test for multiple restrictions.
187. Compute likelihood ratio test for nested models.
188. Conduct simulation study for time series forecasts.
189. Perform out-of-sample validation.
190. Plot predicted vs actual values.
191. Compute prediction intervals for regression.
192. Plot residual diagnostics for time series models.
193. Fit exponential smoothing model.
194. Fit Holt-Winters additive model.
195. Fit Holt-Winters multiplicative model.
196. Forecast using Holt-Winters model.
197. Combine multiple models for ensemble forecasting.
198. Build end-to-end workflow: preprocessing → regression → diagnostics → forecasting.
199. Automate model selection and evaluation for multiple datasets.
200. Build full workflow: exploratory analysis, regression, diagnostics, time series forecasting, visualization, and reporting.

---

# **LightGBM Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, data handling, simple model training, basic evaluation*

1. Install LightGBM using pip and import `lightgbm` as `lgb`.
2. Check LightGBM version.
3. Load a sample dataset (e.g., sklearn’s `load_boston`).
4. Convert dataset to Pandas DataFrame.
5. Split dataset into features (`X`) and target (`y`).
6. Split dataset into train and test sets using `train_test_split`.
7. Create a LightGBM dataset using `lgb.Dataset()`.
8. Train a simple LightGBM model using `lgb.train()` with default parameters.
9. Train a LightGBM classifier for binary classification.
10. Train a LightGBM regressor for regression.
11. Print model parameters.
12. Make predictions on training data.
13. Make predictions on test data.
14. Evaluate regression using RMSE.
15. Evaluate regression using MAE.
16. Evaluate classification using accuracy.
17. Evaluate classification using AUC-ROC.
18. Plot ROC curve.
19. Plot Precision-Recall curve.
20. Compute confusion matrix.
21. Plot feature importance using `plot_importance()`.
22. Extract feature importance values programmatically.
23. Save trained model to file.
24. Load model from file.
25. Update model with additional training data.
26. Use `early_stopping_rounds` for model training.
27. Use validation data in `lgb.train()`.
28. Set `num_boost_round` manually.
29. Set `learning_rate` parameter.
30. Set `max_depth` parameter.
31. Set `num_leaves` parameter.
32. Set `min_data_in_leaf` parameter.
33. Set `feature_fraction` parameter.
34. Set `bagging_fraction` parameter.
35. Set `bagging_freq` parameter.
36. Set `lambda_l1` parameter.
37. Set `lambda_l2` parameter.
38. Set `objective` for regression.
39. Set `objective` for binary classification.
40. Set `objective` for multiclass classification.
41. Specify `metric` for regression.
42. Specify `metric` for classification.
43. Set categorical features manually.
44. Use automatic categorical detection for LightGBM.
45. Handle missing values automatically.
46. Train using GPU (`device='gpu'`).
47. Train using CPU (`device='cpu'`).
48. Extract number of trees used in model.
49. Visualize individual tree using `plot_tree()`.
50. Limit tree depth during visualization.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced model training, hyperparameter tuning, cross-validation, feature engineering*

51. Perform k-fold cross-validation using `lgb.cv()`.
52. Set `nfold=5` in cross-validation.
53. Use early stopping in cross-validation.
54. Use stratified k-fold for classification.
55. Perform grid search manually with `for` loops.
56. Use learning rate scheduler.
57. Use column sampling (`feature_fraction`) in cross-validation.
58. Use row sampling (`bagging_fraction`).
59. Handle categorical variables properly in CV.
60. Extract best number of boosting rounds from CV.
61. Train model using best number of boosting rounds.
62. Tune `num_leaves` for best performance.
63. Tune `max_depth` for best performance.
64. Tune `min_data_in_leaf` for best performance.
65. Tune `learning_rate` for best performance.
66. Tune `feature_fraction` for best performance.
67. Tune `bagging_fraction` for best performance.
68. Tune `bagging_freq` for best performance.
69. Tune `lambda_l1` for best performance.
70. Tune `lambda_l2` for best performance.
71. Use randomized search for hyperparameter tuning.
72. Use Bayesian optimization for hyperparameter tuning.
73. Use Optuna with LightGBM.
74. Save CV results for later analysis.
75. Plot CV metrics over boosting rounds.
76. Visualize feature importance after CV.
77. Extract SHAP values for features.
78. Plot SHAP summary plot.
79. Plot SHAP dependence plot.
80. Identify most impactful features using SHAP.
81. Handle imbalanced dataset by setting `is_unbalance=True`.
82. Handle imbalanced dataset by adjusting `scale_pos_weight`.
83. Train multiclass classification model.
84. Evaluate multiclass classification using logloss.
85. Compute multiclass AUC.
86. Use label encoding for multiclass target.
87. Use one-hot encoding for features.
88. Handle missing values using imputation before training.
89. Generate polynomial features for LightGBM.
90. Generate interaction features.
91. Use target encoding for categorical variables.
92. Use mean encoding for categorical variables.
93. Remove highly correlated features before training.
94. Use PCA for dimensionality reduction.
95. Use feature selection with `SelectKBest`.
96. Create custom metric function.
97. Pass custom metric to `lgb.train()`.
98. Create custom objective function.
99. Pass custom objective to `lgb.train()`.
100. Implement multi-output regression using LightGBM.

…*(questions 101–130 continue with medium-level: advanced cross-validation strategies, nested CV, time-series CV, feature interaction exploration, early stopping with custom metrics, boosting from scratch, combining LightGBM with pipelines, LightGBM with sklearn wrappers, integrating LightGBM with Pandas and NumPy workflows, memory-efficient training for large datasets)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Ensemble methods, stacking, advanced time series, model interpretation, deployment*

131. Train LightGBM model on large dataset using `Dataset` API.
132. Use categorical feature handling for large datasets.
133. Use LightGBM with Dask for distributed training.
134. Use LightGBM with Spark for distributed datasets.
135. Train model incrementally using `init_model`.
136. Combine LightGBM with XGBoost in a stacking ensemble.
137. Combine LightGBM with CatBoost in stacking.
138. Train model for regression, then use residuals in second LightGBM model.
139. Use LightGBM with sklearn `Pipeline`.
140. Use LightGBM as part of voting classifier.
141. Use LightGBM in bagging ensemble.
142. Use LightGBM with cross-validated feature selection.
143. Train multi-step time series forecasting using LightGBM.
144. Train recursive time series model using LightGBM.
145. Use lag features for time series prediction.
146. Use rolling window features.
147. Use expanding window features.
148. Incorporate calendar features (month, day, weekday) into model.
149. Incorporate holidays into model features.
150. Incorporate trend features for time series.
151. Evaluate time series forecasts using RMSE.
152. Evaluate MAPE for time series forecasts.
153. Evaluate MAE for time series forecasts.
154. Plot predicted vs actual time series.
155. Plot residuals over time.
156. Detect anomalies in residuals.
157. Use LightGBM for ranking tasks.
158. Set `objective='lambdarank'` for ranking.
159. Set group parameter for ranking.
160. Evaluate ranking model using NDCG.
161. Evaluate ranking model using MAP.
162. Visualize ranking predictions.
163. Optimize hyperparameters for ranking.
164. Implement LightGBM with early stopping for ranking.
165. Use monotone constraints in LightGBM.
166. Apply monotone constraints for selected features.
167. Apply categorical constraints.
168. Use custom evaluation function for ranking.
169. Train LightGBM on streaming data.
170. Incrementally update LightGBM model with new batches.
171. Extract leaf indices for training data.
172. Use leaf indices for interaction feature engineering.
173. Extract split gain for each feature.
174. Identify weak features using gain.
175. Visualize tree structure for advanced interpretation.
176. Visualize top k trees.
177. Visualize depth of trees.
178. Visualize leaf value distributions.
179. Use SHAP interaction values.
180. Visualize SHAP interaction heatmap.
181. Deploy LightGBM model with pickle.
182. Deploy LightGBM model using joblib.
183. Deploy LightGBM in a REST API.
184. Deploy LightGBM using Flask.
185. Deploy LightGBM using FastAPI.
186. Deploy LightGBM using Streamlit.
187. Monitor model performance over time.
188. Retrain model on new data periodically.
189. Detect concept drift in streaming data.
190. Retrain incrementally on drifted data.
191. Automate hyperparameter tuning pipeline.
192. Combine LightGBM predictions with other models (stacking/blending).
193. Use LightGBM in a Kaggle competition workflow.
194. Use LightGBM with cross-validation folds for robust performance.
195. Combine LightGBM with feature selection for best results.
196. Interpret SHAP values to explain predictions to stakeholders.
197. Compute global feature importance using SHAP.
198. Compute local feature importance for individual predictions.
199. Visualize model predictions with feature explanations.
200. Build full end-to-end workflow: data preprocessing, LightGBM training, hyperparameter tuning, evaluation, interpretation, and deployment.

---

# **XGBoost Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, dataset loading, simple model training, basic evaluation*

1. Install XGBoost using pip and import `xgboost` as `xgb`.
2. Check XGBoost version.
3. Load a sample dataset (e.g., sklearn’s `load_boston`).
4. Convert dataset to Pandas DataFrame.
5. Split dataset into features (`X`) and target (`y`).
6. Split dataset into train and test sets using `train_test_split`.
7. Create DMatrix for training using `xgb.DMatrix()`.
8. Train a simple XGBoost model with default parameters.
9. Train an XGBoost classifier for binary classification.
10. Train an XGBoost regressor for regression.
11. Print model parameters.
12. Make predictions on training data.
13. Make predictions on test data.
14. Evaluate regression using RMSE.
15. Evaluate regression using MAE.
16. Evaluate classification using accuracy.
17. Evaluate classification using AUC-ROC.
18. Plot ROC curve.
19. Plot Precision-Recall curve.
20. Compute confusion matrix.
21. Plot feature importance using `plot_importance()`.
22. Extract feature importance values programmatically.
23. Save trained model to file using `save_model()`.
24. Load model from file using `load_model()`.
25. Update model with additional training data using `xgb.train()`.
26. Use `early_stopping_rounds` for model training.
27. Use validation data in `xgb.train()`.
28. Set `num_boost_round` manually.
29. Set `learning_rate` (eta) parameter.
30. Set `max_depth` parameter.
31. Set `min_child_weight` parameter.
32. Set `gamma` parameter for regularization.
33. Set `subsample` parameter.
34. Set `colsample_bytree` parameter.
35. Set `reg_alpha` parameter.
36. Set `reg_lambda` parameter.
37. Set `objective` for regression.
38. Set `objective` for binary classification.
39. Set `objective` for multiclass classification.
40. Specify `eval_metric` for regression.
41. Specify `eval_metric` for classification.
42. Handle categorical features manually.
43. Handle missing values automatically.
44. Train using GPU (`tree_method='gpu_hist'`).
45. Train using CPU (`tree_method='hist'`).
46. Extract number of trees used in model.
47. Visualize individual tree using `plot_tree()`.
48. Limit tree depth during visualization.
49. Plot multiple trees in one figure.
50. Extract leaf indices from the model.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced model training, cross-validation, hyperparameter tuning, feature engineering*

51. Perform k-fold cross-validation using `xgb.cv()`.
52. Set `nfold=5` in cross-validation.
53. Use early stopping in cross-validation.
54. Use stratified k-fold for classification.
55. Perform grid search manually with `for` loops.
56. Use learning rate scheduler.
57. Use column sampling (`colsample_bytree`) in cross-validation.
58. Use row sampling (`subsample`).
59. Handle categorical variables properly in CV.
60. Extract best number of boosting rounds from CV.
61. Train model using best number of boosting rounds.
62. Tune `max_depth` for best performance.
63. Tune `min_child_weight` for best performance.
64. Tune `gamma` for best performance.
65. Tune `subsample` for best performance.
66. Tune `colsample_bytree` for best performance.
67. Tune `learning_rate` (eta) for best performance.
68. Tune `reg_alpha` for best performance.
69. Tune `reg_lambda` for best performance.
70. Use randomized search for hyperparameter tuning.
71. Use Bayesian optimization for hyperparameter tuning.
72. Use Optuna with XGBoost.
73. Save CV results for later analysis.
74. Plot CV metrics over boosting rounds.
75. Visualize feature importance after CV.
76. Extract SHAP values for features.
77. Plot SHAP summary plot.
78. Plot SHAP dependence plot.
79. Identify most impactful features using SHAP.
80. Handle imbalanced dataset by setting `scale_pos_weight`.
81. Train multiclass classification model.
82. Evaluate multiclass classification using logloss.
83. Compute multiclass AUC.
84. Use label encoding for multiclass target.
85. Use one-hot encoding for features.
86. Handle missing values using imputation before training.
87. Generate polynomial features.
88. Generate interaction features.
89. Use target encoding for categorical variables.
90. Use mean encoding for categorical variables.
91. Remove highly correlated features before training.
92. Use PCA for dimensionality reduction.
93. Use feature selection with `SelectKBest`.
94. Create custom evaluation metric.
95. Pass custom metric to `xgb.train()`.
96. Create custom objective function.
97. Pass custom objective to `xgb.train()`.
98. Implement multi-output regression using XGBoost.
99. Train a regressor with monotone constraints.
100. Visualize partial dependence plot for a feature.

…*(questions 101–130 continue with medium-level: advanced CV strategies, nested CV, time-series CV, feature interaction exploration, early stopping with custom metrics, XGBoost pipelines with sklearn, memory-efficient training, handling sparse matrices, categorical encoding, regularization tuning, incremental training, early stopping diagnostics)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Ensemble methods, stacking, advanced time series, model interpretation, deployment*

131. Train XGBoost model on large dataset using `DMatrix` API.
132. Use categorical feature handling for large datasets.
133. Use XGBoost with Dask for distributed training.
134. Use XGBoost with Spark for distributed datasets.
135. Train model incrementally using `xgb.train()` with `xgb_model`.
136. Combine XGBoost with LightGBM in a stacking ensemble.
137. Combine XGBoost with CatBoost in stacking.
138. Train model for regression, then use residuals in second XGBoost model.
139. Use XGBoost with sklearn `Pipeline`.
140. Use XGBoost as part of voting classifier.
141. Use XGBoost in bagging ensemble.
142. Use XGBoost with cross-validated feature selection.
143. Train multi-step time series forecasting using XGBoost.
144. Train recursive time series model using XGBoost.
145. Use lag features for time series prediction.
146. Use rolling window features.
147. Use expanding window features.
148. Incorporate calendar features (month, day, weekday).
149. Incorporate holidays into model features.
150. Incorporate trend features for time series.
151. Evaluate time series forecasts using RMSE.
152. Evaluate MAPE for time series forecasts.
153. Evaluate MAE for time series forecasts.
154. Plot predicted vs actual time series.
155. Plot residuals over time.
156. Detect anomalies in residuals.
157. Use XGBoost for ranking tasks.
158. Set `objective='rank:pairwise'` for ranking.
159. Set group parameter for ranking.
160. Evaluate ranking model using NDCG.
161. Evaluate ranking model using MAP.
162. Visualize ranking predictions.
163. Optimize hyperparameters for ranking.
164. Implement XGBoost with early stopping for ranking.
165. Use monotone constraints in XGBoost.
166. Apply monotone constraints for selected features.
167. Apply categorical constraints.
168. Use custom evaluation function for ranking.
169. Train XGBoost on streaming data.
170. Incrementally update XGBoost model with new batches.
171. Extract leaf indices for training data.
172. Use leaf indices for interaction feature engineering.
173. Extract split gain for each feature.
174. Identify weak features using gain.
175. Visualize tree structure for advanced interpretation.
176. Visualize top k trees.
177. Visualize depth of trees.
178. Visualize leaf value distributions.
179. Use SHAP interaction values.
180. Visualize SHAP interaction heatmap.
181. Deploy XGBoost model with pickle.
182. Deploy XGBoost model using joblib.
183. Deploy XGBoost in a REST API.
184. Deploy XGBoost using Flask.
185. Deploy XGBoost using FastAPI.
186. Deploy XGBoost using Streamlit.
187. Monitor model performance over time.
188. Retrain model on new data periodically.
189. Detect concept drift in streaming data.
190. Retrain incrementally on drifted data.
191. Automate hyperparameter tuning pipeline.
192. Combine XGBoost predictions with other models (stacking/blending).
193. Use XGBoost in a Kaggle competition workflow.
194. Use XGBoost with cross-validation folds for robust performance.
195. Combine XGBoost with feature selection for best results.
196. Interpret SHAP values to explain predictions to stakeholders.
197. Compute global feature importance using SHAP.
198. Compute local feature importance for individual predictions.
199. Visualize model predictions with feature explanations.
200. Build full end-to-end workflow: data preprocessing, XGBoost training, hyperparameter tuning, evaluation, interpretation, and deployment.

---

# **CatBoost Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, data handling, simple model training, basic evaluation*

1. Install CatBoost using pip and import `catboost`.
2. Check CatBoost version.
3. Load a sample dataset (e.g., sklearn’s `load_boston`).
4. Convert dataset to Pandas DataFrame.
5. Split dataset into features (`X`) and target (`y`).
6. Split dataset into train and test sets using `train_test_split`.
7. Initialize CatBoostRegressor with default parameters.
8. Initialize CatBoostClassifier with default parameters.
9. Train a simple CatBoost model on training data.
10. Make predictions on training data.
11. Make predictions on test data.
12. Evaluate regression using RMSE.
13. Evaluate regression using MAE.
14. Evaluate classification using accuracy.
15. Evaluate classification using AUC-ROC.
16. Plot ROC curve for classifier.
17. Plot Precision-Recall curve.
18. Compute confusion matrix.
19. Print model parameters.
20. Extract feature importance values using `.get_feature_importance()`.
21. Plot feature importance using CatBoost built-in plotting.
22. Save trained model using `.save_model()`.
23. Load model using `.load_model()`.
24. Use `eval_set` for validation during training.
25. Use `early_stopping_rounds` to avoid overfitting.
26. Set `iterations` parameter for number of boosting rounds.
27. Set `learning_rate` parameter.
28. Set `depth` parameter.
29. Set `l2_leaf_reg` parameter.
30. Set `border_count` parameter for numerical features.
31. Set `bagging_temperature` for sampling.
32. Handle categorical features automatically.
33. Specify categorical features manually.
34. Use ordered boosting (`boosting_type='Ordered'`).
35. Use plain boosting (`boosting_type='Plain'`).
36. Handle missing values automatically.
37. Enable verbose training for monitoring.
38. Use `random_seed` for reproducibility.
39. Extract best iteration using `.get_best_iteration()`.
40. Plot loss curve using `.plot_loss_curve()`.
41. Limit trees visualized during plotting.
42. Extract prediction probabilities using `.predict_proba()`.
43. Convert DataFrame to CatBoost Pool.
44. Train model using CatBoost Pool.
45. Monitor evaluation metrics using `eval_metric`.
46. Use multiple evaluation metrics simultaneously.
47. Extract evaluation results programmatically.
48. Train model with GPU (`task_type='GPU'`).
49. Train model with CPU (`task_type='CPU'`).
50. Apply model to new unseen data.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced model training, cross-validation, hyperparameter tuning, feature engineering*

51. Perform k-fold cross-validation using `cv()` function.
52. Set `fold_count=5` in CV.
53. Use stratified folds for classification.
54. Use early stopping in CV.
55. Perform manual grid search for hyperparameters.
56. Perform randomized search for hyperparameters.
57. Tune `learning_rate` for optimal performance.
58. Tune `depth` for optimal performance.
59. Tune `l2_leaf_reg` for optimal performance.
60. Tune `bagging_temperature` for optimal performance.
61. Tune `border_count` for optimal performance.
62. Tune `iterations` for optimal performance.
63. Tune boosting type (`Ordered` vs `Plain`).
64. Use cross-validation to determine best iterations.
65. Use cross-validation for early stopping.
66. Plot CV metrics over iterations.
67. Extract feature importance from CV results.
68. Compute SHAP values for feature interpretation.
69. Plot SHAP summary plot.
70. Plot SHAP dependence plot.
71. Identify most impactful features using SHAP.
72. Handle imbalanced dataset by adjusting `class_weights`.
73. Handle imbalanced dataset using `auto_class_weights`.
74. Train multiclass classification model.
75. Evaluate multiclass model using multi-class logloss.
76. Evaluate multiclass AUC.
77. Use label encoding for multiclass target.
78. Use one-hot encoding for categorical features.
79. Handle missing values using imputation before training.
80. Generate polynomial features.
81. Generate interaction features.
82. Remove highly correlated features before training.
83. Use PCA for dimensionality reduction.
84. Use feature selection (`SelectKBest`) with CatBoost.
85. Use custom loss function.
86. Pass custom evaluation metric.
87. Train model on sparse matrix input.
88. Train model on large dataset using Pool.
89. Extract leaf indices for training data.
90. Use leaf indices for feature engineering.
91. Combine CatBoost with sklearn `Pipeline`.
92. Combine CatBoost with other classifiers in stacking.
93. Combine CatBoost with LightGBM in ensemble.
94. Train regression model with monotone constraints.
95. Train classification model with monotone constraints.
96. Implement early stopping with custom metric.
97. Use multiple evaluation sets simultaneously.
98. Extract per-iteration evaluation results.
99. Visualize per-class feature importance.
100. Visualize cumulative feature importance.

…*(questions 101–130 continue with medium-level: cross-validation strategies, nested CV, time-series CV, categorical encoding techniques, memory-efficient training, incremental training, interaction features, tuning regularization parameters, early stopping diagnostics, feature selection, hyperparameter optimization with Optuna, LightGBM vs CatBoost comparisons)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Ensemble methods, stacking, advanced time series, model interpretation, deployment*

131. Train CatBoost model on large dataset using Pool API.
132. Use categorical feature handling for large datasets.
133. Use CatBoost with Dask for distributed training.
134. Use CatBoost with GPU for large datasets.
135. Train model incrementally using `init_model`.
136. Combine CatBoost with LightGBM in stacking ensemble.
137. Combine CatBoost with XGBoost in stacking.
138. Train model for regression, then use residuals in second CatBoost model.
139. Use CatBoost with sklearn `Pipeline`.
140. Use CatBoost as part of voting classifier.
141. Use CatBoost in bagging ensemble.
142. Use CatBoost with cross-validated feature selection.
143. Train multi-step time series forecasting using CatBoost.
144. Train recursive time series model using CatBoost.
145. Use lag features for time series prediction.
146. Use rolling window features.
147. Use expanding window features.
148. Incorporate calendar features (month, day, weekday).
149. Incorporate holidays into model features.
150. Incorporate trend features for time series.
151. Evaluate time series forecasts using RMSE.
152. Evaluate MAPE for time series forecasts.
153. Evaluate MAE for time series forecasts.
154. Plot predicted vs actual time series.
155. Plot residuals over time.
156. Detect anomalies in residuals.
157. Use CatBoost for ranking tasks.
158. Set `objective='YetiRank'` for ranking.
159. Set group parameter for ranking tasks.
160. Evaluate ranking model using NDCG.
161. Evaluate ranking model using MAP.
162. Visualize ranking predictions.
163. Optimize hyperparameters for ranking.
164. Implement CatBoost with early stopping for ranking.
165. Use monotone constraints in CatBoost.
166. Apply monotone constraints for selected features.
167. Apply categorical constraints.
168. Use custom evaluation function for ranking.
169. Train CatBoost on streaming data.
170. Incrementally update CatBoost model with new batches.
171. Extract leaf indices for training data.
172. Use leaf indices for interaction feature engineering.
173. Extract split gain for each feature.
174. Identify weak features using gain.
175. Visualize tree structure for advanced interpretation.
176. Visualize top k trees.
177. Visualize depth of trees.
178. Visualize leaf value distributions.
179. Compute SHAP interaction values.
180. Visualize SHAP interaction heatmap.
181. Deploy CatBoost model with pickle.
182. Deploy CatBoost model using joblib.
183. Deploy CatBoost in a REST API.
184. Deploy CatBoost using Flask.
185. Deploy CatBoost using FastAPI.
186. Deploy CatBoost using Streamlit.
187. Monitor model performance over time.
188. Retrain model on new data periodically.
189. Detect concept drift in streaming data.
190. Retrain incrementally on drifted data.
191. Automate hyperparameter tuning pipeline.
192. Combine CatBoost predictions with other models (stacking/blending).
193. Use CatBoost in a Kaggle competition workflow.
194. Use CatBoost with cross-validation folds for robust performance.
195. Combine CatBoost with feature selection for best results.
196. Interpret SHAP values to explain predictions to stakeholders.
197. Compute global feature importance using SHAP.
198. Compute local feature importance for individual predictions.
199. Visualize model predictions with feature explanations.
200. Build full end-to-end workflow: data preprocessing, CatBoost training, hyperparameter tuning, evaluation, interpretation, and deployment.

---

# **Optuna Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, simple study creation, optimization basics*

1. Install Optuna using pip and import `optuna`.
2. Check Optuna version.
3. Understand the concept of a “study” in Optuna.
4. Create a simple study using `optuna.create_study()`.
5. Understand objective function structure in Optuna.
6. Write a simple objective function for `y = x^2`.
7. Optimize the objective function using `study.optimize()`.
8. Specify number of trials using `n_trials`.
9. Retrieve the best value using `study.best_value`.
10. Retrieve the best parameters using `study.best_params`.
11. Access all trials using `study.trials`.
12. Access a single trial using `study.trials[0]`.
13. Extract trial value using `trial.value`.
14. Extract trial parameters using `trial.params`.
15. Use `trial.suggest_int()` to select integer hyperparameters.
16. Use `trial.suggest_float()` to select float hyperparameters.
17. Use `trial.suggest_categorical()` to select categorical hyperparameters.
18. Implement a simple two-hyperparameter optimization.
19. Visualize study history using `optuna.visualization.plot_optimization_history()`.
20. Visualize parameter importance using `optuna.visualization.plot_param_importances()`.
21. Visualize parallel coordinate plot using `plot_parallel_coordinate()`.
22. Visualize contour plot for two parameters.
23. Create a new study with `direction='maximize'`.
24. Optimize multiple objectives using multi-objective study.
25. Understand trial states: `TrialState.COMPLETE`.
26. Understand trial states: `TrialState.PRUNED`.
27. Understand trial states: `TrialState.RUNNING`.
28. Prune a trial manually using `trial.should_prune()`.
29. Use `trial.report()` to log intermediate values.
30. Save a study using `RDBStorage` with SQLite.
31. Load a study from a database.
32. Stop optimization early using `timeout`.
33. Use `catch` argument in study optimization.
34. Use `show_progress_bar=True` for progress visualization.
35. Understand difference between `optimize` and `enqueue_trial`.
36. Add a trial manually using `study.enqueue_trial()`.
37. Handle exceptions inside the objective function.
38. Set random seed for reproducibility.
39. Optimize a noisy function and visualize convergence.
40. Extract all intermediate values from trials.
41. Compute average best value across multiple studies.
42. Compare multiple studies using visualization.
43. Plot slice plot using `plot_slice()`.
44. Filter trials by state using `study.trials_dataframe()`.
45. Compute statistics of all trial parameters.
46. Retrieve intermediate values for pruning.
47. Handle categorical variables for ML tuning.
48. Understand how Optuna chooses next trial using TPE sampler.
49. Create a study using `TPESampler()`.
50. Understand alternative samplers: `RandomSampler()`.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced objective design, ML model tuning, hyperparameter spaces, pruning, study analysis*

51. Optimize hyperparameters for `sklearn.GradientBoostingClassifier`.
52. Optimize hyperparameters for `sklearn.RandomForestClassifier`.
53. Optimize hyperparameters for `sklearn.SVC`.
54. Optimize hyperparameters for `sklearn.LogisticRegression`.
55. Optimize hyperparameters for `XGBoostClassifier`.
56. Optimize hyperparameters for `XGBoostRegressor`.
57. Optimize hyperparameters for `LightGBMClassifier`.
58. Optimize hyperparameters for `LightGBMRegressor`.
59. Optimize hyperparameters for `CatBoostClassifier`.
60. Optimize hyperparameters for `CatBoostRegressor`.
61. Use `train_test_split` inside objective function for validation.
62. Use cross-validation inside objective function.
63. Return average metric from CV as objective value.
64. Use multiple hyperparameters for tuning simultaneously.
65. Suggest integer hyperparameters for tree depth.
66. Suggest float hyperparameters for learning rate.
67. Suggest categorical hyperparameters for booster type.
68. Apply pruning based on intermediate metric.
69. Use `optuna.integration.XGBoostPruningCallback` for XGBoost pruning.
70. Use `optuna.integration.LightGBMPruningCallback` for LightGBM pruning.
71. Implement early stopping for CatBoost in Optuna.
72. Track trial durations and compare performance.
73. Use logging to monitor trial progress.
74. Store study results in SQLite for later analysis.
75. Load study and continue optimization.
76. Optimize hyperparameters with constraints.
77. Optimize hyperparameters with conditional logic.
78. Use multi-objective optimization: maximize accuracy, minimize time.
79. Visualize Pareto front for multi-objective optimization.
80. Compare importance of parameters in multi-objective study.
81. Use pruning thresholds to stop unpromising trials early.
82. Use custom sampler strategies for exploration.
83. Implement Optuna study for hyperparameter tuning on time series.
84. Tune lag features for time series model.
85. Tune rolling window sizes.
86. Use nested cross-validation with Optuna.
87. Optimize sklearn pipeline parameters with Optuna.
88. Optimize feature selection thresholds inside objective.
89. Optimize dimensionality reduction components (e.g., PCA n_components).
90. Tune regularization parameters (alpha, lambda) for ML models.
91. Tune dropout rates for neural networks with Optuna.
92. Tune hidden layer sizes in neural networks.
93. Tune learning rate schedules in neural networks.
94. Optimize XGBoost `max_depth` and `min_child_weight`.
95. Optimize LightGBM `num_leaves` and `min_data_in_leaf`.
96. Optimize CatBoost `depth` and `l2_leaf_reg`.
97. Evaluate optimized model on holdout set.
98. Track best trial and parameters programmatically.
99. Save best trial parameters to JSON.
100. Visualize parameter importance for regression tasks.

…*(questions 101–130 continue with medium-level: more ML tuning scenarios, cross-validation integration, complex conditional hyperparameter spaces, pruning strategies, integration with sklearn pipelines, optuna logging, parallel optimization, study storage, time-series optimization, ensemble optimization, nested optimization strategies, validation set handling)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Multi-objective optimization, parallel/distributed optimization, advanced pruning, deployment, integration*

131. Use `MultiObjectiveStudy` for optimizing multiple metrics.
132. Optimize accuracy and inference time simultaneously.
133. Visualize trade-offs in multi-objective study.
134. Use `CmaEsSampler` for advanced sampling.
135. Use `NSGAIISampler` for multi-objective TSP optimization.
136. Run Optuna in parallel with multiprocessing.
137. Run Optuna with multiple machines using RDBStorage.
138. Integrate Optuna with MLflow for experiment tracking.
139. Integrate Optuna with Weights & Biases for logging.
140. Resume interrupted optimization from database.
141. Use pruning for extremely long-running trials.
142. Tune hyperparameters for deep learning model in PyTorch.
143. Tune hyperparameters for TensorFlow/Keras model.
144. Use pruning callback in Keras model training.
145. Track GPU memory usage during tuning.
146. Optimize neural network architectures with Optuna.
147. Use Optuna to find best optimizer and learning rate.
148. Optimize batch size in neural networks.
149. Optimize number of epochs with early stopping.
150. Apply conditional hyperparameters based on model type.
151. Compare multiple study results programmatically.
152. Visualize hyperparameter evolution over trials.
153. Visualize intermediate values for pruning decisions.
154. Use `Trial.suggest_loguniform()` for log-scale hyperparameters.
155. Use `Trial.suggest_discrete_uniform()` for discrete values.
156. Use `Trial.suggest_int(step=…)` for stepwise search.
157. Optimize ensemble weights in stacking models.
158. Optimize hyperparameters for multiple datasets in one study.
159. Optimize multiple ML pipelines simultaneously.
160. Track trial runtime distribution.
161. Visualize early pruning effectiveness.
162. Identify bottlenecks in objective function.
163. Optimize hyperparameters for NLP models (e.g., Transformers).
164. Optimize learning rate scheduler parameters.
165. Tune dropout and attention parameters.
166. Optimize sequence length for NLP model.
167. Optimize optimizer type (Adam, AdamW, SGD).
168. Optimize weight decay.
169. Optimize number of layers and units in deep model.
170. Use Optuna for AutoML tasks.
171. Integrate Optuna with TPOT.
172. Integrate Optuna with AutoKeras.
173. Use Optuna to find best augmentation parameters.
174. Optimize image preprocessing parameters.
175. Use Optuna for feature engineering parameters.
176. Optimize polynomial feature degree.
177. Optimize feature selection thresholds.
178. Use Optuna to prune unpromising feature sets.
179. Track best feature subsets across trials.
180. Store optimized models alongside study results.
181. Deploy best model parameters to production.
182. Automate retraining workflow with Optuna.
183. Schedule periodic re


-optimization using Optuna.
184. Combine Optuna with FastAPI for automated tuning service.
185. Use Optuna for continuous hyperparameter tuning in production.
186. Monitor production model performance and trigger re-optimization.
187. Integrate Optuna with ML monitoring tools (e.g., Evidently).
188. Optimize hyperparameters for reinforcement learning agents.
189. Optimize reward shaping parameters.
190. Optimize environment hyperparameters for RL.
191. Track multiple objective metrics in RL.
192. Prune long RL episodes based on reward.
193. Optimize GAN parameters with Optuna.
194. Optimize generator and discriminator architecture.
195. Optimize learning rates for generator and discriminator.
196. Use Optuna for multi-stage training pipelines.
197. Automate experiment logging, comparison, and reporting.
198. Create a dashboard for Optuna study visualization.
199. Analyze study trends and hyperparameter interactions.
200. Build full end-to-end workflow: define objective, tune ML/DL model, analyze study, interpret results, and deploy best model.

---

# **PyCaret Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, simple experiments, and basic model evaluation*

1. Install PyCaret using pip and import a module (e.g., `classification` or `regression`).
2. Check PyCaret version.
3. Load a sample dataset (e.g., `pycaret.datasets.get_data('diabetes')`).
4. Convert dataset to Pandas DataFrame if not already.
5. Inspect dataset using `.head()` and `.info()`.
6. Identify categorical and numerical features.
7. Identify the target column.
8. Initialize a classification experiment using `setup()`.
9. Initialize a regression experiment using `setup()`.
10. Understand the parameters of `setup()`.
11. Automatically encode categorical variables.
12. Automatically handle missing values.
13. Automatically normalize or scale features.
14. Apply feature transformation (e.g., log transformation).
15. Apply polynomial features.
16. Apply train/test split automatically.
17. Apply custom train/test split ratio.
18. Enable session reproducibility using `session_id`.
19. Understand experiment log output.
20. Compare multiple models using `compare_models()`.
21. Sort models based on metric.
22. Display top N models from comparison.
23. Select the best model automatically.
24. Create a model using `create_model()`.
25. Print model parameters.
26. View model performance using `plot_model()`.
27. Plot AUC curve.
28. Plot Confusion Matrix.
29. Plot Precision-Recall curve.
30. Plot Feature Importance.
31. Plot Residuals (regression).
32. Plot Learning Curve.
33. Plot Prediction Error.
34. Plot Feature Interaction.
35. Plot Class Prediction Probability.
36. Evaluate model metrics with `evaluate_model()`.
37. Make predictions on holdout set using `predict_model()`.
38. Interpret predictions with SHAP using `interpret_model()`.
39. Tune hyperparameters using `tune_model()`.
40. Save a trained model using `save_model()`.
41. Load a saved model using `load_model()`.
42. Finalize model for deployment using `finalize_model()`.
43. Understand difference between `create_model()` and `finalize_model()`.
44. Compare models on multiple metrics.
45. Apply cross-validation inside PyCaret.
46. Understand different cross-validation folds.
47. Automatically handle imbalanced datasets using SMOTE.
48. Manually specify categorical features.
49. Manually specify numeric features.
50. Enable logging of all experiments.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced model training, ensembling, tuning, feature engineering, and experiment optimization*

51. Tune a model’s hyperparameters with `tune_model()`.
52. Specify optimization metric in `tune_model()`.
53. Use Bayesian optimization in tuning.
54. Use random grid search in tuning.
55. Use learning rate search for boosting models.
56. Create ensemble using `blend_models()`.
57. Create a voting ensemble using `stack_models()`.
58. Select meta-model for stacking.
59. Evaluate stacked model performance.
60. Perform bagging ensemble with `create_model()`.
61. Perform boosting ensemble with `create_model()`.
62. Combine multiple feature engineering transformations.
63. Apply feature selection with `feature_selection=True`.
64. Manually select top K features.
65. Apply PCA using `pca=True`.
66. Specify number of PCA components.
67. Apply polynomial features selectively.
68. Apply outlier removal during preprocessing.
69. Handle missing values with specific strategies.
70. Encode target variable for classification.
71. Automatically encode multi-class targets.
72. Compare models using multiple metrics (accuracy, F1, AUC).
73. Sort comparison results by custom metric.
74. Select top N models for further tuning.
75. Automate iterative model tuning pipeline.
76. Plot hyperparameter importance after tuning.
77. Visualize learning curves after tuning.
78. Interpret tuned model with SHAP summary plot.
79. Extract SHAP values for individual predictions.
80. Evaluate regression models using MAE, MSE, RMSE.
81. Evaluate classification models using accuracy, F1, ROC-AUC.
82. Track experiment results programmatically.
83. Save comparison results to CSV.
84. Plot residuals for ensemble models.
85. Plot prediction error for ensemble models.
86. Compare model performances on validation set.
87. Compare model performances on holdout set.
88. Select best model from comparison for deployment.
89. Automate model finalization pipeline.
90. Deploy model in production-ready format.
91. Generate predictions for new unseen data.
92. Save pipeline with preprocessing + model together.
93. Load pipeline for end-to-end predictions.
94. Understand `fold_strategy` in cross-validation.
95. Change `fold_strategy` to time series split.
96. Change `fold_strategy` to stratified KFold.
97. Enable advanced feature interaction.
98. Automatically detect categorical interactions.
99. Apply natural log transformations to skewed features.
100. Apply Yeo-Johnson transformations to numeric features.

…*(questions 101–130 continue with medium-level: hyperparameter tuning for all PyCaret model types, ensembling strategies, conditional tuning, iterative experiments, feature engineering combinations, automatic logging, comparison of multiple pipelines, integration with MLflow, automated pruning, batch predictions, experiment tracking, cross-validation strategies, time-series model tuning, NLP model tuning, regression vs classification specific tuning)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Time series, NLP, deployment, automation, interpretability, production pipelines*

131. Initialize a time-series regression experiment in PyCaret.
132. Apply lag features for time series.
133. Apply rolling mean/median features.
134. Automatically split training and test set for time series.
135. Tune time-series models using `tune_model()`.
136. Compare time-series models using `compare_models()`.
137. Stack time-series models using `stack_models()`.
138. Use XGBoost, LightGBM, and CatBoost for time series.
139. Automate feature engineering for time-series.
140. Apply seasonality features (month, day, weekday).
141. Apply trend features (linear, exponential).
142. Evaluate predictions using MAE, RMSE, MAPE.
143. Plot predicted vs actual values for time series.
144. Deploy time-series model for forecasting.
145. Initialize NLP experiment in PyCaret.
146. Preprocess text (tokenization, lowercasing, stopword removal).
147. Apply TF-IDF feature engineering.
148. Apply word embeddings in PyCaret NLP.
149. Compare multiple NLP models using `compare_models()`.
150. Tune hyperparameters for NLP models.
151. Stack NLP models for improved performance.
152. Evaluate NLP models using F1, accuracy, ROC-AUC.
153. Generate predictions for new text data.
154. Interpret NLP model predictions.
155. Deploy NLP model with preprocessing + model.
156. Automate model deployment using `save_model()` and `load_model()`.
157. Export PyCaret pipeline for API integration.
158. Use PyCaret with FastAPI for deployment.
159. Use PyCaret with Streamlit for visualization.
160. Integrate PyCaret with MLflow for experiment tracking.
161. Automate experiment logging for multiple datasets.
162. Compare multiple datasets with PyCaret experiments.
163. Automate iterative pipeline creation and testing.
164. Perform ensemble of regression and classification pipelines.
165. Analyze feature importance across models.
166. Interpret ensemble predictions using SHAP.
167. Generate model reports automatically.
168. Automate selection of best model across experiments.
169. Apply custom metric in `tune_model()`.
170. Implement iterative hyperparameter tuning.
171. Track experiment metrics programmatically.
172. Visualize performance trends over multiple experiments.
173. Deploy multiple models with ensemble predictions.
174. Automate batch predictions for large datasets.
175. Monitor production model performance.
176. Schedule model retraining automatically.
177. Detect concept drift and trigger retraining.
178. Implement continuous learning pipeline.
179. Integrate PyCaret pipelines with SQL/NoSQL databases.
180. Apply PyCaret pipelines on cloud storage datasets.
181. Export model + pipeline to pickle.
182. Export model + pipeline to joblib.
183. Automate experiment report generation in HTML.
184. Compare experiment results across multiple sessions.
185. Track experiment versioning.
186. Perform cross-validation with custom metrics.
187. Customize preprocessing steps in pipeline.
188. Enable/disable specific feature engineering steps.
189. Automate feature selection across multiple experiments.
190. Apply robust scaling for outlier-heavy datasets.
191. Integrate PyCaret with Optuna for advanced tuning.
192. Use PyCaret for AutoML workflow.
193. Compare PyCaret AutoML with manual pipelines.
194. Evaluate final model on completely new dataset.
195. Visualize model predictions and residuals.
196. Document experiment results automatically.
197. Build end-to-end ML workflow with PyCaret.
198. Deploy workflow as API with preprocessing.
199. Monitor predictions for drift or anomalies.
200. Build full end-to-end automated pipeline: data prep → model training → tuning → ensembling → evaluation → interpretation → deployment → monitoring.

---

# **TPOT Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, simple AutoML runs, and basic evaluation*

1. Install TPOT using pip and import `TPOTClassifier` or `TPOTRegressor`.
2. Check TPOT version.
3. Load a sample dataset (e.g., sklearn’s `load_breast_cancer`).
4. Convert dataset to Pandas DataFrame.
5. Split dataset into features (`X`) and target (`y`).
6. Split dataset into train and test sets using `train_test_split`.
7. Initialize a `TPOTClassifier` with default parameters.
8. Initialize a `TPOTRegressor` with default parameters.
9. Fit TPOT model on training data.
10. Make predictions on test data.
11. Evaluate classification accuracy.
12. Evaluate regression RMSE.
13. Evaluate classification F1 score.
14. Evaluate classification ROC-AUC.
15. Evaluate regression R² score.
16. Print TPOT configuration dictionary.
17. Understand TPOT optimization algorithm (genetic programming).
18. Set `generations` parameter.
19. Set `population_size` parameter.
20. Use default `scoring` metric.
21. Specify custom `scoring` metric.
22. Set `cv` folds.
23. Enable early stopping with `max_time_mins`.
24. Limit training time using `max_time_mins`.
25. Enable verbose output during optimization.
26. Understand the pipeline structure generated by TPOT.
27. Export optimized pipeline to Python script using `export()`.
28. Use TPOT’s `warm_start=True` for incremental optimization.
29. Resume optimization from previous run.
30. Use `random_state` for reproducibility.
31. Evaluate feature importance after pipeline selection.
32. Visualize confusion matrix using predictions.
33. Plot learning curves for the optimized pipeline.
34. Handle missing values in dataset.
35. Handle categorical features with one-hot encoding.
36. Scale numeric features before TPOT optimization.
37. Use `TPOTClassifier` for binary classification.
38. Use `TPOTClassifier` for multiclass classification.
39. Use `TPOTRegressor` for regression tasks.
40. Apply simple hyperparameter constraints in configuration dictionary.
41. Include or exclude specific estimators in configuration.
42. Include or exclude preprocessing operators.
43. Limit pipeline depth.
44. Limit number of operators in pipeline.
45. Export pipeline to `.py` script and examine code.
46. Import exported pipeline for retraining.
47. Evaluate exported pipeline on test data.
48. Enable parallelism with `n_jobs=-1`.
49. Track generation progress in verbose mode.
50. Compare pipeline performance against baseline sklearn models.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Advanced optimization, custom configuration, integration, and pipeline analysis*

51. Customize TPOT configuration dictionary for specific model types.
52. Include only tree-based models in optimization.
53. Include only linear models in optimization.
54. Restrict preprocessing operators (e.g., exclude PCA).
55. Optimize pipeline using custom scoring metrics (F1, RMSE, AUC).
56. Increase number of generations for deeper search.
57. Increase population size for broader exploration.
58. Limit optimization runtime per generation.
59. Use genetic algorithm parameters: mutation rate.
60. Use genetic algorithm parameters: crossover rate.
61. Analyze evolution of pipelines over generations.
62. Plot best score per generation.
63. Evaluate top pipelines on holdout set.
64. Extract final optimized model from TPOT.
65. Inspect hyperparameters of each operator in the pipeline.
66. Integrate TPOT pipelines with sklearn `Pipeline`.
67. Use TPOT for feature selection automatically.
68. Preselect features before TPOT optimization.
69. Remove highly correlated features before optimization.
70. Apply dimensionality reduction before TPOT.
71. Apply PCA in TPOT pipelines.
72. Apply scaling in TPOT pipelines.
73. Apply normalization in TPOT pipelines.
74. Handle imbalanced datasets with SMOTE in TPOT.
75. Use custom preprocessing functions in TPOT pipelines.
76. Save and load TPOT models using `joblib`.
77. Compare multiple TPOT runs for consistency.
78. Apply multi-class classification with TPOT.
79. Optimize pipeline for regression tasks.
80. Analyze feature importance in tree-based pipelines.
81. Interpret linear model coefficients in pipelines.
82. Evaluate cross-validation performance for all pipelines.
83. Analyze variance in pipelines across CV folds.
84. Evaluate TPOT pipelines on unseen datasets.
85. Limit number of pipeline operators to improve speed.
86. Set `memory` parameter to cache intermediate transformations.
87. Use ensemble operators in TPOT pipelines.
88. Enable warm-start for iterative improvement.
89. Export multiple top pipelines for comparison.
90. Track optimization metrics per pipeline.
91. Use TPOT with parallel processing (`n_jobs`).
92. Integrate TPOT pipelines into larger ML workflow.
93. Optimize for multiple metrics in separate runs.
94. Compare performance of tree-based vs linear pipelines.
95. Apply feature interactions automatically.
96. Optimize preprocessing + model jointly.
97. Apply one-hot encoding selectively.
98. Apply categorical feature transformations automatically.
99. Use TPOT for automated hyperparameter tuning of ensemble models.
100. Evaluate improvement over manually tuned sklearn models.

…*(questions 101–130 continue with medium-level: complex pipeline design, feature engineering combinations, conditional operator inclusion, nested CV, advanced scoring metrics, parallel optimization, feature importance interpretation, iterative pipeline refinement, automated exporting and deployment, integration with other AutoML tools, comparison across multiple datasets)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Multi-dataset optimization, time series, NLP, deployment, customization, production pipelines*

131. Use TPOT for multi-dataset optimization.
132. Integrate TPOT with preprocessing pipelines for time series.
133. Create lag features for time series tasks.
134. Apply rolling/expanding features automatically in TPOT.
135. Use TPOT for multi-step forecasting.
136. Optimize pipeline for time series regression.
137. Compare multiple time-series pipelines.
138. Integrate TPOT with NLP feature engineering (TF-IDF, embeddings).
139. Optimize pipelines for text classification tasks.
140. Apply custom tokenization in TPOT NLP pipelines.
141. Evaluate TPOT-generated NLP pipelines on validation data.
142. Export TPOT NLP pipelines for deployment.
143. Automate pipeline retraining with new data.
144. Monitor TPOT pipelines in production.
145. Deploy TPOT pipeline as REST API using Flask.
146. Deploy TPOT pipeline using FastAPI.
147. Deploy TPOT pipeline with Streamlit frontend.
148. Apply batch predictions with TPOT pipelines.
149. Track pipeline performance over time.
150. Detect performance drift and trigger retraining.
151. Automate retraining pipelines with TPOT exports.
152. Combine multiple TPOT-generated pipelines in ensemble.
153. Stack pipelines for better performance.
154. Blend pipelines for improved metrics.
155. Optimize ensemble weights using validation set.
156. Analyze feature importance across pipelines.
157. Extract and interpret SHAP values from tree-based pipelines.
158. Evaluate residuals in regression pipelines.
159. Compare pipeline performance on multiple metrics.
160. Generate automated reports for TPOT experiments.
161. Use TPOT with custom genetic algorithm parameters.
162. Tune mutation and crossover rates for better optimization.
163. Restrict or enforce specific operators in pipeline evolution.
164. Apply warm-start for incremental improvements.
165. Track optimization progress programmatically.
166. Visualize pipeline evolution using matplotlib or seaborn.
167. Evaluate multiple exports for robustness.
168. Save multiple top pipelines for experimentation.
169. Integrate TPOT with MLflow for experiment tracking.
170. Integrate TPOT with Weights & Biases.
171. Automate hyperparameter tuning with TPOT and Optuna hybrid.
172. Optimize preprocessing + model selection jointly.
173. Automate feature selection and engineering using TPOT operators.
174. Evaluate generalization on completely new datasets.
175. Handle imbalanced datasets with custom pipelines.
176. Evaluate F1 score, precision, recall for classification pipelines.
177. Evaluate R², RMSE, MAE for regression pipelines.
178. Deploy ensemble pipelines for production use.
179. Automate retraining based on data drift.
180. Schedule TPOT AutoML runs with cron or scheduler.
181. Integrate TPOT pipelines with SQL/NoSQL databases.
182. Apply TPOT pipelines to cloud-based datasets (S3, GCS).
183. Track pipeline runtime and resource usage.
184. Optimize pipelines under runtime constraints.
185. Optimize pipelines under memory constraints.
186. Customize TPOT operator set for specific domain.
187. Compare TPOT results with PyCaret/Optuna/LGBM/XGBoost workflows.
188. Export optimized pipeline as reusable Python module.
189. Automate versioning of exported pipelines.
190. Create dashboard for TPOT optimization progress.
191. Autom


ate reporting for multiple TPOT experiments.
192. Combine TPOT with preprocessing + postprocessing scripts.
193. Integrate TPOT in CI/CD workflow for ML.
194. Evaluate effect of pipeline depth on performance.
195. Apply TPOT to multi-class classification with many classes.
196. Apply TPOT to multi-output regression tasks.
197. Optimize TPOT pipeline for sparse datasets.
198. Optimize TPOT pipeline for high-dimensional datasets.
199. Track and compare multiple TPOT runs for reproducibility.
200. Build full end-to-end AutoML workflow: dataset preprocessing → TPOT optimization → pipeline export → evaluation → deployment → monitoring.

---

# **Common AutoML Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, dataset handling, basic training and evaluation*

1. Install the AutoML framework.
2. Check version of the framework.
3. Import main AutoML class (`PyCaret: setup/create_model`, `H2O: H2OAutoML`, `TPOT: TPOTClassifier/Regressor`, `Auto-sklearn: AutoSklearnClassifier/Regressor`, `FLAML: AutoML`).
4. Load a sample dataset.
5. Convert dataset to Pandas DataFrame if required.
6. Split dataset into features (`X`) and target (`y`).
7. Split dataset into train and test sets.
8. Handle missing values automatically.
9. Handle categorical features automatically.
10. Encode categorical variables.
11. Normalize or scale numeric features automatically.
12. Initialize AutoML model with default parameters.
13. Train AutoML model on training data.
14. Make predictions on test data.
15. Evaluate regression models using RMSE.
16. Evaluate regression models using MAE.
17. Evaluate classification models using accuracy.
18. Evaluate classification models using F1 score.
19. Evaluate classification models using ROC-AUC.
20. Compare multiple models generated by AutoML.
21. Sort models by performance metric.
22. Display top N models.
23. Select best model automatically.
24. Print model parameters.
25. Access training logs/output.
26. Access cross-validation metrics.
27. Use default train/test split.
28. Use custom train/test split ratio.
29. Use random seed for reproducibility.
30. Track AutoML training time.
31. Limit AutoML training time.
32. Limit maximum iterations or generations.
33. Enable verbose output during training.
34. Save trained model/pipeline.
35. Load trained model/pipeline.
36. Finalize model/pipeline for deployment.
37. Make predictions on new/unseen data.
38. Evaluate holdout dataset performance.
39. Apply simple feature preprocessing.
40. Apply automated feature preprocessing.
41. Plot performance metrics (if supported).
42. Plot confusion matrix.
43. Plot ROC curve.
44. Plot Precision-Recall curve.
45. Track best trial/model automatically.
46. Export pipeline/code for production.
47. Handle imbalanced datasets automatically.
48. Apply cross-validation automatically.
49. Evaluate cross-validation performance.
50. Extract feature importance values.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Hyperparameter tuning, pipeline customization, ensembling, advanced feature handling*

51. Tune hyperparameters automatically.
52. Specify optimization metric.
53. Limit search space for hyperparameters.
54. Apply Bayesian optimization (if supported).
55. Apply grid search (if supported).
56. Apply random search.
57. Use multiple evaluation metrics.
58. Enable early stopping for unpromising models.
59. Set maximum model complexity (e.g., depth, iterations).
60. Include/exclude specific model types.
61. Include/exclude specific preprocessing operators.
62. Apply feature selection automatically.
63. Apply feature selection manually.
64. Apply PCA/dimensionality reduction.
65. Apply polynomial or interaction features.
66. Handle high-cardinality categorical features.
67. Handle sparse datasets.
68. Handle numeric features with skewed distribution.
69. Apply target encoding for categorical features.
70. Apply one-hot encoding for categorical features.
71. Compare ensemble models.
72. Create voting ensemble.
73. Create stacking ensemble.
74. Combine multiple models in pipeline.
75. Evaluate ensemble models.
76. Extract ensemble model parameters.
77. Track model performance per fold.
78. Use k-fold cross-validation.
79. Use stratified k-fold for classification.
80. Evaluate cross-validation mean metrics.
81. Evaluate cross-validation standard deviation.
82. Track best hyperparameter configuration.
83. Visualize hyperparameter importance.
84. Export top N pipelines/models.
85. Import exported pipelines/models for retraining.
86. Apply custom train/test split inside AutoML.
87. Track intermediate metrics for early stopping/pruning.
88. Limit training resources (CPU/GPU).
89. Limit memory usage.
90. Track number of models trained.
91. Track time per model.
92. Apply automated preprocessing + model selection pipeline.
93. Handle class imbalance with SMOTE/oversampling.
94. Evaluate multi-class classification performance.
95. Evaluate multi-output regression.
96. Evaluate multi-label classification.
97. Track best trial/model in multi-objective scenarios.
98. Apply automated feature interaction detection.
99. Track per-operator or per-estimator performance.
100. Export best pipeline as Python code.

…*(questions 101–130 continue with medium-level: advanced pipeline customization, iterative AutoML runs, nested CV, automated feature engineering combinations, logging and experiment tracking, integration with sklearn pipelines, conditional hyperparameter search, optimization under time/memory constraints, comparison across multiple datasets)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Multi-dataset, NLP, time series, deployment, production pipelines, automation, interpretability*

131. Apply AutoML to time series regression.
132. Create lag features automatically.
133. Apply rolling/expanding window features.
134. Apply seasonality features (month, day, weekday).
135. Evaluate time series forecasts using RMSE/MAE/MAPE.
136. Apply AutoML for NLP text classification.
137. Preprocess text automatically.
138. Apply TF-IDF vectorization.
139. Apply word embeddings.
140. Compare multiple pipelines for NLP tasks.
141. Evaluate multi-class NLP models.
142. Generate predictions for new text data.
143. Interpret NLP predictions (feature importance if supported).
144. Export NLP pipelines for deployment.
145. Automate retraining for streaming or updated datasets.
146. Monitor deployed model performance over time.
147. Detect performance drift and trigger retraining.
148. Deploy pipeline as REST API.
149. Deploy pipeline with Streamlit or dashboard.
150. Track batch prediction performance.
151. Compare multiple AutoML frameworks on same dataset.
152. Evaluate robustness of top pipelines.
153. Apply multi-objective optimization (accuracy + speed).
154. Track model selection process programmatically.
155. Visualize pipeline evolution or search history.
156. Track per-operator contribution to final pipeline.
157. Automate feature selection + model selection jointly.
158. Apply custom metric for optimization.
159. Evaluate effect of different preprocessing steps.
160. Apply domain-specific constraints to pipelines.
161. Optimize under runtime constraints.
162. Optimize under memory constraints.
163. Use parallel processing if supported.
164. Track resource usage during AutoML.
165. Apply automated hyperparameter pruning.
166. Save all candidate models for comparison.
167. Track hyperparameter importance across runs.
168. Analyze top N pipelines for interpretability.
169. Generate automated experiment reports.
170. Integrate with MLflow or experiment tracking tool.
171. Combine AutoML with custom preprocessing scripts.
172. Automate retraining workflow with scheduler.
173. Compare pipelines across multiple datasets.
174. Apply AutoML for multi-output regression tasks.
175. Apply AutoML for multi-label classification tasks.
176. Evaluate top pipeline ensemble performance.
177. Interpret tree-based pipeline features with SHAP.
178. Analyze linear model coefficients in pipelines.
179. Visualize residuals and prediction errors.
180. Evaluate model calibration.
181. Apply advanced feature engineering (interaction, polynomial).
182. Automate pipeline export + deployment.
183. Monitor deployed model predictions.
184. Track model drift over time.
185. Retrain pipelines automatically when drift detected.
186. Compare performance across AutoML frameworks programmatically.
187. Track effect of hyperparameter changes on pipeline.
188. Optimize pipeline for sparse datasets.
189. Optimize pipeline for high-dimensional datasets.
190. Export pipelines as reusable Python modules.
191. Apply AutoML to very large datasets efficiently.
192. Track convergence of optimization process.
193. Evaluate ensemble diversity.
194. Compare single best model vs ensemble.
195. Apply AutoML to imbalanced classification tasks.
196. Apply automated feature scaling and transformation.
197. Document all experiments automatically.
198. Build reproducible end-to-end AutoML workflow.
199. Automate prediction reporting.
200. Build full end-to-end workflow: data prep → AutoML optimization → evaluation → interpretation → deployment → monitoring.

---

# **spaCy Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, data handling, and basic NLP pipelines*

1. Install spaCy using pip.
2. Check spaCy version.
3. Download an English model (e.g., `en_core_web_sm`).
4. Load the English model using `spacy.load()`.
5. Load a larger English model (e.g., `en_core_web_md`).
6. Create a blank spaCy model for English.
7. Process a simple sentence using `nlp()`.
8. Access tokens in a processed document.
9. Access token text using `.text`.
10. Access token lemma using `.lemma_`.
11. Access token part-of-speech using `.pos_`.
12. Access token detailed tag using `.tag_`.
13. Access token dependency using `.dep_`.
14. Access token shape using `.shape_`.
15. Access whether token is alpha using `.is_alpha`.
16. Access whether token is stopword using `.is_stop`.
17. Iterate through tokens in a sentence.
18. Print token text, lemma, POS, and dependency.
19. Access sentence spans using `.sents`.
20. Split text into sentences.
21. Access named entities using `.ents`.
22. Print entity text and label.
23. Access entity start and end positions.
24. Access entity label string using `.label_`.
25. Visualize entities using `displacy.render()`.
26. Visualize entities in Jupyter notebook.
27. Visualize dependencies using `displacy.render()`.
28. Change rendering style for dependencies.
29. Access noun chunks using `.noun_chunks`.
30. Iterate through noun chunks and print text.
31. Use `Doc` object properties: `.text`, `.vector`, `.sentiment`.
32. Access document vector for similarity tasks.
33. Compute similarity between two tokens.
34. Compute similarity between two documents.
35. Access sentence vectors (mean of token vectors).
36. Apply lowercasing to tokens.
37. Apply tokenization to custom text.
38. Access whitespace and punctuation tokens.
39. Remove stopwords from a document.
40. Count number of tokens, sentences, entities.
41. Filter tokens by POS.
42. Filter tokens by entity type.
43. Add custom attributes to tokens using `Token.set_extension()`.
44. Add custom attributes to spans using `Span.set_extension()`.
45. Add custom attributes to Doc using `Doc.set_extension()`.
46. Process a batch of texts using `nlp.pipe()`.
47. Disable unnecessary pipeline components for speed.
48. Measure processing speed with and without components.
49. Save model to disk using `nlp.to_disk()`.
50. Load model from disk using `spacy.load()`.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Pipeline components, text preprocessing, training, and intermediate NLP tasks*

51. Access pipeline components using `nlp.pipe_names`.
52. Get named component using `nlp.get_pipe()`.
53. Add custom pipeline component using `nlp.add_pipe()`.
54. Remove a pipeline component using `nlp.remove_pipe()`.
55. Move a pipeline component to a specific position.
56. Access tokenizer configuration.
57. Customize tokenization rules.
58. Add special cases to tokenizer.
59. Customize entity ruler patterns.
60. Add patterns to entity ruler dynamically.
61. Access existing rules in entity ruler.
62. Remove patterns from entity ruler.
63. Use Matcher to find token patterns.
64. Create a Matcher object.
65. Add patterns to Matcher.
66. Apply Matcher to a Doc object.
67. Extract matches from Matcher.
68. Filter matches by label or pattern ID.
69. Use PhraseMatcher for multi-word expressions.
70. Load patterns from JSON for PhraseMatcher.
71. Apply PhraseMatcher to a batch of texts.
72. Access span start and end indices from matches.
73. Apply regex patterns to token text.
74. Remove punctuation using token filtering.
75. Remove numbers using token filtering.
76. Convert text to lowercase with spaCy.
77. Lemmatize tokens programmatically.
78. Use `.lemma_` to normalize text.
79. Extract noun phrases for information retrieval.
80. Extract verb phrases for analysis.
81. Use dependency tree to find subjects and objects.
82. Visualize dependencies in Jupyter notebook.
83. Access token ancestors and children in dependency tree.
84. Apply sentiment scoring with spaCy pipeline (if model supports).
85. Calculate token-level statistics (frequency, POS counts).
86. Filter entities by label type (e.g., PERSON, ORG).
87. Count entity occurrences in text.
88. Apply entity linking (if model supports).
89. Train custom named entity recognizer (NER).
90. Convert training data to spaCy format.
91. Split annotated data into train/test.
92. Initialize blank NER model.
93. Add NER to pipeline.
94. Add labels to NER.
95. Update NER with training examples.
96. Apply minibatch training with `spacy.util.minibatch`.
97. Evaluate trained NER on test data.
98. Save custom NER model.
99. Load custom NER model.
100. Apply matcher to find domain-specific patterns.

…*(questions 101–130 continue with medium-level: training text classifiers, text categorization, multi-label classification, word vectors, similarity computations, custom pipeline components, pipeline optimization, preprocessing pipelines, multi-language support, batch processing, model evaluation and visualization)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Custom pipelines, deep learning integration, production-ready models, advanced NLP tasks*

131. Add custom sentiment analysis component.
132. Integrate spaCy with TensorFlow/Keras for text classification.
133. Integrate spaCy with PyTorch for sequence labeling.
134. Train text classifier with `TextCategorizer`.
135. Train multi-label text classifier.
136. Evaluate classifier with precision, recall, F1.
137. Apply early stopping during training.
138. Fine-tune pre-trained transformer models in spaCy.
139. Use `spacy-transformers` pipeline for NER.
140. Train custom transformer-based NER.
141. Freeze transformer layers during training.
142. Apply dropout in custom components.
143. Use GPU acceleration for training.
144. Measure GPU utilization during training.
145. Extract embeddings for words using `token.vector`.
146. Extract embeddings for sentences using `Doc.vector`.
147. Apply similarity search using embeddings.
148. Cluster documents based on embeddings.
149. Reduce dimensionality with PCA or UMAP.
150. Visualize embeddings in 2D or 3D.
151. Apply topic modeling on spaCy tokens.
152. Use spaCy vectors in downstream ML tasks.
153. Integrate spaCy embeddings with sklearn classifier.
154. Train custom dependency parser.
155. Evaluate parser on standard metrics (UAS, LAS).
156. Apply multi-language models.
157. Translate text and process with spaCy.
158. Apply custom stopword lists.
159. Apply token filters for domain-specific tasks.
160. Optimize pipeline for processing speed.
161. Parallelize text processing with `nlp.pipe()`.
162. Apply streaming large datasets.
163. Handle extremely long documents efficiently.
164. Build named entity linking (NEL) component.
165. Integrate spaCy with knowledge base (KB).
166. Evaluate NEL performance.
167. Apply rule-based and ML-based NER together.
168. Combine matcher results with NER predictions.
169. Save complete custom pipeline.
170. Load complete custom pipeline.
171. Apply pipeline for batch predictions.
172. Monitor pipeline performance over time.
173. Deploy spaCy pipeline as REST API.
174. Deploy spaCy pipeline with FastAPI or Flask.
175. Deploy spaCy pipeline in production for streaming data.
176. Automate retraining of models.
177. Track changes in language usage over time.
178. Apply domain-specific NER for medical, legal, or financial texts.
179. Evaluate domain-specific model performance.
180. Integrate spaCy with text summarization.
181. Extract keyphrases and keywords from documents.
182. Apply coreference resolution in custom pipeline.
183. Integrate with transformer-based models for QA.
184. Evaluate pipeline performance with standard NLP benchmarks.
185. Use spaCy with custom embeddings.
186. Apply custom token embeddings for domain adaptation.
187. Fine-tune word vectors in spaCy.
188. Evaluate semantic similarity between documents.
189. Apply clustering on named entities.
190. Visualize entity relationships.
191. Create dependency-based search queries.
192. Apply search in knowledge graphs.
193. Integrate spaCy with FLAML or other AutoML for NLP tasks.
194. Combine rule-based and ML-based classification.
195. Evaluate pipeline robustness across datasets.
196. Track model versioning for deployed pipelines.
197. Build end-to-end NLP workflow: preprocessing → pipeline → evaluation → deployment.
198. Automate retraining based on new incoming data.
199. Generate production-ready documentation for pipeline.
200. Build full spaCy NLP system: data prep → tokenization → training → evaluation → deployment → monitoring.

---

# **Gensim Mastery Workbook – 200 Practical Questions**

---

## **Phase 1: Basics (Questions 1–50)**

*Focus: Installation, setup, corpora handling, and basic preprocessing*

1. Install Gensim using pip.
2. Check Gensim version.
3. Import core modules (`corpora`, `models`, `similarities`).
4. Load a sample dataset (e.g., Reuters or any text corpus).
5. Convert dataset to a list of strings.
6. Tokenize text into words.
7. Lowercase all tokens.
8. Remove punctuation from tokens.
9. Remove stopwords using custom list.
10. Remove stopwords using NLTK or spaCy.
11. Remove numbers from tokens.
12. Apply simple stemming using NLTK.
13. Apply lemmatization using spaCy.
14. Create a dictionary from tokenized documents using `corpora.Dictionary()`.
15. Inspect dictionary token to ID mapping.
16. Filter out extreme tokens with `filter_extremes()`.
17. Convert documents into Bag-of-Words (BoW) format.
18. Inspect BoW of a sample document.
19. Count total unique tokens.
20. Count total number of documents.
21. Convert tokenized corpus into a TF-IDF model.
22. Inspect TF-IDF weights for a sample document.
23. Save dictionary to disk.
24. Load dictionary from disk.
25. Save corpus in BoW format.
26. Load corpus from disk.
27. Iterate through corpus to view token IDs and frequencies.
28. Handle empty documents in corpus.
29. Remove rare tokens programmatically.
30. Remove overly common tokens programmatically.
31. Apply simple bigram detection using `Phrases`.
32. Apply trigram detection using `Phrases`.
33. Inspect bigrams and trigrams detected.
34. Add detected bigrams to tokenized corpus.
35. Add detected trigrams to tokenized corpus.
36. Convert corpus to bag-of-ngrams representation.
37. Count number of bigrams/trigrams per document.
38. Create a dictionary including n-grams.
39. Save preprocessed corpus for future use.
40. Load preprocessed corpus for analysis.
41. Access token IDs from dictionary.
42. Access token counts across corpus.
43. Remove tokens below minimum frequency threshold.
44. Remove tokens above maximum frequency threshold.
45. Apply custom token filters.
46. Track preprocessing steps programmatically.
47. Split dataset into training and testing sets.
48. Prepare corpus for topic modeling.
49. Inspect first 10 documents after preprocessing.
50. Visualize token frequency distribution.

---

## **Phase 2: Medium (Questions 51–130)**

*Focus: Topic modeling, similarity queries, vectorization, and intermediate NLP tasks*

51. Create a TF-IDF model from corpus.
52. Transform BoW corpus into TF-IDF corpus.
53. Train a Latent Semantic Indexing (LSI) model.
54. Inspect topics generated by LSI.
55. Evaluate similarity of documents using LSI.
56. Train a Latent Dirichlet Allocation (LDA) model.
57. Inspect top words per topic in LDA.
58. Visualize topic distributions per document.
59. Assign dominant topic to each document.
60. Calculate coherence score for LDA model.
61. Tune number of topics for LDA.
62. Tune number of passes/iterations for LDA.
63. Apply online training for large corpora.
64. Save trained LDA model.
65. Load trained LDA model.
66. Apply Hierarchical Dirichlet Process (HDP) modeling.
67. Compare LDA vs HDP topic quality.
68. Filter corpus for rare topics.
69. Apply LSI similarity index.
70. Apply LDA similarity index.
71. Compute similarity between documents.
72. Retrieve top N most similar documents to a query.
73. Preprocess query for similarity search.
74. Convert query into BoW or TF-IDF format.
75. Use similarity matrix for fast retrieval.
76. Build and persist similarity index.
77. Load saved similarity index.
78. Apply streaming similarity search for large corpus.
79. Filter similarity results by threshold.
80. Evaluate retrieval performance (precision/recall).
81. Apply gensim’s `Matutils.cossim` for vector similarity.
82. Apply weighted TF-IDF similarity.
83. Create custom similarity functions.
84. Apply online TF-IDF updates for streaming documents.
85. Combine bigrams/trigrams in vector space.
86. Visualize topic distribution using pyLDAvis.
87. Install pyLDAvis for visualization.
88. Prepare LDA model and corpus for pyLDAvis.
89. Explore inter-topic distance map.
90. Interpret top words per topic in visualization.
91. Track word probability distribution per topic.
92. Compare multiple LDA models programmatically.
93. Automate hyperparameter tuning for LDA (number of topics, alpha, eta).
94. Evaluate coherence for multiple LDA models.
95. Apply LDA to unseen documents.
96. Transform unseen document into topic distribution.
97. Retrieve dominant topic for new document.
98. Update dictionary with new vocabulary.
99. Update LDA model with new documents incrementally.
100. Track model changes over incremental updates.

…*(questions 101–130 continue with medium-level: Word2Vec embeddings, Doc2Vec models, similarity search using embeddings, vector arithmetic, training embeddings on large corpus, incremental updates, model evaluation, batch processing, exporting/loading models, applying embeddings to downstream ML tasks, multi-word expressions integration, visualization of embeddings with t-SNE/UMAP, word analogy queries, nearest neighbors search, optimizing vector dimensionality, negative sampling, hierarchical softmax)*

---

## **Phase 3: Advanced (Questions 131–200)**

*Focus: Deep learning integration, large-scale pipelines, domain adaptation, deployment, and automation*

131. Train Word2Vec embeddings on corpus.
132. Train CBOW vs Skip-gram models.
133. Set vector dimensionality, window size, and min count.
134. Apply negative sampling for training.
135. Apply hierarchical softmax for training.
136. Save trained Word2Vec model.
137. Load Word2Vec model from disk.
138. Retrieve most similar words to a query word.
139. Compute similarity between two words.
140. Compute similarity between two documents.
141. Train Doc2Vec model on corpus.
142. Tag documents for Doc2Vec training.
143. Infer vector for unseen document using Doc2Vec.
144. Use embeddings in downstream ML tasks.
145. Cluster documents using embeddings.
146. Visualize embeddings with t-SNE.
147. Visualize embeddings with UMAP.
148. Train FastText embeddings.
149. Evaluate OOV (out-of-vocabulary) word handling in FastText.
150. Apply word analogy queries with embeddings.
151. Compute nearest neighbors for embedding vectors.
152. Train embeddings on large corpus with streaming.
153. Apply multiprocessing for faster embedding training.
154. Save embeddings in binary and text format.
155. Load embeddings from binary and text format.
156. Integrate embeddings with sklearn classifier/regressor.
157. Fine-tune embeddings for domain-specific tasks.
158. Apply embeddings to topic modeling as features.
159. Apply embeddings to document similarity search.
160. Use embeddings for semantic search.
161. Build retrieval system using embeddings.
162. Evaluate retrieval system performance.
163. Apply embeddings for clustering (KMeans, Agglomerative).
164. Track convergence of clustering with embeddings.
165. Optimize dimensionality for embeddings.
166. Train embeddings with custom tokenization.
167. Remove stopwords before embedding training.
168. Apply bigram/trigram detection before embedding training.
169. Use embeddings to compute sentence similarity.
170. Use embeddings in recommendation systems.
171. Track model performance over incremental updates.
172. Automate retraining with new corpus data.
173. Deploy embeddings for real-time search.
174. Deploy Doc2Vec/Word2Vec as REST API.
175. Apply embeddings in chatbots/NLP pipelines.
176. Apply embeddings in semantic clustering.
177. Use embeddings for document summarization tasks.
178. Integrate embeddings with deep learning models.
179. Combine embeddings with transformer models.
180. Apply embeddings for entity linking.
181. Fine-tune embeddings for multi-language corpora.
182. Handle extremely large vocabulary efficiently.
183. Optimize training for GPU acceleration.
184. Reduce memory usage for large embedding models.
185. Apply incremental updates to embeddings.
186. Track embedding drift over time.
187. Save entire embedding pipeline for deployment.
188. Load embedding pipeline in production.
189. Automate batch similarity search.
190. Apply embeddings to topic coherence scoring.
191. Compare embeddings across multiple corpora.
192. Evaluate embeddings for downstream NLP classification.
193. Integrate embeddings with gensim LDA/LSI pipelines.
194. Visualize semantic clusters using embeddings.
195. Generate nearest neighbor reports.
196. Track vector arithmetic operations (king - man + woman).
197. Build end-to-end embedding-based NLP workflow.
198. Automate retraining workflow for embeddings.
199. Monitor deployed embedding system performance.
200. Build full end-to-end Gensim NLP system: preprocessing → dictionary/corpus → embeddings → topic modeling → similarity → deployment → monitoring.

---