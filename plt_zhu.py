import matplotlib.pyplot as plt


# total_width, n = 0.8, 2
# width = total_width / n
# # x1 = x - width / 2
# # x2 = x1 + width
# x_width = range(0,len(y_data))
x1 = [5, 10, 15, 20, 25]
y1 = [1.2193, 1.2002, 1.2797, 1.0784, 1.2055]


plt.bar(x1, y1, width=0.5, lw=0.5,  facecolor='#f7a400', label="TripAdvisor")
# plt.bar(x2, y2_data, width=0.3, lw=0.5, fc="#3a9efd", label="Inconsistency data")
# print(y_data)
# for a, b in zip(x1, y_data):
# 	plt.text(a, b+0.01, '%.2f' % b, ha='center', va='bottom', fontsize=24)
#
# for a, b in zip(x2, y2_data):
# 	plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=24)

# plt.xticks([index + 0.15 for index in range(0, 26, 5)], x1, fontsize=28)
# plt.yticks([index for index in np.arange(0, 110, 10)], fontsize=28)
plt.legend(loc="upper right",fontsize=28)
plt.show()