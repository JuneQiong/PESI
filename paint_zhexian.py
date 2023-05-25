# trip r2-f 2.0928 2.2608 2.2623 2.1721 2.2088 2.1357
# amov f2-f 1.9153 2.1801 2.2806 2.1394 2.2233 2.2257

# trip r1-f
# amov f1-f
import matplotlib.pyplot as plt

# y1 = [2.0928, 2.2608, 2.2623, 2.1721, 2.2088, 2.1357]
# x1 = range(1, 7)
# x2 = range(1, 7)
# y2 = [1.9153, 2.1801, 2.3421, 2.1394, 2.2233, 2.2257]
# plt.plot(x1, y1,label='TripAdvisor',linewidth=3,color='orange',marker='o', markersize=20)
# plt.plot(x2, y2,label='AmazonMov',linewidth=3,color='blue',marker='o', markersize=20)
# plt.xlabel('The Value of $\gamma$ ', fontsize=30)
# plt.ylabel('The value of R2-F', fontsize=30)
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.title('The impact of $\gamma$ on R2-F on TripAdvisor and AamzonMov.', fontsize=36, pad=30)
# plt.rcParams.update({'font.size': 25})
# plt.legend(loc='upper right')
# plt.show()


# trip r1-f 16.237, 16.8282, 17.0401, 16.7539, 16.6677, 16.7688
# amov f1-f  15.3416, 15.7308, 16.1074, 15.6590, 15.7408, 15.7900

y1 = [16.237, 16.8282, 17.0401, 16.7539, 16.6677, 16.7688]
x1 = range(1, 7)
x2 = range(1, 7)
y2 = [15.3416, 15.7308, 16.1074, 15.6590, 15.7408, 15.7900]
plt.plot(x1, y1,label='TripAdvisor',linewidth=3,color='orange',marker='o', markersize=20)
plt.plot(x2, y2,label='AmazonMov',linewidth=3,color='blue',marker='o', markersize=20)
plt.xlabel('The Value of $\gamma$ ', fontsize=30)
plt.ylabel('The value of R1-F', fontsize=30)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.title('The impact of $\gamma$ on R1-F on TripAdvisor and AamzonMov.', fontsize=36, pad=30)
plt.rcParams.update({'font.size': 25})
plt.legend(loc='upper right')
plt.show()

# bleu-4 of length
# x1 = [5, 10, 15, 20, 25]
# y1 = [1.2193, 1.2002, 1.2797, 1.0784, 1.2055]
# # x2 = range(1, 7)
# y2 = [1.0020, 1.0934, 1.1498, 1.1105, 1.1275]
# plt.plot(x1, y1,label='TripAdvisor',linewidth=3,color='orange',marker='o', markersize=10)
# plt.plot(x1, y2,label='AmazonMov',linewidth=3,color='blue',marker='o', markersize=10)
# plt.xlabel('The Value of length', fontsize=25)
# plt.ylabel('The value of BLEU-4', fontsize=25)
# plt.title('The impact of length on BLEU-4 on TripAdvisor and AmazonMov.', fontsize=30)
# plt.rcParams.update({'font.size': 20})
# plt.legend(loc='upper right')
# plt.show()

# rouge-1 of length
# x1 = [5, 10, 15, 20, 25]
# y1 = [14.9212, 16.5076, 17.1374, 17.0147, 16.5559]
# # x2 = range(1, 7)
# y2 = [12.0356, 13.5341, 15.0882, 14.3291, 12.5832]
# plt.plot(x1, y1,label='TripAdvisor',linewidth=3,color='orange',marker='o', markersize=10)
# plt.plot(x1, y2,label='AmazonMov',linewidth=3,color='blue',marker='o', markersize=10)
# plt.xlabel('The Value of length', fontsize=25)
# plt.ylabel('The value of ROUGE-1', fontsize=25)
# plt.title('The impact of length on ROUGE-1 on TripAdvisor and AmazonMov.', fontsize=30)
# plt.rcParams.update({'font.size': 20})
# plt.legend(loc='upper right')
# plt.show()

# bleu-4 of emsize
# x1 = range(0,7)
# x1_tick = ['16', '32', '64', '128', '256', '512', '1024']
# y1 = [0.8072,  0.9650, 0.9775,  1.1094,  1.1573, 1.2797, 1.1053]
# # x2 = range(1, 7)
# y2 = [0.4911, 0.7567, 0.9912, 0.9939, 1.0835, 1.1498, 1.1109]
# plt.plot(x1, y1,label='TripAdvisor',linewidth=3,color='orange',marker='o', markersize=10)
# plt.plot(x1, y2,label='AmazonMov',linewidth=3,color='blue',marker='o', markersize=10)
# plt.xlabel('The Value of Embedding Size', fontsize=25)
# plt.ylabel('The value of BLEU-4', fontsize=25)
# plt.title('The impact of embedding size on BLEU-4 on TripAdvisor and AmazonMov.', fontsize=30)
# plt.rcParams.update({'font.size': 20})
# plt.xticks(x1, x1_tick)
# plt.legend(loc='upper right')
# plt.show()

# usr of emsize
# x1 = range(0, 7)
# x1_tick = ['16', '32', '64', '128', '256', '512', '1024']
# y1 = [0.0003, 0.0070, 0.0237, 0.0688, 0.2010, 0.5322, 0.4302]# usr
# # y1 = [2.2134, 2.2691, 2.3023, 2.3963, 2.2010, 2.2303, 1.9855] # r2f
# # x2 = range(1, 7)
# y2 = [0.0040, 0.0363, 0.1099, 0.1735, 0.2984, 0.5737, 0.4229] # usr
# # y2 = [1.8217, 2.0964, 2.2591, 2.0721, 2.1885, 2.2477, 2.1419]# r2f
# plt.plot(x1, y1,label='TripAdvisor',linewidth=3,color='orange',marker='o', markersize=10)
# plt.plot(x1, y2,label='AmazonMov',linewidth=3,color='blue',marker='o', markersize=10)
# plt.xlabel('The Value of Embedding Size', fontsize=25)
# plt.ylabel('The value of USR', fontsize=25)
# plt.title('The impact of embedding size on USR on TripAdvisor and AmazonMov.', fontsize=30)
# plt.rcParams.update({'font.size': 20})
# plt.xticks(x1, x1_tick)
# plt.legend(loc='upper right')
# plt.show()