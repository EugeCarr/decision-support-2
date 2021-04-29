from matplotlib import pyplot as plt
import openpyxl


y = parameter.history
t = np.arange(0, len(y), 1)
fig, ax1 = plt.subplots()
ax1.plot(t, y)

ax1.set_xlabel('Month')
# ax1.set_ylabel('Price of bio feedstock')
# ax1.set_ylabel('Proportion bio-PET')
# ax1.set_ylabel('Levy rate')
ax1.set_ylabel('Emissions')

fig.tight_layout()
plt.show()
