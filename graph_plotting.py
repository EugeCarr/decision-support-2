from matplotlib import pyplot as plt
import openpyxl


def search_var_header(x_header):
    assert type(x_header) == str, ('variable header must be a string, not:', type(x_header))
    found = False
    i = 1
    while not found:
        for j in range(1, sheet.max_column + 1):
            if sheet.cell(row=i, column=j).value == x_header:
                x_header_ret = sheet.cell(row=i, column=j)
                found = True
                break

        if i == 20 and not found:
            break

        i += 1

    if found:
        # print (x_header_ret.value)
        return x_header_ret
    if not found:
        raise AssertionError('column header', x_header, 'could not be found')
    return


def get_values(header, highest_row):
    assert isinstance(header, openpyxl.cell.cell.Cell), ('header must be type cell not:', type(header))
    assert type(highest_row) == int, ('highest row must be an integer,not:', type(highest_row))
    values = []

    for row in range(header.row + 1, highest_row + 1):
        cell_val = sheet.cell(row=row, column=header.column).value

        assert type(cell_val) != str
        values.append(cell_val)

    return values


wb = openpyxl.load_workbook('Results from simulations.xlsx')

sheet = wb['First Try']

x_var_name = 'Month'

y_var_name = 'liquidity'

y2_var_name = 'proportion_bio'

x_var_header = search_var_header(x_var_name)
x_values = get_values(x_var_header, sheet.max_row)

y_var_header = search_var_header(y_var_name)
y_values = get_values(y_var_header, sheet.max_row)

y2_var_header = search_var_header(y2_var_name)
y2_values = get_values(y2_var_header, sheet.max_row)

fig, ax1 = plt.subplots()
ax1.plot(x_values, y_values, label='Liquidity')
ax1.set_ylabel('Liquidity')
ax1.set_xlabel(x_var_name)

# ax1.set_ylabel('Liquidity')
# h1, l1 = ax1.get_legend_handles_labels()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# ax2.set_ylabel('Proportion bio')  # we already handled the x-label with ax1
# h2, l2 = ax2.get_legend_handles_labels()
ax2.plot(x_values, y2_values, 'm--', label='Proportion bio-based')
ax2.set_ylabel('Proportion bio-based')

# ax2.tick_params(axis='y')
#
# ax2.set_xlabel(x_var_name)

# plt.legend(handles=[p1, p2], loc="best")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax2.legend(fontsize=8, loc=(0.50, -0.15), frameon=False)
# ax1.legend(loc=0, frameon=False)
ax1.legend(loc=(0.00, -0.15), fontsize=8, frameon=False)

plt.tick_params(labelsize=8)
ax1.tick_params(labelsize=8)

# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
#
# lines = lines_1 + lines_2
# labels = labels_1 + labels_2

# ax1.legend(lines, labels, loc=0)

fig.tight_layout()
plt.show()


# from matplotlib import rc
# rc('mathtext', default='regular')


# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# lns1 = ax.plot(time, Swdown, '-', label = 'Swdown')
# lns2 = ax.plot(time, Rn, '-', label = 'Rn')
# ax2 = ax.twinx()
# lns3 = ax2.plot(time, temp, '-r', label = 'temp')
#
# # added these three lines
# lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)
#
# ax.grid()
# ax.set_xlabel("Time (h)")
# ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
# ax2.set_ylabel(r"Temperature ($^\circ$C)")
# ax2.set_ylim(0, 35)
# ax.set_ylim(-20,100)
# plt.show()

# fig = plt.figure(figsize=(17,11))
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()
# 5
# ax1.set_xlim(-90,170)
# ax2.set_xlim(91,320)
# ax1.plot(dateN,hetN,color='magenta',lw=3, label='Arctic Climatology Flux')
# ax2.plot(dateS,hetS,color='lime',lw=3, label='Antarctic Climatology Flux')
# ax1.plot(dateN,stdaboveN,color='magenta', alpha=0.2,lw=3, label='Arctic Standard Deviation')
# ax2.plot(dateS,stdaboveS,color='lime', alpha=0.2,lw=3, label='Antarctic Standard Deviation')
# ax1.plot(dateN,stdbelowN,color='magenta', alpha=0.2,lw=3)
# ax2.plot(dateS,stdbelowS,color='lime', alpha=0.2,lw=3)
# ax1.plot(dateN,het2010_11,color='b',alpha=1,lw=3, label='2011 Arctic Flux')
# ax1.plot(dateN2020,het2019_20,color='r',lw=3, label='2020 Arctic Flux')
# #ax1.axvline(x=74,color='k')
# ax2.fill_between(dateS,stdaboveS, stdbelowS, color='lime',alpha=0.2)
# ax1.fill_between(dateN,stdaboveN, stdbelowN, color='magenta',alpha=0.3)
# ax2.legend(fontsize=8,loc=(0.01,0.823),frameon=False)
# ax1.legend(loc=(0.01,0.57),fontsize=8,frameon=False)
# ax1.set_xlabel(r"Arctic Scale: Begins October 1$^{st}$, as Days from January 1$^{st}$",fontsize=28)
# ax2.set_xlabel(r"Antarctic Scale: Begins April 1$^{st}$",fontsize=28)
# plt.tick_params(labelsize=24)
# ax1.tick_params(labelsize=24)
# ax1.set_ylim(0,120)
# ax1.set_ylabel("Reaction Flux [10$^1$$^9$ molecules s$^-$$^1$ cm$^-$$^2$]", fontsize=28, labelpad=0)
# plt.title("Reaction Flux HCl+ClONO$_2$", fontsize=38, pad=0)
# plt.tight_layout()
# plt.savefig('reactionflux.png')