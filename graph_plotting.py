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
        print (x_header_ret.value)
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
p1 = ax1.plot(x_values, y_values)

ax1.set_xlabel(x_var_name)

ax1.set_ylabel('Liquidity')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Proportion bio')  # we already handled the x-label with ax1
# ax2.plot(x_values, y2_values, color=color, marker='', markersize=3)
p2 = ax2.plot(x_values, y2_values, 'm--')
ax2.tick_params(axis='y')

# plt.legend(handles=[p1,p2], loc="best")

fig.tight_layout()
plt.show()
