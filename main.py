# this is the run file
from datetime import datetime
from simulation_0_0 import simulate

if __name__ == '__main__':
    start = datetime.now()
    now = start.strftime("%d/%m/%Y %H:%M:%S")
    print(' =============================================== \n',
          'DECISION SUPPORT FOR SUSTAINABLE PROCESS DESIGN \n',
          now, '\n',
          '===============================================',
          )

    months = 240 + 48

    simulate(months, table=False, plot=True, Excel_p=False)

    end = datetime.now()
    now = end.strftime("%d/%m/%Y %H:%M:%S")
    print('\n ==================== \n',
          'SIMULATION COMPLETE \n',
          now, '\n',
          '==================== \n',
          )
