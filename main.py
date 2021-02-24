# this is the run file
from datetime import datetime
from simulation_0_0 import simulate

if __name__ == '__main__':
    start = datetime.now()
    now = start.strftime("%d/%m/%Y %H:%M:%S")
    print(' =============================================== \n',
          'DECISION SUPPORT FOR SUSTAINABLE PROCESS DESIGN \n',
          now, '\n',
          '=============================================== \n',
          )

    months = 120

    simulate(months, table=False, plot=False)

    end = datetime.now()
    elapsed = end - start
    elapsed = elapsed.total_seconds()
    print(' Iterations:', months,
          '\n Simulation time:', elapsed, 'seconds')
