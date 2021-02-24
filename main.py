# this is the run file
from datetime import datetime
import simulation_0_0

if __name__ == '__main__':
    start = datetime.now()
    now = start.strftime("%d/%m/%Y %H:%M:%S")
    print(' =============================================== \n',
          'DECISION SUPPORT FOR SUSTAINABLE PROCESS DESIGN \n',
          now, '\n',
          '=============================================== \n',
          )

    simulation_0_0.simulate(120, False)

    end = datetime.now()
    elapsed = end - start
    elapsed = elapsed.total_seconds()
    print('Simulation time:', elapsed, 'seconds')
