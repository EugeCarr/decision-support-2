# this is the run file
from datetime import datetime
import simulation_0_0

if __name__ == '__main__':
    now = datetime.now()
    start = now.strftime("%d/%m/%Y %H:%M:%S")
    print(' =============================================== \n',
          'DECISION SUPPORT FOR SUSTAINABLE PROCESS DESIGN \n',
          start, '\n',
          '=============================================== \n',
          )

    simulation_0_0.simulate(120)
