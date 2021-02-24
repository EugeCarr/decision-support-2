# this is the run file
from datetime import datetime

if __name__ == '__main__':
    now = datetime.now()
    start = now.strftime("%d/%m/%Y %H:%M:%S")
    print(' =============================================== \n',
          'DECISION SUPPORT FOR SUSTAINABLE PROCESS DESIGN \n',
          start, '\n',
          '=============================================== \n',
          )
