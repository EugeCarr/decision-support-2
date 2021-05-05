# this is the run file
from datetime import datetime
from simulation_0_0 import simulate


def run(param):
    start = datetime.now()
    now = start.strftime("%d/%m/%Y %H:%M:%S")
    print(' =============================================== \n',
          'DECISION SUPPORT FOR SUSTAINABLE PROCESS DESIGN \n',
          now, '\n',
          '===============================================',
          )

    months = 240 + 48

    simulate(months, table=False, plot=False, Excel_p=True,
             # manufacturer settings
             capacity_root_coeff=param, speed_of_build=0.2, time_to_build=15.0,
             # regulator settings
             notice_period=24, fraction=0.1, start_levy=1.0, ratio_jump=0.5, wait_time=48, compliance_threshold=0.5,
             decade_jump=0.5)

    end = datetime.now()
    now = end.strftime("%d/%m/%Y %H:%M:%S")
    print('\n ==================== \n',
          'SIMULATION COMPLETE \n',
          now, '\n',
          '==================== \n',
          )
    return


if __name__ == '__main__':
    for x in [1.0, 2.0, 3.0, 4.0]:
        run(x)
