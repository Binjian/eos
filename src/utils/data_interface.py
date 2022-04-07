# system import
import socket
# import json
import os, time, math
from collections import deque
import signal
import queue

# third party import


# local import
from ..l045a_rdpg import logger, logc, logd, dictLogger
from ..l045a_rdpg import (
    hmi_lock,
    program_exit,
    episode_end,
    episode_done,
    episode_count,
    motionpowerQueue,
    vcu_calib_table_row_start,
    obs_len,
)
from ..comm.tbox.scripts.tbox_sim import *


def reset_capture_handler():
    """
    callback function for delay capture stop
    """
    get_truck_status.start = False
    logger.info(f"reset_capture_handler called", extra=dictLogger)
    raise Exception("reset capture to stop")


# declare signal handler callback
signal.Signal(signal.SIGALRM, reset_capture_handler)


def get_truck_status():
    """
    get truck status (observation) from vcu
    observe thread handler
    """
    global program_exit
    global motionpowerQueue, obs_len
    global episode_count, episode_done, episode_end
    global vcu_calib_table_row_start

    # logger.info(f'Start Initialization!', extra=dictLogger)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket.socket.settimeout(s, None)
    s.bind((get_truck_status.myHost, get_truck_status.myPort))
    # s.listen(5)
    # datetime.datetime.now().strftime("%Y%b%d-%H%M%S")
    th_exit = False
    logc.info(f"Initialization Done!", extra=dictLogger)
    # qobject_size = 0

    vel_hist_dQ = deque(maxlen=20)  # accumulate 1s of velocity values
    # vel_cycle_dQ = deque(maxlen=30)  # accumulate 1.5s (one cycle) of velocity values
    vel_cycle_dQ = deque(
        maxlen=obs_len
    )  # accumulate 1.5s (one cycle) of velocity values

    while not th_exit:  # th_exit is local; program_exit is global
        with hmi_lock:  # wait for tester to kick off or to exit
            if program_exit == True:  # if program_exit is True, exit thread
                logger.info(
                    "%s",
                    "Capture thread exit due to processing request!!!",
                    extra=dictLogger,
                )
                th_exit = True
                continue
        candata, addr = s.recvfrom(2048)
        # logger.info('Data received!!!', extra=dictLogger)
        pop_data = json.loads(candata)
        if len(pop_data) != 1:
            logc.critical("udp sending multiple shots!")
            break
        epi_delay_stop = False
        for key, value in pop_data.items():
            if key == "status":  # state machine chores
                # print(candata)
                if value == "begin":
                    get_truck_status.start = True
                    logc.info("%s", "Episode will start!!!", extra=dictLogger)
                    th_exit = False
                    # ts_epi_start = time.time()

                    vel_hist_dQ.clear()
                    epi_delay_stop = False
                    with hmi_lock:
                        episode_done = False
                        episode_end = False

                elif value == "end_valid":
                    # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                    # cannot sleep the thread since data capturing in the same thread, use signal alarm instead
                    get_truck_status.start = (
                        True  # do not stopping data capture immediately
                    )
                    get_truck_status.motpow_t = []
                    while not motionpowerQueue.empty():
                        motionpowerQueue.get()
                    logc.info("%s", "Episode done!!!", extra=dictLogger)
                    th_exit = False
                    vel_hist_dQ.clear()
                    epi_delay_stop = True
                    with hmi_lock:
                        episode_count += 1  # valid round increments
                        episode_done = True
                        episode_end = True
                elif value == "end_invalid":
                    get_truck_status.start = False
                    logc.info(f"Episode is interrupted!!!", extra=dictLogger)
                    get_truck_status.motpow_t = []
                    vel_hist_dQ.clear()
                    # motionpowerQueue.queue.clear()
                    # logc.info(
                    #     f"Episode motionpowerQueue has {motionpowerQueue.qsize()} states remaining",
                    #     extra=dictLogger,
                    # )
                    while not motionpowerQueue.empty():
                        motionpowerQueue.get()
                    # logc.info(
                    #     f"Episode motionpowerQueue gets cleared!", extra=dictLogger
                    # )
                    th_exit = False
                    epi_delay_stop = False
                    with hmi_lock:
                        episode_done = False
                        episode_end = True
                        episode_count += 1  # invalid round increments
                elif value == "exit":
                    get_truck_status.start = False
                    get_truck_status.motpow_t = []
                    vel_hist_dQ.clear()
                    while not motionpowerQueue.empty():
                        motionpowerQueue.get()
                    # logc.info("%s", "Program will exit!!!", extra=dictLogger)
                    th_exit = True
                    epi_delay_stop = False
                    # for program exit, need to set episode states
                    # final change to inform main thread
                    with hmi_lock:
                        episode_done = False
                        episode_end = True
                        program_exit = True
                        episode_count += 1
                    break
                    # time.sleep(0.1)
            elif key == "data":
                # logger.info('Data received before Capture starting!!!', extra=dictLogger)
                # logger.info(f'ts:{value["timestamp"]}vel:{value["velocity"]}ped:{value["pedal"]}', extra=dictLogger)
                # DONE add logic for episode valid and invalid
                if epi_delay_stop:
                    signal.alarm(3)
                try:
                    if get_truck_status.start:  # starts episode

                        velocity = float(value["velocity"])
                        pedal = float(value["pedal"])
                        brake = float(value["brake_pressure"])
                        current = float(value["A"])
                        voltage = float(value["V"])

                        motion_power = [
                            velocity,
                            pedal,
                            brake,
                            current,
                            voltage,
                        ]  # 3 +2 : im 5

                        get_truck_status.motpow_t.append(
                            motion_power
                        )  # obs_reward [speed, pedal, brake, current, voltage]
                        vel_hist_dQ.append(velocity)
                        vel_cycle_dQ.append(velocity)

                        if len(get_truck_status.motpow_t) >= obs_len:
                            if len(vel_cycle_dQ) != vel_cycle_dQ.maxlen:
                                logc.warning(  # the recent 1.5s average velocity
                                    f"cycle deque is inconsistent!",
                                    extra=dictLogger,
                                )

                            vel_aver = sum(vel_cycle_dQ) / vel_cycle_dQ.maxlen
                            vel_min = min(vel_cycle_dQ)
                            vel_max = max(vel_cycle_dQ)

                            # 0~20km/h; 7~25km/h; 10~30km/h; 15~35km/h; ...
                            # average concept
                            # 10; 16; 20; 25; 30; 35; 40; 45; 50; 55; 60;
                            #   13; 18; 22; 27; 32; 37; 42; 47; 52; 57; 62;
                            # here upper bound rule adopted
                            if vel_max < 20:
                                vcu_calib_table_row_start = 0
                            elif vel_max < 100:
                                vcu_calib_table_row_start = (
                                    math.floor((vel_max - 20) / 5) + 1
                                )
                            else:
                                logc.warning(
                                    f"cycle higher than 100km/h!",
                                    extra=dictLogger,
                                )
                                vcu_calib_table_row_start = 16

                            logd.info(
                                f"Cycle velocity: Aver{vel_aver},Min{vel_min},Max{vel_max},StartIndex{vcu_calib_table_row_start}!",
                                extra=dictLogger,
                            )
                            # logd.info(
                            #     f"Producer Queue has {motionpowerQueue.qsize()}!",
                            #     extra=dictLogger,
                            # )
                            motionpowerQueue.put(get_truck_status.motpow_t)
                            get_truck_status.motpow_t = []
                except Exception as X:
                    logc.info(
                        X,  # f"Valid episode, Reset data capturing to stop after 3 seconds!",
                        extra=dictLogger,
                    )
            else:
                logc.critical("udp sending unknown signal (neither status nor data)!")
                break

    logger.info(f"get_truck_status dies!!!", extra=dictLogger)

    s.close()


get_truck_status.motpow_t = []
get_truck_status.myHost = "127.0.0.1"
get_truck_status.myPort = 8002
get_truck_status.start = False
get_truck_status.qobject_len = 12  # sequence length 1.5*12s


set_tbox_sim_path(os.getcwd() + "/comm/tbox")
# this is the calibration table consumer for flashing
# @eye
def flash_vcu(tablequeue):
    global program_exit

    flash_count = 0
    th_exit = False

    logc.info(f"Initialization Done!", extra=dictLogger)
    while not th_exit:
        # time.sleep(0.1)
        with hmi_lock:
            if program_exit:
                th_exit = True
                continue
        try:
            # print("1 tablequeue size: {}".format(tablequeue.qsize()))
            table = tablequeue.get(block=False, timeout=1)  # default block = True
            # print("2 tablequeue size: {}".format(tablequeue.qsize()))
        except queue.Empty:
            pass
        else:

            # tf.print('calib table:', table, output_stream=output_path)
            logc.info(f"flash starts", extra=dictLogger)
            send_float_array("TQD_trqTrqSetNormal_MAP_v", table, sw_diff=True)
            # time.sleep(1.0)
            logc.info(f"flash done, count:{flash_count}", extra=dictLogger)
            flash_count += 1
            # watch(flash_count)

    # motionpowerQueue.join()
    logc.info(f"flash_vcu dies!!!", extra=dictLogger)
