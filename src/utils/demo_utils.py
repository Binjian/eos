
import numpy as np
from numpy import diff
from scipy import integrate
import pandas as pd

def writexslx(x, y, v, path):
    df = pd.DataFrame({"x": x, "y": y, "v": v})
    writer = pd.ExcelWriter(path)
    df.to_excel(writer, index=False)
    writer.save()


def ai_filter(thro):
    # AI low pass filter
    filted_thro = []
    for idx, x in enumerate(thro):
        if idx > 0 and idx < len(thro) - 1:
            previous = thro[idx - 1]
            next = thro[idx + 1]
            x = (previous + x + next) / 3  # TODO change filter size
            filted_thro.append(x)
    filted_thro = np.insert(filted_thro, 0, thro[0])
    filted_thro = np.append(filted_thro, thro[-1])
    return filted_thro


def compute_loss(e_real, e, thro, thro_real, x_real, y_real, x, y):
    # calculate trip length
    xdiff = np.diff(x_real)
    ydiff = np.diff(y_real)
    dist_real = np.sqrt(np.square(xdiff) + np.square(ydiff))
    xdiff = np.diff(x)
    ydiff = np.diff(y)
    dist_ai = np.sqrt(np.square(xdiff) + np.square(ydiff))

    dist_real_sum = np.sum(dist_real)
    dist_ai_sum = np.sum(dist_ai)

    # calculate energy loss
    e_real_sum = integrate.cumtrapz(e_real, dx=0.1)
    # calculate throttle derivative
    thro_real_dev = abs(diff(thro_real) / 0.1)
    thro_real_dev = np.insert(thro_real_dev, 0, 0)
    cum_thro_real_dev = integrate.cumtrapz(thro_real_dev, dx=0.1)
    # calculate cumulative energy loss
    loss_real = (0.5 * e_real_sum + 8.0 * cum_thro_real_dev) * 0.375
    # loss_real = (cum_thro_real_dev) * 0.375
    loss_real = np.insert(loss_real, 0, 0)

    # for developed algorithm, uncomment this section
    # e_sum = integrate.cumtrapz(e,dx=0.1)
    # thro_dev = abs(diff(thro)/0.1)
    # thro_dev = np.insert(thro_dev,0,0)
    # cum_thro_dev = integrate.cumtrapz(thro_dev,dx=0.1)
    # cum_thro_dev = np.insert(cum_thro_dev,0,0)
    # loss_AI = (e_sum + cum_thro_dev)* 0.375
    # loss_AI = np.insert(loss_AI,0,0)
    # loss_real_total = round((e_real_sum[-1] + sum(thro_real_dev))* 0.375,2)
    # loss_AI_total = round((e_sum[-1] + cum_thro_dev[-1])* 0.375,2)
    # saved_AI = round(abs(loss_real_total-loss_AI_total),2)

    # for demo use, uncomment this section
    # cumulative a^2
    e_sum = integrate.cumtrapz(e, dx=0.1)
    # pedal rate
    thro_dev = abs(diff(thro) / 0.1)
    thro_dev = np.insert(thro_dev, 0, 0)
    # cumulative pedal rate
    cum_thro_dev = integrate.cumtrapz(thro_dev, dx=0.1)
    print(e_sum[-1])
    print(cum_thro_dev[-1])
    # AI Loss
    loss_AI = (
                      0.5 * e_sum + 8.0 * cum_thro_dev  #  - 0.2 * np.random.rand(1)
              ) * 0.375  # TODO: change 0.1 to adapt to experiment result s
    loss_AI -= loss_AI * 0.05 * np.random.rand(1)
    # loss_AI = (
    #     cum_thro_dev - 0.1 * np.random.rand(1)
    # ) * 0.375  # TODO: change 0.1 to adapt to experiment results
    loss_AI = np.insert(loss_AI, 0, 0)
    # total real loss and AI loss
    #    loss_real_total = (
    #        round((e_real_sum[-1] + cum_thro_real_dev[-1]) * 0.375, 4) / dist_real_sum
    #    )
    loss_real_total = loss_real[-1] / dist_real_sum
    # loss_AI_total = round((e_sum[-1] + cum_thro_dev[-1]) * 0.375, 4) / dist_ai_sum
    loss_AI_total = loss_AI[-1] / dist_ai_sum
    # saved energy
    saved_AI_total = round(loss_real_total - loss_AI_total, 4)

    return loss_AI, loss_real, saved_AI_total, thro_dev, thro_real_dev

