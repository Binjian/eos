import io
import webbrowser as wb

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import style
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def visual(t_real, v_real, v_wltc):
    plt.plot(t_real, v_real, color="green")
    plt.plot(t_real, v_wltc, color="blue")
    plt.legend(["Actual Speed", "Target Speed"])
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/h)")
    plt.title("Comparison between actual V and WLTC V")
    # plt.pause(0.0001)


def compare_pic(t_real, t, e, e_real, thro_dev, thro_real_dev):
    fig, (ax1, ax2, ax3) = plt.subplots(3, constrained_layout=True)
    fig.suptitle("Energy Consumption Report")
    # Manual Energy plot
    sub1 = ax1.plot(t_real, e_real, color="red")
    ax1.set_title("Manual Energy Consumption")
    ax1.set(ylabel="Energy (wh)")
    ax1.annotate("{:.2f}".format(e_real[-1]), (t_real[-1], e_real[-1]))
    # AI Energy plot
    sub2 = ax2.plot(t, e, color="green")
    ax2.set_title("AI Energy Consumption")
    ax2.set(xlabel="Time (s)", ylabel="Energy (wh)")
    ax2.annotate("{:.2f}".format(e[-1]), (t[-1], e[-1]))
    # Thro rate comparison
    sub3 = ax3.plot(t_real, thro_real_dev)
    sub4 = ax3.plot(t, thro_dev)
    ax3.set_title("Throttle Rate Comparison")
    ax3.set(xlabel="Time (s)", ylabel="Rate (10%/s)")
    ax3.legend([sub3, sub4], labels=["Manual Throttle Rate", "AI Throttle Rate"])
    plt.savefig("../data/Comparison.png")


def gen_report(diff, diff1, e):
    report = SimpleDocTemplate("../data/report.pdf")
    img = "../data/Comparison.png"
    styles = getSampleStyleSheet()
    avg_diff = round((diff + diff1) / 2, 2)
    # title format
    title = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        parent=styles["Heading2"],
        fontSize=16,
        alignment=1,
        spaceAfter=14,
    )
    # warning format in red
    red = ParagraphStyle(
        "small",
        parent=styles["h1"],
        fontSize=14,
        leading=8,
        textColor="red",
        spaceAfter=14,
    )
    # valid format in green
    green = ParagraphStyle(
        "small", fontSize=14, leading=8, spaceAfter=14, textColor="green"
    )
    report_title = Paragraph("Energy Consumption Report", title)
    v_lim = Paragraph("Velocity Offset Limit: 5 km/h", red)

    # check if energy is saved
    e_saved_km = round(e * 1000, 2)
    if e >= 0:
        e_saved = Paragraph(
            "Energy consumption reduction: " + str(e_saved_km) + "wh/km", green
        )
    else:
        e_saved = Paragraph("Energy is not saved: " + str(e_saved_km) + "wh/km", red)

    # check if velocity offset valid
    if diff < 5 and diff1 < 5:
        v_actual = Paragraph(
            "Actual Velocity Offset: " + str(round(max(diff, diff1), 2)) + " km/h",
            green,
        )
        vali = Paragraph("Validation: Valid", green)
    else:
        v_actual = Paragraph(
            "Actual Velocity Offset: " + str(round(max(diff, diff1), 2)) + " km/h", red
        )
        vali = Paragraph("Validation: Invalid", red)
        e_saved = Paragraph("Engery Saved!", red)

    # insert plot
    im = Image(img, 6 * inch, 4.5 * inch)
    report.build([report_title, v_lim, v_actual, vali, e_saved, im])
    # open PDF
    wb.open("../data/report.pdf")
