import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
import webbrowser as wb


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
    fig.suptitle("Energy Storage Report")
    sub1 = ax1.plot(t_real, e_real, color="red")
    ax1.set_title("Manual Energy Loss")
    ax1.set(ylabel="Energy Loss (wh)")
    ax1.annotate("{:.2f}".format(e_real[-1]), (t_real[-1], e_real[-1]))
    sub2 = ax2.plot(t, e, color="green")
    ax2.set_title("AI Energy Loss")
    ax2.set(xlabel="Time (s)", ylabel="Energy Loss (wh)")
    ax2.annotate("{:.2f}".format(e[-1]), (t[-1], e[-1]))
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
    title = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        parent=styles["Heading2"],
        fontSize=16,
        alignment=1,
        spaceAfter=14,
    )

    red = ParagraphStyle(
        "small",
        parent=styles["h1"],
        fontSize=14,
        leading=8,
        textColor="red",
        spaceAfter=14,
    )

    green = ParagraphStyle(
        "small", fontSize=14, leading=8, spaceAfter=14, textColor="green"
    )
    report_title = Paragraph("Energy Loss Report", title)
    v_lim = Paragraph("Velocity Offset Limit: 5 km/h", red)

    if diff < 5 and diff1 < 5:
        v_actual = Paragraph(
            "Actual Velocity Offset: " + str(round(max(diff, diff1), 2)) + " km/h",
            green,
        )
        vali = Paragraph("Validation: Valid", green)
        e_saved = Paragraph("You have saved total energy of " + str(e) + " wh", green)
    else:
        v_actual = Paragraph(
            "Actual Velocity Offset: " + str(round(max(diff, diff1), 2)) + " km/h", red
        )
        vali = Paragraph("Validation: Invalid", red)
        e_saved = Paragraph("You have saved total energy of N\A wh", red)

    im = Image(img, 6 * inch, 4.5 * inch)
    report.build([report_title, v_lim, v_actual, vali, e_saved, im])
    wb.open("../data/report.pdf")
