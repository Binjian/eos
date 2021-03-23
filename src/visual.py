import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch


def visual(t_real, v_real, v_wltc):
    plt.plot(t_real, v_real, color="green")
    plt.plot(t_real, v_wltc, color="blue")
    plt.legend(["Actual Speed", "Target Speed"])
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/h)")
    plt.title("Comparison between actual V and WLTC V")
    plt.pause(0.0001)

def compare_pic(t_real,t,e,e_real):
    fig, (ax1,ax2) = plt.subplots(2,constrained_layout = True)
    fig.suptitle('Energy Storage Report') 
    sub1 = ax1.plot(t_real, e_real, color="red")
    ax1.set_title("Manual Energy Loss")
    ax1.set(ylabel="Energy Loss")
    sub2 = ax2.plot(t, e, color="green")
    ax2.set_title("AI Energy Loss")
    ax2.set(xlabel="Time (s)",ylabel=" Energy Loss")
    fig.legend(
        [sub1, sub2], labels=["Manual Loss", "AI Loss"]
    )
    plt.savefig("../data/Comparison.png")

def gen_report(diff,diff1,e):
    report = SimpleDocTemplate("../data/report.pdf")
    img = "../data/Comparison.png"
    styles = getSampleStyleSheet()
    avg_diff = round((diff+diff1)/2,2)
    title = ParagraphStyle(
        'title',
        fontName="Helvetica-Bold",
        parent=styles['Heading2'],
        fontSize=16,
        alignment=1,
        spaceAfter=14
    )

    red = ParagraphStyle(
    'small',
        parent=styles['h1'],
        fontSize=14,
        leading=8,
        textColor="red",
        spaceAfter=14,
    )

    green = ParagraphStyle(
    'small',
        fontSize=14,
        leading=8,
        spaceAfter=14,
        textColor="green"
    )
    report_title = Paragraph("Energy Loss Report", title)
    v_lim = Paragraph("Velocity Offset Limit: 5 km/h",red)

    if diff < 5 and diff1 < 5:
        v_actual = Paragraph("Actual Velocity Offset: "+str(avg_diff)+" km/h",green)
        vali = Paragraph("Validation: Valid",green)
    else:
        v_actual = Paragraph("Actual Velocity Offset: "+str(avg_diff)+" km/h",red)
        vali = Paragraph("Validation: Invalid",red)

    e_saved = Paragraph("You have saved total energy of "+str(e),green)
    im = Image(img, 6*inch, 4.5*inch)
    report.build([report_title,v_lim,v_actual,vali,e_saved,im])
