import numpy as np
from xml.dom.minidom import Document


def createmap(n):
    ## define start point
    start = np.array([], dtype=float)

    ## define initial road params
    x0 = -56.53979238754325
    y0 = -34.39446366782007
    hdg = 0.0
    length = 30000

    d = {}
    createVar = locals()
    listTemp = range(1, n + 1)

    ## Make XODR file
    # create xml file
    doc = Document()

    # create root
    opendrive = doc.createElement("OpenDRIVE")
    doc.appendChild(opendrive)

    # header
    header = doc.createElement("header")
    header.setAttribute("revMajor", "1")
    header.setAttribute("revMinor", "6")
    header.setAttribute("name", " ")
    header.setAttribute("version", "1.00")
    header.setAttribute("date", "Thu Jan 28 14:30 2021")
    header.setAttribute("north", "0.0000000000000000e+00")
    header.setAttribute("south", "0.0000000000000000e+00")
    header.setAttribute("east", "0.0000000000000000e+00")
    header.setAttribute("west", "0.0000000000000000e+00")
    opendrive.appendChild(header)

    for i in listTemp:
        # road
        createVar["road" + str(i)] = doc.createElement("road")
        createVar["road" + str(i)].setAttribute("name", " ")
        createVar["road" + str(i)].setAttribute("length", str(length))
        createVar["road" + str(i)].setAttribute("id", str(i))
        createVar["road" + str(i)].setAttribute("junction", "-1")
        createVar["road" + str(i)].setAttribute("rule", "RHT")
        opendrive.appendChild(createVar["road" + str(i)])

        # link
        createVar["link" + str(i)] = doc.createElement("link")
        createVar["link_content" + str(i)] = doc.createTextNode("")
        createVar["link" + str(i)].appendChild(createVar["link_content" + str(i)])
        createVar["road" + str(i)].appendChild(createVar["link" + str(i)])

        # planview
        createVar["planview" + str(i)] = doc.createElement("planView")
        createVar["road" + str(i)].appendChild(createVar["planview" + str(i)])

        # geometry
        createVar["geom" + str(i)] = doc.createElement("geometry")
        createVar["geom" + str(i)].setAttribute("s", "0.0000000000000000e+00")
        createVar["geom" + str(i)].setAttribute("x", str(x0))
        createVar["geom" + str(i)].setAttribute("y", str(y0 + 30 * (i - 1)))
        createVar["geom" + str(i)].setAttribute("hdg", str(hdg))
        createVar["geom" + str(i)].setAttribute("length", str(length))
        createVar["line" + str(i)] = doc.createElement("line")
        createVar["geom" + str(i)].appendChild(createVar["line" + str(i)])
        createVar["planview" + str(i)].appendChild(createVar["geom" + str(i)])

        # elevation
        createVar["elevation" + str(i)] = doc.createElement("elevationProfile")
        createVar["elevation_content" + str(i)] = doc.createElement("elevation")
        createVar["elevation_content" + str(i)].setAttribute(
            "s", "0.0000000000000000e+00"
        )
        createVar["elevation_content" + str(i)].setAttribute(
            "a", "0.0000000000000000e+00"
        )
        createVar["elevation_content" + str(i)].setAttribute(
            "b", "0.0000000000000000e+00"
        )
        createVar["elevation_content" + str(i)].setAttribute(
            "c", "0.0000000000000000e+00"
        )
        createVar["elevation_content" + str(i)].setAttribute(
            "d", "0.0000000000000000e+00"
        )
        createVar["elevation" + str(i)].appendChild(
            createVar["elevation_content" + str(i)]
        )
        createVar["road" + str(i)].appendChild(createVar["elevation" + str(i)])

        # lateral profile
        createVar["lateral" + str(i)] = doc.createElement("lateralProfile")
        createVar["lat_content" + str(i)] = doc.createTextNode(" ")
        createVar["lateral" + str(i)].appendChild(createVar["lat_content" + str(i)])
        createVar["road" + str(i)].appendChild(createVar["lateral" + str(i)])

        # lanes
        createVar["lanes" + str(i)] = doc.createElement("lanes")
        createVar["lane_Sec" + str(i)] = doc.createElement("laneSection")
        createVar["lane_Sec" + str(i)].setAttribute("s", "0.0000000000000000e+00")

        # left lane 1
        createVar["left" + str(i)] = doc.createElement("left")
        createVar["lane" + str(i)] = doc.createElement("lane")
        createVar["lane" + str(i)].setAttribute("id", "3")
        createVar["lane" + str(i)].setAttribute("type", "border")
        createVar["lane" + str(i)].setAttribute("level", "false")
        createVar["link" + str(i + 1)] = doc.createElement("link")
        createVar["link_content" + str(i + 1)] = doc.createTextNode(" ")
        createVar["link" + str(i + 1)].appendChild(
            createVar["link_content" + str(i + 1)]
        )
        createVar["lane" + str(i)].appendChild(createVar["link" + str(i + 1)])
        createVar["width" + str(i)] = doc.createElement("width")
        createVar["width" + str(i)].setAttribute("sOffset", "0.0000000000000000e+00")
        createVar["width" + str(i)].setAttribute("a", "6.0000000000000000e+00")
        createVar["width" + str(i)].setAttribute("b", "0.0000000000000000e+00")
        createVar["width" + str(i)].setAttribute("c", "0.0000000000000000e+00")
        createVar["width" + str(i)].setAttribute("d", "0.0000000000000000e+00")
        createVar["lane" + str(i)].appendChild(createVar["width" + str(i)])
        createVar["left" + str(i)].appendChild(createVar["lane" + str(i)])

        # left lane 2
        createVar["lane" + str(i + 1)] = doc.createElement("lane")
        createVar["lane" + str(i + 1)].setAttribute("id", "2")
        createVar["lane" + str(i + 1)].setAttribute("type", "border")
        createVar["lane" + str(i + 1)].setAttribute("level", "false")
        createVar["link" + str(i + 2)] = doc.createElement("link")
        createVar["link_content" + str(i + 2)] = doc.createTextNode(" ")
        createVar["link" + str(i + 2)].appendChild(
            createVar["link_content" + str(i + 2)]
        )
        createVar["lane" + str(i + 1)].appendChild(createVar["link" + str(i + 2)])
        createVar["width" + str(i + 1)] = doc.createElement("width")
        createVar["width" + str(i + 1)].setAttribute(
            "sOffset", "0.0000000000000000e+00"
        )
        createVar["width" + str(i + 1)].setAttribute("a", "1.6799999999999999e+00")
        createVar["width" + str(i + 1)].setAttribute("b", "0.0000000000000000e+00")
        createVar["width" + str(i + 1)].setAttribute("c", "0.0000000000000000e+00")
        createVar["width" + str(i + 1)].setAttribute("d", "0.0000000000000000e+00")
        createVar["lane" + str(i + 1)].appendChild(createVar["width" + str(i + 1)])
        createVar["left" + str(i)].appendChild(createVar["lane" + str(i + 1)])

        # left lane 3
        createVar["lane" + str(i + 2)] = doc.createElement("lane")
        createVar["lane" + str(i + 2)].setAttribute("id", "1")
        createVar["lane" + str(i + 2)].setAttribute("type", "driving")
        createVar["lane" + str(i + 2)].setAttribute("level", "false")
        createVar["link" + str(i + 3)] = doc.createElement("link")
        createVar["link_content" + str(i + 3)] = doc.createTextNode(" ")
        createVar["link" + str(i + 3)].appendChild(
            createVar["link_content" + str(i + 3)]
        )
        createVar["lane" + str(i + 2)].appendChild(createVar["link" + str(i + 3)])
        createVar["width" + str(i + 2)] = doc.createElement("width")
        createVar["width" + str(i + 2)].setAttribute(
            "sOffset", "0.0000000000000000e+00"
        )
        createVar["width" + str(i + 2)].setAttribute("a", "3.5699999999999998e+00")
        createVar["width" + str(i + 2)].setAttribute("b", "0.0000000000000000e+00")
        createVar["width" + str(i + 2)].setAttribute("c", "0.0000000000000000e+00")
        createVar["width" + str(i + 2)].setAttribute("d", "0.0000000000000000e+00")

        # roadmark
        createVar["roadmark" + str(i)] = doc.createElement("roadmark")
        createVar["roadmark" + str(i)].setAttribute("color", "standard")
        createVar["roadmark" + str(i)].setAttribute("height", "1.9999999552965164e-02")
        createVar["roadmark" + str(i)].setAttribute("laneChange", "none")
        createVar["roadmark" + str(i)].setAttribute(
            "sOffeset", "0.0000000000000000e+00"
        )
        createVar["roadmark" + str(i)].setAttribute("type", "solid")
        createVar["roadmark" + str(i)].setAttribute("weight", "standard")
        createVar["roadmark" + str(i)].setAttribute("width", "1.2000000000000000e-01")
        createVar["type" + str(i)] = doc.createElement("type")
        createVar["type" + str(i)].setAttribute("name", "solid")
        createVar["type" + str(i)].setAttribute("width", "1.2000000000000000e-01")
        createVar["line" + str(i)] = doc.createElement("line")
        createVar["line" + str(i)].setAttribute("length", "0.0000000000000000e+00")
        createVar["line" + str(i)].setAttribute("space", "0.0000000000000000e+00")
        createVar["line" + str(i)].setAttribute("tOffset", "0.0000000000000000e+00")
        createVar["line" + str(i)].setAttribute("sOffset", "0.0000000000000000e+00")
        createVar["line" + str(i)].setAttribute("rule", "no passing")
        createVar["line" + str(i)].setAttribute("width", "1.2000000000000000e-01")

        # integrate left lane
        createVar["type" + str(i)].appendChild(createVar["line" + str(i)])
        createVar["roadmark" + str(i)].appendChild(createVar["type" + str(i)])
        createVar["lane" + str(i + 2)].appendChild(createVar["width" + str(i + 2)])
        createVar["lane" + str(i + 2)].appendChild(createVar["roadmark" + str(i)])
        createVar["left" + str(i)].appendChild(createVar["lane" + str(i + 2)])
        createVar["lane_Sec" + str(i)].appendChild(createVar["left" + str(i)])
        createVar["lanes" + str(i)].appendChild(createVar["lane_Sec" + str(i)])
        createVar["road" + str(i)].appendChild(createVar["lanes" + str(i)])

        # center lanes
        createVar["center" + str(i)] = doc.createElement("center")
        createVar["lane" + str(i + 3)] = doc.createElement("lane")
        createVar["lane" + str(i + 3)].setAttribute("id", "0")
        createVar["lane" + str(i + 3)].setAttribute("type", "driving")
        createVar["lane" + str(i + 3)].setAttribute("level", "false")
        createVar["link" + str(i + 4)] = doc.createElement("link")
        createVar["link_content" + str(i + 4)] = doc.createTextNode(" ")
        createVar["link" + str(i + 4)].appendChild(
            createVar["link_content" + str(i + 4)]
        )
        createVar["lane" + str(i + 3)].appendChild(createVar["link" + str(i + 4)])
        createVar["roadmark" + str(i + 1)] = doc.createElement("roadmark")
        createVar["roadmark" + str(i + 1)].setAttribute(
            "sOffeset", "0.0000000000000000e+00"
        )
        createVar["roadmark" + str(i + 1)].setAttribute("weight", "standard")
        createVar["roadmark" + str(i + 1)].setAttribute("color", "standard")
        createVar["roadmark" + str(i + 1)].setAttribute(
            "width", "1.2000000000000000e-01"
        )
        createVar["roadmark" + str(i + 1)].setAttribute("laneChange", "both")
        createVar["roadmark" + str(i + 1)].setAttribute(
            "height", "1.9999999552965164e-02"
        )
        createVar["roadmark" + str(i + 1)].setAttribute("type", "broken")
        createVar["type" + str(i + 1)] = doc.createElement("type")
        createVar["type" + str(i + 1)].setAttribute("name", "broken")
        createVar["type" + str(i + 1)].setAttribute("width", "1.2000000000000000e-01")
        createVar["line" + str(i + 1)] = doc.createElement("line")
        createVar["line" + str(i + 1)].setAttribute("length", "4.0000000000000000e+00")
        createVar["line" + str(i + 1)].setAttribute("space", "8.0000000000000000e+00")
        createVar["line" + str(i + 1)].setAttribute("tOffset", "0.0000000000000000e+00")
        createVar["line" + str(i + 1)].setAttribute("sOffset", "0.0000000000000000e+00")
        createVar["line" + str(i + 1)].setAttribute("rule", "caution")
        createVar["line" + str(i + 1)].setAttribute("width", "1.2000000000000000e-01")

        # integrate center lane
        createVar["type" + str(i + 1)].appendChild(createVar["line" + str(i + 1)])
        createVar["roadmark" + str(i + 1)].appendChild(createVar["type" + str(i + 1)])
        createVar["lane" + str(i + 3)].appendChild(createVar["roadmark" + str(i + 1)])
        createVar["center" + str(i)].appendChild(createVar["lane" + str(i + 3)])
        createVar["lane_Sec" + str(i)].appendChild(createVar["center" + str(i)])

        # right lane 1
        createVar["right" + str(i)] = doc.createElement("right")
        createVar["lane" + str(i + 4)] = doc.createElement("lane")
        createVar["lane" + str(i + 4)].setAttribute("id", "-1")
        createVar["lane" + str(i + 4)].setAttribute("type", "driving")
        createVar["lane" + str(i + 4)].setAttribute("level", "false")
        createVar["link" + str(i + 5)] = doc.createElement("link")
        createVar["link_content" + str(i + 5)] = doc.createTextNode(" ")
        createVar["link" + str(i + 5)].appendChild(
            createVar["link_content" + str(i + 5)]
        )
        createVar["lane" + str(i + 4)].appendChild(createVar["link" + str(i + 5)])
        createVar["width" + str(i + 4)] = doc.createElement("width")
        createVar["width" + str(i + 4)].setAttribute(
            "sOffset", "0.0000000000000000e+00"
        )
        createVar["width" + str(i + 4)].setAttribute("a", "3.5699999999999998e+00")
        createVar["width" + str(i + 4)].setAttribute("b", "0.0000000000000000e+00")
        createVar["width" + str(i + 4)].setAttribute("c", "0.0000000000000000e+00")
        createVar["width" + str(i + 4)].setAttribute("d", "0.0000000000000000e+00")

        # roadmark
        createVar["roadmark" + str(i + 2)] = doc.createElement("roadmark")
        createVar["roadmark" + str(i + 2)].setAttribute(
            "sOffeset", "0.0000000000000000e+00"
        )
        createVar["roadmark" + str(i + 2)].setAttribute("weight", "standard")
        createVar["roadmark" + str(i + 2)].setAttribute("color", "standard")
        createVar["roadmark" + str(i + 2)].setAttribute(
            "width", "1.2000000000000000e-01"
        )
        createVar["roadmark" + str(i + 2)].setAttribute("laneChange", "none")
        createVar["roadmark" + str(i + 2)].setAttribute(
            "height", "1.9999999552965164e-02"
        )
        createVar["roadmark" + str(i + 2)].setAttribute("type", "solid")
        createVar["type" + str(i + 2)] = doc.createElement("type")
        createVar["type" + str(i + 2)].setAttribute("name", "solid")
        createVar["type" + str(i + 2)].setAttribute("width", "1.2000000000000000e-01")
        createVar["line" + str(i + 2)] = doc.createElement("line")
        createVar["line" + str(i + 2)].setAttribute("length", "0.0000000000000000e+00")
        createVar["line" + str(i + 2)].setAttribute("space", "0.0000000000000000e+00")
        createVar["line" + str(i + 2)].setAttribute("tOffset", "0.0000000000000000e+00")
        createVar["line" + str(i + 2)].setAttribute("sOffset", "0.0000000000000000e+00")
        createVar["line" + str(i + 2)].setAttribute("rule", "no passing")
        createVar["line" + str(i + 2)].setAttribute("width", "1.2000000000000000e-01")
        createVar["type" + str(i + 2)].appendChild(createVar["line" + str(i + 2)])
        createVar["roadmark" + str(i + 2)].appendChild(createVar["type" + str(i + 2)])
        createVar["lane" + str(i + 4)].appendChild(createVar["width" + str(i + 4)])
        createVar["lane" + str(i + 4)].appendChild(createVar["roadmark" + str(i + 2)])
        createVar["right" + str(i)].appendChild(createVar["lane" + str(i + 4)])

        # right lane 2
        createVar["lane" + str(i + 5)] = doc.createElement("lane")
        createVar["lane" + str(i + 5)].setAttribute("id", "-2")
        createVar["lane" + str(i + 5)].setAttribute("type", "border")
        createVar["lane" + str(i + 5)].setAttribute("level", "false")
        createVar["link" + str(i + 6)] = doc.createElement("link")
        createVar["link_content" + str(i + 6)] = doc.createTextNode(" ")
        createVar["link" + str(i + 6)].appendChild(
            createVar["link_content" + str(i + 6)]
        )
        createVar["lane" + str(i + 5)].appendChild(createVar["link" + str(i + 6)])
        createVar["width" + str(i + 5)] = doc.createElement("width")
        createVar["width" + str(i + 5)].setAttribute(
            "sOffset", "0.0000000000000000e+00"
        )
        createVar["width" + str(i + 5)].setAttribute("a", "1.6799999999999999e+00")
        createVar["width" + str(i + 5)].setAttribute("b", "0.0000000000000000e+00")
        createVar["width" + str(i + 5)].setAttribute("c", "0.0000000000000000e+00")
        createVar["width" + str(i + 5)].setAttribute("d", "0.0000000000000000e+00")
        createVar["lane" + str(i + 5)].appendChild(createVar["width" + str(i + 5)])
        createVar["right" + str(i)].appendChild(createVar["lane" + str(i + 5)])

        # right lane 3
        createVar["lane" + str(i + 6)] = doc.createElement("lane")
        createVar["lane" + str(i + 6)].setAttribute("id", "-3")
        createVar["lane" + str(i + 6)].setAttribute("type", "border")
        createVar["lane" + str(i + 6)].setAttribute("level", "false")
        createVar["link" + str(i + 7)] = doc.createElement("link")
        createVar["link_content" + str(i + 7)] = doc.createTextNode(" ")
        createVar["link" + str(i + 7)].appendChild(
            createVar["link_content" + str(i + 7)]
        )
        createVar["lane" + str(i + 6)].appendChild(createVar["link" + str(i + 7)])
        createVar["width" + str(i + 6)] = doc.createElement("width")
        createVar["width" + str(i + 6)].setAttribute(
            "sOffset", "0.0000000000000000e+00"
        )
        createVar["width" + str(i + 6)].setAttribute("a", "6.0000000000000000e+00")
        createVar["width" + str(i + 6)].setAttribute("b", "0.0000000000000000e+00")
        createVar["width" + str(i + 6)].setAttribute("c", "0.0000000000000000e+00")
        createVar["width" + str(i + 6)].setAttribute("d", "0.0000000000000000e+00")
        createVar["lane" + str(i + 6)].appendChild(createVar["width" + str(i + 6)])
        createVar["right" + str(i)].appendChild(createVar["lane" + str(i + 6)])

        # integrate right lane
        createVar["lane_Sec" + str(i)].appendChild(createVar["right" + str(i)])

        # surface, signals and objects
        createVar["objects" + str(i)] = doc.createElement("objects")
        createVar["surface" + str(i)] = doc.createElement("surface")
        createVar["signals" + str(i)] = doc.createElement("signals")
        createVar["content" + str(i)] = doc.createTextNode(" ")
        createVar["content" + str(i + 1)] = doc.createTextNode(" ")
        createVar["content" + str(i + 2)] = doc.createTextNode(" ")
        createVar["objects" + str(i)].appendChild(createVar["content" + str(i)])
        createVar["surface" + str(i)].appendChild(createVar["content" + str(i + 1)])
        createVar["signals" + str(i)].appendChild(createVar["content" + str(i + 2)])
        createVar["road" + str(i)].appendChild(createVar["objects" + str(i)])
        createVar["road" + str(i)].appendChild(createVar["signals" + str(i)])
        createVar["road" + str(i)].appendChild(createVar["surface" + str(i)])

        start = np.append(start, [x0, y0 + 30 * (i - 1), 3, hdg, 0])

    # write into files
    f = open("../data/straight.xodr", "w")
    doc.writexml(f, indent="\t", newl="\n", addindent="\t", encoding="utf-8")
    f.close()

    return start
