from collections import namedtuple

PEDAL_SCALES = [
    0,
    0.02,
    0.04,
    0.08,
    0.12,
    0.16,
    0.20,
    0.24,
    0.28,
    0.32,
    0.38,
    0.44,
    0.50,
    0.62,
    0.74,
    0.86,
    1.0,
]
VELOCITY_SCALES_MULE = [
    0,
    7,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    100,
]  # in km/h, 21 elements
VELOCITY_SCALES_VB = [
    0,
    7,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
]  # in km/h, 14 elements

TRIANGLE_TEST_CASE_TARGET_VELOCITIES = [
    0,
    1.8,
    3.6,
    5.4,
    7.2,
    9,
    10.8,
    12.6,
    14.4,
    16.2,
    14.4,
    12.6,
    10.8,
    9,
    7.2,
    5.4,
    3.6,
    1.8,
    0,
    0,
    0,
]  # triangle test case in km/h

Truck = namedtuple(
    "Truck",
    [
        "TruckName",  # Name of the truck: VB7, M2, MP2, etc.
        "VIN",  # Vehicle Identification Number
        "Plate",  # License plate number
        "Maturity",  # "VB", "MULE", "MP"
        "CloudSignalFrequency",  # Hz
        "CloudGearFrequency",  # Hz
        "CloudUnitDuration",  # cloud unit duration in seconds
        "CloudUnitNumber",  # cloud number of units of cloud observation
        "PedalRange",  # percentage of pedal opening [0, 100]
        "PedalScale",  # scale number of pedal opening 17
        "PedalScaleList",  # scale list of pedal
        "VelocityRange",  # range of velocity [0, 100] in km/h
        "VelocityScale",  # scale number of velocity 14 or 21
        "VelocityScaleList",  # scale list of velocity
        "ObservationNumber",  # number of observation, 3: velocity, throttle, brake
        "KvaserObservationNumber",  # Kvaser number of one observation unit: 30 as count number
        "KvaserObservationFrequency",  # Kvaser observation frequency: 20 Hz
        "KvaserCountdownTime",  # Kvaser countdown time: 3 seconds
        "ActionBudget",  # maximal delta torque to be overlapped on the torque map 250 in Nm
        "ActionLowerBound",  # minimal percentage of delta torque to be overlapped on the torque map: 0.8
        "ActionUpperBound",  # maximal percentage of delta torque to be overlapped on the torque map: 1.0
        "ActionBias",  # bias of delta torque to be overlapped on the torque map: 0.0
    ],
)
truck_list = [
    Truck(
        TruckName="VB4",
        VIN="HMZABAAHXMF011054",
        Plate="77777777",
        Maturity="VB1",
        CloudSignalFrequency=50,
        CloudGearFrequency=2,
        CloudUnitDuration=1,
        CloudUnitNumber=4,
        PedalRange=[0.0, 1.0],
        PedalScale=17,
        PedalScaleList=PEDAL_SCALES,
        VelocityRange=[0.0, 120],
        VelocityScale=14,
        VelocityScaleList=VELOCITY_SCALES_VB,
        ObservationNumber=3,
        KvaserObservationNumber=30,
        KvaserObservationFrequency=20,
        KvaserCountdownTime=3,
        ActionBudget=250,  # 250 Nm
        ActionLowerBound=0.8,  # 80%
        ActionUpperBound=1.0,  # 100%
        ActionBias=0.0,  # No bias
    ),
    Truck(
        TruckName="VB1",
        VIN="HMZABAAH1MF011055",
        Plate="77777777",
        Maturity="VB1",
        CloudSignalFrequency=50,
        CloudGearFrequency=2,
        CloudUnitDuration=1,
        CloudUnitNumber=4,
        PedalRange=[0.0, 1.0],
        PedalScale=17,
        PedalScaleList=PEDAL_SCALES,
        VelocityRange=[0.0, 120],
        VelocityScale=14,
        VelocityScaleList=VELOCITY_SCALES_VB,
        ObservationNumber=3,
        KvaserObservationNumber=30,
        KvaserObservationFrequency=20,
        KvaserCountdownTime=3,
        ActionBudget=250,  # 250 Nm
        ActionLowerBound=0.8,  # 80%
        ActionUpperBound=1.0,  # 100%
        ActionBias=0.0,  # No bias
    ),
    Truck(
        TruckName="SU_BDC8937",
        VIN="HMZABAAH4MF014497",
        Plate="SU-BDC8937",
        Maturity="VB",
        CloudSignalFrequency=50,
        CloudGearFrequency=2,
        CloudUnitDuration=1,
        CloudUnitNumber=4,
        PedalRange=[0.0, 1.0],
        PedalScale=17,
        PedalScaleList=PEDAL_SCALES,
        VelocityRange=[0.0, 120],
        VelocityScale=14,
        VelocityScaleList=VELOCITY_SCALES_VB,
        ObservationNumber=3,
        KvaserObservationNumber=30,
        KvaserObservationFrequency=20,
        KvaserCountdownTime=3,
        ActionBudget=250,  # 250 Nm
        ActionLowerBound=0.8,  # 80%
        ActionUpperBound=1.0,  # 100%
        ActionBias=0.0,  # No bias
    ),
    Truck(
        TruckName="VB7",
        VIN="HMZABAAH7MF011058",
        Plate="77777777",
        Maturity="VB",
        CloudSignalFrequency=50,
        CloudGearFrequency=2,
        CloudUnitDuration=1,
        CloudUnitNumber=4,
        PedalRange=[0.0, 1.0],
        PedalScale=17,
        PedalScaleList=PEDAL_SCALES,
        VelocityRange=[0.0, 120],
        VelocityScale=14,
        VelocityScaleList=VELOCITY_SCALES_VB,
        ObservationNumber=3,
        KvaserObservationNumber=30,
        KvaserObservationFrequency=20,
        KvaserCountdownTime=3,
        ActionBudget=250,  # 250 Nm
        ActionLowerBound=0.8,  # 80%
        ActionUpperBound=1.0,  # 100%
        ActionBias=0.0,  # No bias
    ),
    Truck(
        TruckName="VB6",
        VIN="HMZABAAH5MF011057",
        Plate="66666666",
        Maturity="VB",
        CloudSignalFrequency=50,
        CloudGearFrequency=2,
        CloudUnitDuration=1,
        CloudUnitNumber=3,
        PedalRange=[0.0, 1.0],
        PedalScale=17,
        PedalScaleList=PEDAL_SCALES,
        VelocityRange=[0.0, 120],
        VelocityScale=14,
        VelocityScaleList=VELOCITY_SCALES_VB,
        ObservationNumber=3,
        KvaserObservationNumber=30,
        KvaserObservationFrequency=20,
        KvaserCountdownTime=3,
        ActionBudget=250,  # 250 Nm
        ActionLowerBound=0.8,  # 80%
        ActionUpperBound=1.0,  # 100%
        ActionBias=0.0,  # No bias
    ),
    Truck(
        TruckName="M2",
        VIN=None,  # "987654321654321M4"
        Plate="2222222",
        Maturity="MULE",
        CloudSignalFrequency=50,
        CloudGearFrequency=2,
        CloudUnitDuration=1,
        CloudUnitNumber=5,
        PedalRange=[0.0, 1.0],
        PedalScale=17,
        PedalScaleList=PEDAL_SCALES,
        VelocityRange=[0.0, 100],
        VelocityScale=21,
        VelocityScaleList=VELOCITY_SCALES_MULE,
        ObservationNumber=3,
        KvaserObservationNumber=30,
        KvaserObservationFrequency=20,
        KvaserCountdownTime=3,
        ActionBudget=250,  # 250 Nm
        ActionLowerBound=0.8,  # 80%
        ActionUpperBound=1.0,  # 100%
        ActionBias=0.0,  # No bias
    ),
    Truck(
        TruckName="HQB",
        VIN="NEWRIZON020220328",
        Plate="00000000",
        Maturity="VB",
        CloudSignalFrequency=50,
        CloudGearFrequency=2,
        CloudUnitDuration=1,
        CloudUnitNumber=5,
        PedalRange=[0.0, 1.0],
        PedalScale=17,
        PedalScaleList=PEDAL_SCALES,
        VelocityRange=[0.0, 120],
        VelocityScale=14,
        VelocityScaleList=VELOCITY_SCALES_VB,
        ObservationNumber=3,
        KvaserObservationNumber=30,
        KvaserObservationFrequency=20,
        KvaserCountdownTime=3,
        ActionBudget=250,  # 250 Nm
        ActionLowerBound=0.8,  # 80%
        ActionUpperBound=1.0,  # 100%
        ActionBias=0.0,  # No bias
    ),  # HQ Bench
]

trucks_by_name = dict(zip([truck.TruckName for truck in truck_list], truck_list))
trucks_by_vin = dict(zip([truck.VIN for truck in truck_list], truck_list))
