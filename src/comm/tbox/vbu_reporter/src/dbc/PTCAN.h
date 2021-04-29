#ifndef _PTCAN_H
#define  _PTCAN_H

#include "dbc_node.h"

extern double VCU_BMSPackCurrent_53_CAN1;
extern double VCU_BMSPackVoltage_53_CAN1;
extern double VCU_BMSMainRelayStatus_53_CAN1;
extern double VCU_BMSSOC_53_CAN1;
extern double VCU_BMSDIsplaySOC_53_CAN1;
extern double VCU_BMSFaultLevel_53_CAN1;
extern double VCU_BMSThermalRunaway_53_CAN1;
extern double VCU_BMS_InletTemp_53_CAN1;
extern double VCU_BMS_OutletTemp_53_CAN1;
extern double VCU_BMS_CellTempHighAlarm_53_CAN1;
extern double VCU_BMS_CellTempLowAlarm_53_CAN1;
#define VCU_BMS2_CANID (0x53)

extern double ABS_Active_68_CAN1;
extern double EBD_Active_68_CAN1;
extern double ABS_EBDFailed_68_CAN1;
extern double ABS_Failed_68_CAN1;
extern double ABS_VehicleSpeed_68_CAN1;
extern double ABS_VehicleSpeedValid_68_CAN1;
extern double ABS_Rolling_Counter_68_CAN1;
extern double ABS_Checksum_68_CAN1;
#define ABS_VehicleSpeedAndStatus_CANID (0x68)

extern double ABS_WheelSpeed_FL_75_CAN1;
extern double ABS_WheelSpeed_FL_Status_75_CAN1;
extern double ABS_WheelSpeed_FR_75_CAN1;
extern double ABS_WheelSpeed_FR_Status_75_CAN1;
extern double ABS_Rolling_Counter_75_CAN1;
extern double ABS_Checksum_75_CAN1;
#define ABS_WheelSpeed_Front_Whl_CANID (0x75)

extern double ABS_WheelSpeedPlusCounter_FL_115_CAN1;
extern double ABS_WheelSpeedPlusCounter_FL_inv_115_CAN1;
extern double ABS_WheelSpeedPlusCounter_FR_115_CAN1;
extern double ABS_WheelSpeedPlusCounter_FR_inv_115_CAN1;
extern double ABS_WheelSpeedPlusCounter_RL_115_CAN1;
extern double ABS_WheelSpeedPlusCounter_RL_inv_115_CAN1;
extern double ABS_WheelSpeedPlusCounter_RR_115_CAN1;
extern double ABS_WheelSpeedPlusCounter_RR_inv_115_CAN1;
extern double ABS_Rolling_Counter_115_CAN1;
extern double ABS_Checksum_115_CAN1;
#define ABS_WheelSpeed_PulseCounter_CANID (0x115)

extern double ABS_WheelSpeed_RL_79_CAN1;
extern double ABS_WheelSpeed_RL_Status_79_CAN1;
extern double ABS_WheelSpeed_RR_79_CAN1;
extern double ABS_WheelSpeed_RR_Status_79_CAN1;
extern double ABS_Rolling_Counter_79_CAN1;
extern double ABS_Checksum_79_CAN1;
#define ABS_WheelSpeed_Rear_Whl_CANID (0x79)

extern double VCU_AxAcceleration_201_CAN1;
extern double VCU_AyAcceleration_201_CAN1;
extern double VCU_AzAcceleration_201_CAN1;
extern double VCU_Yawrate_201_CAN1;
#define VCU_201_CANID (0x201)

extern double VCU_MCU_TrqReq_105_CAN1;
extern double VCU_MCU_Mode_105_CAN1;
extern double VCU_MCU_ActiveRelease_105_CAN1;
extern double VCU_MCU_RelayStatus_105_CAN1;
extern double VCUrollingcounter_105_CAN1;
extern double VCUchecksum_105_CAN1;
#define VCU_MCU_CANID (0x105)

extern double VCU_AccPedalPosition_151_CAN1;
extern double VCU_AccPedalPositionValid_151_CAN1;
extern double VCU_virtualAccPedalPosition_151_CAN1;
extern double VCU_virtualAccPedalPositionValid_151_CAN1;
extern double VCU_BrakePressure_151_CAN1;
extern double VCU_BrakeSwitchStatus_151_CAN1;
extern double VCU_rollingcounter_151_CAN1;
extern double VCU_checksum_151_CAN1;
#define VCU_151_CANID (0x151)

extern double VCUEPFaultLevelWarning_210_CAN1;
extern double VCUPwrDisp_210_CAN1;
extern double VCUMotOverTemp_210_CAN1;
extern double VCUVehRdy_210_CAN1;
extern double VCUHVILError_210_CAN1;
extern double VCUChgLineSts_210_CAN1;
extern double VCU_Actual_210_CAN1;
extern double VCURegenSts_210_CAN1;
extern double VCUImdStopDriving_210_CAN1;
extern double VCURmvAcChger_210_CAN1;
extern double VCUVehPwrLimdLampReq_210_CAN1;
extern double VCUHV_On_210_CAN1;
extern double VCULowBatteryWarning_210_CAN1;
extern double VCUHVBattCutOff_210_CAN1;
extern double VCUMotorWarning_210_CAN1;
extern double VCUHVBattWarning_210_CAN1;
extern double VCUIsolutionWarning_210_CAN1;
extern double VCULimpHome_210_CAN1;
extern double VCUBrakWarning_210_CAN1;
extern double VCUPlease_USE_750V_DC_210_CAN1;
extern double VCUCruiseStatus_210_CAN1;
extern double VCUOverSpeed_210_CAN1;
extern double VCUPowerMode_210_CAN1;
extern double VCUVehicleMode_210_CAN1;
extern double VCUDrivingMode_210_CAN1;
extern double VCUVehicleLimit_210_CAN1;
extern double VCUBattPackTempWarning_210_CAN1;
extern double VCUEPS_Fault_210_CAN1;
extern double VCUEPB_Fault_210_CAN1;
extern double VCUVehicleDrivingMode_210_CAN1;
extern double VCUBMSChargeState_210_CAN1;
#define VCU_IC_CANID (0x210)

extern double VCU_TenkmTripEnergy_501_CAN1;
extern double VCU_HundredKmTripEnergy_501_CAN1;
extern double VCU_SOCLowWarning_501_CAN1;
extern double VCU_RemainingChargingTime_501_CAN1;
extern double VCU_RemainingDrivingRange_501_CAN1;
extern double VCU_LVBattVoltage_501_CAN1;
#define VCU_501_CANID (0x501)

extern double VCU_BMSPackCurrent_153_CAN1;
extern double VCU_BMSPackVoltage_153_CAN1;
extern double VCU_BMSMainRelayStatus_153_CAN1;
extern double VCU_BMSSOC_153_CAN1;
extern double VCU_BMSDIsplaySOC_153_CAN1;
extern double VCU_BMSFaultLevel_153_CAN1;
extern double VCU_BMSThermalRunaway_153_CAN1;
extern double VCU_BMS_InletTemp_153_CAN1;
extern double VCU_BMS_OutletTemp_153_CAN1;
extern double VCU_BMS_CellTempHighAlarm_153_CAN1;
extern double VCU_BMS_CellTempLowAlarm_153_CAN1;
#define VCU_BMS_CANID (0x153)

extern double VCU_TotalChargeEnergy_502_CAN1;
extern double VCU_TotalDischargeEnergy_502_CAN1;
#define VCU_502_CANID (0x502)

extern double VCU_DCDCVoltageReq_250_CAN1;
extern double VCU_DCDCEnable_250_CAN1;
#define VCU_DCDC_CANID (0x250)

extern double VCU_Vehicle_Speed_155_CAN1;
extern double VCU_EPB_Control_Request_155_CAN1;
extern double VCU_Braking_While_Driving_155_CAN1;
extern double VCU_EPB_Control_Mode_155_CAN1;
extern double VCURollingcounter_155_CAN1;
extern double VCUchecksum_155_CAN1;
#define VCU_EPB_CANID (0x155)

extern double VCU_ACSpeed_238_CAN1;
extern double VCU_ACPower_238_CAN1;
extern double VCU_AC_238_CAN1;
#define VCU_AC_CANID (0x238)

extern double VCU_TboxDVRUploadReq_550_CAN1;
#define VCU_Tbox_CANID (0x550)

extern double VCU_VehicleSpeedValidStatus_8F101D0_CAN1;
extern double VCU_VehicleStatus_8F101D0_CAN1;
extern double VCU_GearValidStatus_8F101D0_CAN1;
extern double VCU_VehicleSpeed_8F101D0_CAN1;
extern double VCU_Gears_8F101D0_CAN1;
extern double VCU_VehicleOperatingMode_8F101D0_CAN1;
extern double VCU_DynamicMode_8F101D0_CAN1;
extern double VCU_DGears_8F101D0_CAN1;
extern double VCU_PowerLimit_8F101D0_CAN1;
extern double VCU_ErrLevel3_8F101D0_CAN1;
extern double VCU_GearboxErr_8F101D0_CAN1;
extern double VCU_ErrLevel2_8F101D0_CAN1;
extern double VCU_ErrLevel1_8F101D0_CAN1;
extern double VCU_OilPressureStatus_8F101D0_CAN1;
extern double VCU_EnergyFlow_8F101D0_CAN1;
extern double VCU_AcceleratorPedalOpening_8F101D0_CAN1;
extern double VCU_Life_8F101D0_CAN1;
#define VCU_AVAS_CANID (0x8F101D0)

extern double MCU_MotorSpeed_205_CAN1;
extern double MCU_MotorTorque_205_CAN1;
extern double MCU_MotorMode_205_CAN1;
extern double MCU_MotorSpeedValid_205_CAN1;
extern double MCU_ActiveRelease_205_CAN1;
extern double MCU_MotorHillHold_205_CAN1;
extern double MCU_CalDCCurrent_205_CAN1;
extern double MCU_rollingcounter_205_CAN1;
extern double MCU_checksum_205_CAN1;
#define MCU_205_CANID (0x205)

extern double MCU_AllowMaxTorque_305_CAN1;
extern double MCU_DC_BUSVoltage_305_CAN1;
extern double MCU_DCRelayOutSideVoltage_305_CAN1;
extern double MCU_rollingcounter_305_CAN1;
extern double MCU_checksum_305_CAN1;
#define MCU_305_CANID (0x305)

extern double MCU_IGBTMaxTemperature_405_CAN1;
extern double MCU_MotorTemperature_405_CAN1;
extern double MCU_IPM_Temperature_405_CAN1;
extern double MCU_UVWMaxCurrent_405_CAN1;
extern double MCU_rollingcounter_405_CAN1;
extern double MCU_checksum_405_CAN1;
#define MCU_405_CANID (0x405)

extern double MCU_OverCurrent_505_CAN1;
extern double MCU_IGBTfault_505_CAN1;
extern double MCU_OutputFault_505_CAN1;
extern double MCU_OverSpeed_505_CAN1;
extern double MCU_HallFault_505_CAN1;
extern double MCU_ResolverFault_505_CAN1;
extern double MCU_IGBToverheat_505_CAN1;
extern double MCU_MotorOverheat_505_CAN1;
extern double MCU_BusOverVoltage_505_CAN1;
extern double MCU_BusUnderVoltage_505_CAN1;
extern double MCU_Hardware_OverVoltage_505_CAN1;
extern double MCU_Hardware_UnderVoltage_505_CAN1;
extern double MCU_Drive_Uparm_failure_505_CAN1;
extern double MCU_Drive_Downarm_failure_505_CAN1;
extern double MCU_CPLD_Status_505_CAN1;
extern double MCU_Motor_overTem_limit_505_CAN1;
extern double MCU_IGBT_NTC_overTem_limit_505_CAN1;
extern double MCU_IGBT_Junction_overTem_limit_505_CAN1;
extern double MCU_Radiator_overTem_limit_505_CAN1;
extern double MCU_CAN_Fault_505_CAN1;
extern double MCU_FaultLevel_505_CAN1;
extern double MCU_rollingcounter_505_CAN1;
extern double MCU_checksum_505_CAN1;
#define MCU_505_CANID (0x505)

extern double CDC_DriveMode_220_CAN1;
extern double CDC_RegenMode_220_CAN1;
extern double CDC_GoNotifierOnOff_220_CAN1;
extern double CDC_TowMode_220_CAN1;
extern double CDC_CreepOnOff_220_CAN1;
extern double CDC_rollingcounter_220_CAN1;
extern double CDC_checksum_220_CAN1;
#define CDC_VCU_CANID (0x220)

extern double BCM_LeftTurnLightStatus_8FE0121_CAN1;
extern double BCM_RightTurnLightStatus_8FE0121_CAN1;
extern double BCM_LowBeam_8FE0121_CAN1;
extern double BCM_HighBeam_8FE0121_CAN1;
extern double BCM_FrontFogLight_8FE0121_CAN1;
extern double BCM_RearFogLight_8FE0121_CAN1;
extern double BCM_Park_Light_Status_8FE0121_CAN1;
extern double BCM_Day_Light_Status_8FE0121_CAN1;
extern double BCM_Reversing_Switch_Status_8FE0121_CAN1;
extern double BCM_Horn_Status_8FE0121_CAN1;
extern double BCM_Prak_Switch_Status_8FE0121_CAN1;
extern double BCM_Brake_Light_Status_8FE0121_CAN1;
extern double BCM_DoorStatus_8FE0121_CAN1;
extern double BCM_IG1_Power_8FE0121_CAN1;
extern double BCM_Wash_Motor_Status_8FE0121_CAN1;
extern double BCM_Low_Wiper_Status_8FE0121_CAN1;
extern double BCM_HI_Wiper_Status_8FE0121_CAN1;
extern double BCM_Automatic_wiper_gear_8FE0121_CAN1;
extern double BCM_Automatic_headlight_gear_8FE0121_CAN1;
#define BCM_VCU_CANID (0x8FE0121)

extern double DCDCInputVoltage_303_CAN1;
extern double DCDCOutputVoltage_303_CAN1;
extern double DCDCInputCurrent_303_CAN1;
extern double DCDCOutputCurrent_303_CAN1;
extern double DCDCEnableFeedback_303_CAN1;
extern double DCDCWorkstatus_303_CAN1;
extern double DCDCrollingcounter_303_CAN1;
extern double DCDCchecksum_303_CAN1;
#define DCDC_VCU_303_CANID (0x303)

extern double DCDCTemperature_509_CAN1;
extern double DCDCDeriting_509_CAN1;
extern double DCDCOverHeat_509_CAN1;
extern double DCDCOverInputCurrent_509_CAN1;
extern double DCDCOverOutputCurrent_509_CAN1;
extern double DCDCOverInputVoltage_509_CAN1;
extern double DCDCUnderInputVoltage_509_CAN1;
extern double DCDCOverOutputVoltage_509_CAN1;
extern double DCDCUnderOutputVoltage_509_CAN1;
extern double DCDCrollingcounter_509_CAN1;
extern double DCDCchecksum_509_CAN1;
#define DCDC_VCU_509_CANID (0x509)

extern double ACCompActSpeed_248_CAN1;
extern double ACCompStatus_248_CAN1;
extern double ACOverCurrent_248_CAN1;
extern double ACOverVoltage_248_CAN1;
extern double ACUnderVoltage_248_CAN1;
extern double ACStandby_overvoltage_248_CAN1;
extern double ACStandby_undervoltage_248_CAN1;
extern double ACComFault_248_CAN1;
extern double ACOver_current_FreqReduction_248_CAN1;
extern double ACSpeedFault_248_CAN1;
extern double ACCompCurrent_248_CAN1;
extern double ACCompVoltage_248_CAN1;
#define AC_VCU_CANID (0x248)

extern double EPS_Steering_Angle_70_CAN1;
extern double EPS_steering_angle_spd_70_CAN1;
extern double EPS_Torque_Steering_70_CAN1;
extern double EPS_TORque_Motor_70_CAN1;
extern double EPS_ECU_MODE_CONT_70_CAN1;
extern double EPS_ECU_msg_counter_70_CAN1;
#define EPS_FEEDBACK_STA_CANID (0x70)

extern double EPS_TorqueSensorVoltError_507_CAN1;
extern double EPS_T2TorqueOutRange_507_CAN1;
extern double EPS_T1TorqueOutRange_507_CAN1;
extern double EPS_T1T2Mismatch_507_CAN1;
extern double EPS_ECUCaliFail_507_CAN1;
extern double EPS_ASteeringAngleOutRange_507_CAN1;
extern double EPS_RSteeringAngleOutRange_507_CAN1;
extern double EPS_AMismathcR_507_CAN1;
extern double EPS_MotorCurrentinvalid_507_CAN1;
extern double EPS_MotorDriverCircurtFail_507_CAN1;
extern double EPS_CurrentOverDevition_507_CAN1;
extern double EPS_RelayFail_507_CAN1;
extern double EPS_ECUCurrentSensorInvalid_507_CAN1;
extern double EPS_LVOverVoltage_507_CAN1;
extern double EPS_LVUnderVoltage_507_CAN1;
extern double EPS_CAN_BUS_OFF_507_CAN1;
extern double EPS_PTReadyLostFrame_507_CAN1;
extern double EPS_SpeedLose_507_CAN1;
extern double EPS_MotorPoSensorFault_507_CAN1;
extern double EPS_TemOverHigh_507_CAN1;
extern double EPS_TemOverLow_507_CAN1;
extern double EPS_InvalidSpeed_507_CAN1;
extern double EPS_SpeedOut_507_CAN1;
#define EPS_DM1_CANID (0x507)

extern double EPB_state_80_CAN1;
extern double EPB_Control_Mode_80_CAN1;
extern double EPB_Fault_Level_80_CAN1;
extern double Park_LED_Sta_80_CAN1;
extern double EPB_SolValve_ErrInd_80_CAN1;
extern double DragForce_80_CAN1;
extern double IGN_80_CAN1;
extern double EPB_Key_Sta_80_CAN1;
extern double EPB_Key_Err_80_CAN1;
extern double EPB_PSensor_Err_80_CAN1;
extern double Communication_80_CAN1;
extern double Rollingcounter_80_CAN1;
extern double CheckSum_80_CAN1;
#define EPB_State_CANID (0x80)

extern double T_BOXTime_Year_511_CAN1;
extern double T_BOXTime_Month_511_CAN1;
extern double T_BOXTime_Hour_511_CAN1;
extern double T_BOXTime_Day_511_CAN1;
extern double T_BOXTime_Min_511_CAN1;
extern double T_BOXTime_Sec_511_CAN1;
#define T_BOX_511_CANID (0x511)

extern double T_BOXGPS_Longitude_510_CAN1;
extern double T_BOXGPS_Latitude_510_CAN1;
#define T_BOX_510_CANID (0x510)

extern double IC_BrakeDisWarning_401_CAN1;
extern double IC_SeatbeltUnlock_401_CAN1;
extern double IC_VehOdometer_401_CAN1;
extern double IC_BrakeFluidWarning_401_CAN1;
#define IC_401_CANID (0x401)

extern double AVAS_WorkingStatus_801D0D3_CAN1;
extern double AVAS_HardwareVersion_801D0D3_CAN1;
extern double AVAS_SoftwareVersion_801D0D3_CAN1;
#define AVAS_CANID (0x801D0D3)

extern DbcParserMsgTblType TBL_DP_DBCMSGLIST_PTCAN[];
extern uint16_t u16s_dp_MsgTblSize_PTCAN;
#endif