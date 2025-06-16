

motors = ["DWA", "DWB", "DWC"]

bucket = "s3-assetcare-bucket"

#==============================================old management

# experiment_name_DigitalTransformation = "Data Transformation - Digital input"
experiment_name_AnalogTransformation = "Data Transformation - Analog input"

experiment_name_SpeedTransformation = "Data Transformation - Speed tag name"
experiment_name_SpeedFiltering = "Filtering - Speed tag name"
experiment_name_kmean = "Training Model - K-Mean Clustering"



tag_name_digital = [
    "DWA_INVERTER_RUNNING", 
    "DWB_INVERTER_RUNNING", 
    "DWC_INVERTER_RUNNING"
]

tag_name_analog = [
    "DWA_DSU_DC_VOLTAGE",
    "DWB_ACTUAL_POWER",
    # "DWB_SPEED_REF",
    "DWC_ACTUAL_POWER",
    "DWC_DSU_ACTUAL_POWER",
    "DWC_DSU_MAIN_AC_VOLTAGE",
    # "DWC_INVERTER_RUNNING",
    "AUXILIARY_HPU_AI_PRESSURE_VALUE",
    "AUXILIARY_RIGAIR_AI_AIRPRESSURE_VALUE",
    "DRAWWORKS_ADSBRAKE_BRKCTRL_BRK1_IO_SRVBRKPRESSPV_VALUE",
    "DRAWWORKS_ADSBRAKE_BRKCTRL_BRK2_IO_SRVBRKPRESSPV_VALUE",
    "TOP_DRIVE_CONVERTERS_IN_ED_DELTAP_IO_FEEDBACK_VALUE",
    # "TD_ACTUAL_MOTOR_TEMP",
    # "TD_ACTUAL_MOTOR_TORQUE",
    # "TD_ACTUAL_POWER",
    "TD_DSU_ACTUAL_POWER",
    # "TD_SPEED_REF",
    "TOP_DRIVE_CONVERTERS_IN_ED_ROP_IO_FEEDBACK_VALUE",
    "DWA_ACTUAL_MOTOR_TEMP",
    "DWB_ACTUAL_MOTOR_TEMP",
    "DRILL_SPEED",
    "DW_AMBIENT_TEMP",
    "DW_GEARBOX1_LO_PRESS",
    "DW_GEARBOX2_LO_PRESS",
    "DWA_RTD1_TEMP",
    "DWA_RTD3_TEMP",
    "DWB_RTD3_TEMP",
    "DWC_RTD2_TEMP",
    "DWC_RTD3_TEMP",
    "TD_RTD1_TEMP",
    "TD_RTD2_TEMP",
    "TD_RTD3_TEMP",
    # "DWA_ACTUAL_MOTOR_SPEED",
    "DWA_DSU_MAIN_AC_VOLTAGE",
    # "DWA_SPEED_REF",
    # "DWB_ACTUAL_MOTOR_SPEED",
    "DWB_ACTUAL_MOTOR_TORQUE",
    "DWB_DSU_ACTUAL_POWER",
    "DWB_DSU_DC_VOLTAGE",
    "DWB_DSU_MAIN_AC_VOLTAGE",
    # "DWB_INVERTER_RUNNING",
    # "DWC_ACTUAL_MOTOR_SPEED",
    "DWC_ACTUAL_MOTOR_TORQUE",
    "DWC_DSU_DC_VOLTAGE",
    # "DWC_SPEED_REF",
    "TD_DSU_DC_VOLTAGE",
    "TD_DSU_MAIN_AC_VOLTAGE",
    "DRAWWORKS_ADSBRAKE_BRKCTRL_BRK1_IO_PRKBRKPRESSPV_VALUE",
    "DRAWWORKS_ADSHOOKLOAD_MEM_ACTIVEHOOKLOADPV",
    "TOP_DRIVE_CONVERTERS_IN_ED_WOB_IO_FEEDBACK_VALUE",
    "DRAWWORKS_ADSBRAKE_BRKCTRL_BRK2_IO_PRKBRKPRESSPV_VALUE",
    "DRAWWORKS_HMI_IO_SHOWDRUMRPM",
    "DWA_ACTUAL_MOTOR_TORQUE",
    "DWA_ACTUAL_POWER",
    "DWA_DSU_ACTUAL_POWER",
    # "DWA_INVERTER_RUNNING",
    "TD_ACTUAL_MOTOR_SPEED",
    "TOP_DRIVE_OPENSTATION_TDS_RI_MAKEUPTORQUE_VALUE",
    "TOP_DRIVE_OPENSTATION_TDS_RI_THROTTLE_VALUE",
    "BIT_POSITION_VALUE",
    "DWC_ACTUAL_MOTOR_TEMP",
    "DW_GEARBOX1_LO_TEMP",
    "DW_GEARBOX2_LO_TEMP",
    "DWA_EXHAUST_TEMP",
    "DWA_RTD2_TEMP",
    "DWB_EXHAUST_TEMP",
    "DWB_RTD1_TEMP",
    "DWB_RTD2_TEMP",
    "DWC_EXHAUST_TEMP",
    "DWC_RTD1_TEMP",
]

tag_name_speed = [
    "DWA_ACTUAL_MOTOR_SPEED", 
    "DWB_ACTUAL_MOTOR_SPEED", 
    "DWC_ACTUAL_MOTOR_SPEED"
]




# *============ raw data ===============*
date_folders = [
    "2024-04",
    "2024-05",
    "2024-06",
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
    "2025-01",
    "2025-02",
    "2025-03"
]


columns = [
    "tag_name", 
    "value",
    "time_utc", 
    "units"
]





#==============================================new management
experiment_name_RegularInterval = "1. Regular Interval"
experiment_name_FilteringByDigitalInput = "2. Filtering by Digital input"
experiment_name_TrainingKMeanClustering = "3. Training k-Mean Clustering"

experiment_name_RegularInterval_DigitalInput = "1.1 Digital Input"
runID_RegularInterval_DigitalInput = "a5aa7bac83c54d9590e0fce6392af647"


experiment_name_RegularInterval_AnalogInput = "1.2 Analog Input"
runID_RegularInterval_AnalogInput = "1f1f4191b21349f68cb206076be84c66"

experiment_name_RegularInterval_SpeedTagNames = "1.3 Speed Tag Names"
runID_RegularInterval_SpeedTagNames = "3ab16e5e1fe24392875d6d7d7710680a"









