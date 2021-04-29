#pragma once

#include <stdint.h>
#include <time.h>

#include <functional>
#include <mutex>
#include <thread>

#ifndef ON
#define ON 1
#endif

#ifndef OFF
#define OFF 0
#endif

#define DP_SIGNALTYPE_BOOL ((uint8_t)0)
#define DP_SIGNALTYPE_UINT8 ((uint8_t)1)
#define DP_SIGNALTYPE_UINT16 ((uint8_t)2)
#define DP_SIGNALTYPE_UINT32 ((uint8_t)3)
#define DP_SIGNALTYPE_SINT8 ((uint8_t)4)
#define DP_SIGNALTYPE_SINT16 ((uint8_t)5)
#define DP_SIGNALTYPE_SINT32 ((uint8_t)6)
#define DP_SIGNALTYPE_FLOAT ((uint8_t)7)
#define DP_SIGNALTYPE_IEEEFLOAT ((uint8_t)8)
#define DP_SIGNALTYPE_IEEEDOUBLE ((uint8_t)9)

#define DP_MSGDIR_RX ((uint8_t)0)
#define DP_MSGDIR_TX ((uint8_t)1)

#define DP_BYTEORDER_MOTOROLA ((uint8_t)0)
#define DP_BYTEORDER_INTEL ((uint8_t)1)

#define DP_SIGNTYPE_UNSIGNED ((uint8_t)0)
#define DP_SIGNTYPE_SIGNED ((uint8_t)1)

typedef unsigned long long AddrType;

#define DP_RXMAINFUNCTION_PEIROD ((uint8_t)5)
#define DP_TXMAINFUNCTION_PERIOD ((uint8_t)10)

#define DP_MSGKIND_NORMAL 0
#define DP_MSGKIND_GROUP 1

#define DP_PERIODICMACRO 0x01

#define DP_MSG_INVALID_INDEX (0xFFFF)

#define DBC_ERR_OK (int)(0)
#define DBC_ERR_ID_NOT_FOUND (int)(-1)
#define DBC_ERR_CRC_ERR (int)(-2)
#define DBC_ERR_MSG_CNT_ERR (int)(-3)
#define DBC_ERR_RX_DISABLE (int)(-4)

typedef struct {
  AddrType ad_Addr; /* the pointer to signal's value */
  uint8_t u8_Type;  /* DP_SIGNALTYPE */
  uint16_t u16_StartBit;
  uint8_t u8_DataLen;
  uint8_t u8_ByteOrder;
  float f32_Factor;    /* Factor */
  float f32_Offset;    /* Offset */
  float f32_Min;       /* Physical min value */
  float f32_Max;       /* Physical max value */
  uint8_t u8_SignType; /* Signed value or not*/
} DbcParserSignalTblType;

typedef struct {
  uint8_t u8_RxFlag;       /* new data received flag，only for rx message */
  bool txEnFlag;           /* transmit enable flag，only for tx message */
  uint32_t u32_AgeCounter; /* age counter increases in the main loop */
  uint32_t
      u32_Timeout; /* if u32_AgeCounter reaches u32_Timeout, loss comm happens*/
  uint16_t u16_MsgCntMax;   /* Msg counter max value which calulated from
                               u8_DataLen of MsgCnt signal*/
  uint16_t u16_RecoveryCnt; /* RecoveryCnt is set to 5 when lost comm happens,
                               loss comm recovery is detected only when
                               RecoveryCnt decreased to 0*/
  uint32_t u32_CrcErrCnt;
  uint32_t u32_LostCommCnt;
  uint32_t u32_MsgCntErrCnt;
  bool firstFrameFlag;
  std::mutex MsgMutex; /*Mutex to protect the pt_Buffer for each message*/
} DbcParserMsgStatusType;

typedef struct {
  uint32_t u32_CanId;  /* MsgCanId */
  uint8_t u8_Dir;      /* Tx or Rx */
  uint8_t u8_SendType; /* period or not */
  uint16_t u16_Period;
  const DbcParserSignalTblType* pt_SignalTbl; /* Signal table */
  uint16_t u16_TblSize;                       /* Signal table size */
  uint8_t* pt_Buffer;                         /* pointer to buffer */
  uint16_t u16_MsgLen;                        /* DLC */
  uint8_t u8_MsgParseType;                    /* 0=normal，1=group */
  uint16_t u16_MsgCntIndex;                   /* MsgCnt index in pt_SignalTbl */
  uint16_t u16_CrcIndex;                      /* Crc index in pt_SignalTbl */
  DbcParserMsgStatusType* pt_MsgStatus;       /* Pointer to message status */
} DbcParserMsgTblType;

typedef struct {
  const DbcParserMsgTblType* pt_MsgTbl; /* Pointer to message table */
  uint16_t u16_tblSize;                 /* Message table size */
} DbcParserDbcTblType;

typedef struct {
  uint32_t u32_channel;
  uint32_t u32_canid;
  uint16_t u16_dlc;
  uint8_t* pt_buff;
} DbcParserCanType;

struct DbcTableList {
  DbcParserDbcTblType* dbc_list_;
  int size_;
};

class Dbc_node {
 public:
  // Dbc_node(uint16_t period, const DbcParserDbcTblType* in_dbc_list,
  //          uint16_t in_dbc_num);
  Dbc_node(uint16_t period, const DbcTableList* in_dbc_lists,
           uint16_t in_dbc_lists_num);
  /* indicate that one message has been received and copy data into object */
  int rxIndication(uint32_t canId, const uint8_t* rxBuff, uint16_t dlc,
                   const DbcParserDbcTblType* pt_MsgTbl, uint16_t u16_tblSize);
  /* parse all the messages into signals which have been called rxIndication */
  void parseMsg(const DbcParserDbcTblType* tblList, uint16_t listSize);

  void transmitMessages(int channel, const DbcParserDbcTblType* tblList,
                        uint16_t listSize);
  /* set timeout value(in ms) for specific canid */
  int setTimeout(uint32_t canId, uint16_t timeout);
  /* set timeout by multiplying mult to u16_Period for all the messages */
  void setTimeoutByMult(uint16_t mult);
  /* enable tx & rx, start the thread to determine loss comm and send out the TX
   * messages */
  void start();
  /* disable tx & rx, stop the timer */
  void stop();
  /* set rx status if you want */
  void setRxStatus(bool);
  /* set tx status if you want */
  void setTxStatus(bool);
  /* will call this handler when loss comm occurs */
  void setLossCommHandler(std::function<void(uint32_t)>);
  /* set losscomm enable*/
  void setLossCommCheckEnable(bool status);
  /* will call this handler when transmitting one message  */
  void setTransmitHandler(
      std::function<int(uint16_t, uint32_t, uint8_t*, uint16_t)>);
  /* will call this handler when loss comm recovery is detected */
  void setLossCommRecoveryHandler(std::function<void(uint32_t)>);
  /* will use this handler to calulate the crc value for the message */
  void setCalcCrcHandler(
      std::function<uint32_t(const uint8_t*, uint16_t, uint8_t)>);
  /* lock this mutex when you are giving the signal value to tx messages */
  void lockConstructMutex();
  void unlockConstructMutex();
  int DisablePeriodicTransmitMsg(uint32_t canId);
  int DirectTransmitMsg(uint16_t channel, uint32_t canId, uint8_t* buff,
                        uint16_t dlc);

 private:
  /* the period(in ms) in which ecal_com_dp_MainFunction is called */
  uint16_t u16_MainPeriod;
  /* DBC list for this node */
  // const DbcParserDbcTblType* pt_DbcList;
  /* DBC number for this node */
  // uint16_t u16_DbcNum;
  /* rx enable/disable */
  bool b_RxStatus;
  /* tx enable/disable */
  bool b_TxStatus;
  /* timer counter used in ecal_com_dp_MainFunction */
  uint32_t u32_TimerCnt;
  /* losscomm enable flag*/
  bool losscomm_check_enable;
  /* used to create a timer to activate loopHandler */
  timer_t TimerId;
  /* used to receive message passing signal in loopHandler */
  int EventChid;
  /* thread to start loopHandler */
  std::thread mainThread;
  /* mutex for constructing the tx message */
  std::mutex constructMutex;
  /* mutex for call the direct tx message */
  std::mutex directTxMutex;

  /* function pointer */
  std::function<void(uint32_t)> lossCommHandler;
  std::function<int(uint16_t, uint32_t, uint8_t*, uint16_t)> transmitHandler;
  std::function<void(uint32_t)> lossCommRecoveryHandler;
  std::function<uint32_t(const uint8_t*, uint16_t, uint8_t)> calcCrcHandler;

  /* the thread to receive the signal from timer to run ecal_com_dp_MainFunction
   */
  void loopHandler();
  /* set the timeout value for timer "TimerId" */
  void setTimer(int timer_value);
  /* check counter diff */
  int check_counter_diff(uint16_t last, uint16_t received, uint16_t msgCntMax);
  /* look up the canId and in pt_MsgTbl and copy data into it */
  int ecal_com_dp_RxIndication(uint32_t u32t_canid, uint16_t u8t_datalen,
                               const uint8_t* ptt_buffer,
                               const DbcParserMsgTblType* pt_MsgTbl,
                               uint16_t u16_tblSize);
  /* check whether loss comm occurs and transmitting the tx messages in
   * pt_MsgTbl */
  void ecal_com_dp_MainFunction(int channel,
                                const DbcParserMsgTblType* pt_MsgTbl,
                                uint16_t u16_tblSize);
  /* construct the message whose pointer is pt_MsgTbl into stt_pdu */
  void ecal_com_dp_ContructSignalMsg(const DbcParserMsgTblType* pt_MsgTbl,
                                     DbcParserCanType* stt_pdu);

  /* init all the messages in pt_MsgTbl */
  static void ecal_com_dp_Init(const DbcParserMsgTblType* pt_MsgTbl,
                               uint16_t u16_tblSize);
  /* parse all the messages in pt_MsgTbl */
  static void ecal_com_dp_ParseAllMsg(const DbcParserMsgTblType* pt_MsgTbl,
                                      uint16_t u16_tblSize);
  /* parse one specific message "pt_Msg" */
  static void ecal_com_dp_ParseMsg(const DbcParserMsgTblType* pt_Msg);
  /* parse one signal from ptt_buffer */
  static void ecal_com_dp_ParseSignal(const DbcParserSignalTblType* pt_sigtbl,
                                      const uint8_t* ptt_buffer);
  /* set timeout value for message in pt_MsgTbl, whose id is canId */
  static int ecal_com_dp_setTimeoutByVal(const DbcParserMsgTblType* pt_MsgTbl,
                                         uint16_t u16_tblSize, uint32_t canId,
                                         uint32_t setVal);
  /* set timeout for all the messages in pt_MsgTbl by multiplying u16_mult to
   * period */
  static void ecal_com_dp_setTimeoutByMult(const DbcParserMsgTblType* pt_MsgTbl,
                                           uint16_t u16_tblSize,
                                           uint16_t u16_mult);
  /* set timeout for specific by multiplying u16_mult to period */
  static void ecal_com_dp_setTimeoutMultForMsg(
      const DbcParserMsgTblType* pt_Msg, uint16_t u16_mult);
  /* set message counter max value for specific message calculating from
   * u8_DataLen */
  static void ecal_com_dp_setMsgCntMaxForMsg(const DbcParserMsgTblType* pt_Msg);
  /* default crc calulation in vehicle */
  static uint32_t calcCrcValDefault(const uint8_t*, uint16_t, uint8_t);
  /* construct one signal into ptt_buffer */
  static void ecal_com_dp_ContructOneSignal(
      const DbcParserSignalTblType* pt_sigtbl, uint8_t* ptt_buffer);
  /* look up the message table in pt_MsgTbl according to canId */
  static const DbcParserMsgTblType* ecal_com_dp_GetMsgById(
      const DbcParserMsgTblType* pt_MsgTbl, uint16_t u16_tblSize,
      uint32_t canId);
  /* update the message counter */
  static void ecal_com_dp_UpdateMsgCnt(double*, uint16_t);
};
