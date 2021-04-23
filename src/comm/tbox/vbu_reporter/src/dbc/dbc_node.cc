// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include <assert.h>
// #include <sys/netmgr.h>
// #include <sys/neutrino.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <iostream>

#include "dbc/dbc_node.h"

#define GET_SEC(ms_v)  ((ms_v) / 1000)
#define GET_NSEC(ms_v) (((ms_v) % 1000) * 1000000)

// signal pulse define
// #define PULSE_CODE_TIMER_TIMEOUT   _PULSE_CODE_MINAVAIL
// typedef union {
//         struct _pulse   pulse;
//         /* your other message structures would go
//            here too */
// } TimerMsgType;

#define TIMEOUT_MIN_VALUE          (uint32_t)(150)  // ms
#define TIMEOUT_DEFAULT_MULT       (5)
#define TIMEOUT_RECOVERY_CNT_MAX   (5)
#define MSG_CNT_ERR_RANGE          (2)

// Dbc_node::Dbc_node(uint16_t in_period, const DbcParserDbcTblType*
// in_pt_DbcList,
//                    uint16_t in_u16_DbcNum)
Dbc_node::Dbc_node(uint16_t in_period, const DbcTableList* in_dbc_lists,
                   uint16_t in_dbc_lists_num)
    : u16_MainPeriod(in_period),
      // pt_DbcList(in_pt_DbcList),
      // u16_DbcNum(in_u16_DbcNum),
      b_RxStatus(false),
      b_TxStatus(false),
      u32_TimerCnt(0),
      losscomm_check_enable(false),
      lossCommHandler(nullptr),
      transmitHandler(nullptr),
      lossCommRecoveryHandler(nullptr) {
  assert(in_dbc_lists != nullptr);
  struct sigevent event;
  struct sched_param scheduling_params;
  int prio;

  /* set calcCrcValDefault as default crc handler*/
  setCalcCrcHandler(calcCrcValDefault);

  /* Get our priority. */
  // if (SchedGet( 0, 0, &scheduling_params) != -1)
  // {
  //     prio = scheduling_params.sched_priority;
  // }
  // else
  // {
  //     prio = 10;
  // }

  // EventChid = ChannelCreate(0);
  // event.sigev_notify = SIGEV_PULSE;
  // event.sigev_coid = ConnectAttach(ND_LOCAL_NODE, 0,
  //                                 EventChid,
  //                                 _NTO_SIDE_CHANNEL, 0);
  // event.sigev_priority = prio;
  // event.sigev_code = PULSE_CODE_TIMER_TIMEOUT;

  // timer_create(CLOCK_MONOTONIC, &event, &TimerId);

  /* Init all the message tables*/
  for (int i = 0; i < in_dbc_lists_num; i++) {
    for (int j = 0; j < in_dbc_lists[i].size_; j++) {
      // ecal_com_dp_Init(pt_DbcList[i].pt_MsgTbl, pt_DbcList[i].u16_tblSize);
      ecal_com_dp_Init(in_dbc_lists[i].dbc_list_[j].pt_MsgTbl,
                       in_dbc_lists[i].dbc_list_[j].u16_tblSize);
    }
  }
}

// TODO(terry.lu) : main function with channel
void Dbc_node::transmitMessages(int channel, const DbcParserDbcTblType* tblList,
                                uint16_t listSize) {
  for (int i = 0; i < listSize; i++) {
    ecal_com_dp_MainFunction(channel, tblList[i].pt_MsgTbl,
                             tblList[i].u16_tblSize);
  }
}

void Dbc_node::lockConstructMutex() { constructMutex.lock(); }

void Dbc_node::unlockConstructMutex() { constructMutex.unlock(); }

// TODO(terry.lu) : add channel
int Dbc_node::rxIndication(uint32_t canId, const uint8_t* rxBuff, uint16_t dlc,
                           const DbcParserDbcTblType* msgTblList,
                           uint16_t listSize) {
  int ret;
  for (int i = 0; i < listSize; i++) {
    // for (int j = 0; j < dlc; j++) {
    //   printf("%2.2x ", rxBuff[j]);
    // }
    ret = ecal_com_dp_RxIndication(canId, dlc, rxBuff, msgTblList[i].pt_MsgTbl,
                                   msgTblList[i].u16_tblSize);
    // std::cout << "ret : " << ret << std::endl;
    /* if canId is found, then do not try to loop up in next message table*/
    if (ret != DBC_ERR_ID_NOT_FOUND) {
      break;
    }
  }
  // TODO(terry) : seperate code
  // for (int i = 0; i < u16_DbcNum; i++) {
  //   ecal_com_dp_MainFunction(pt_DbcList[i].pt_MsgTbl,
  //                            pt_DbcList[i].u16_tblSize);
  // }

  return ret;
}

void Dbc_node::parseMsg(const DbcParserDbcTblType* tblList, uint16_t listSize) {
  for (int i = 0; i < listSize; i++) {
    ecal_com_dp_ParseAllMsg(tblList[i].pt_MsgTbl, tblList[i].u16_tblSize);
  }
}

// int Dbc_node::setTimeout(uint32_t canId, uint16_t timeout) {
//   int ret = DBC_ERR_OK;
//   for (int i = 0; i < u16_DbcNum; i++) {
//     ret = ecal_com_dp_setTimeoutByVal(
//         pt_DbcList[i].pt_MsgTbl, pt_DbcList[i].u16_tblSize, canId, timeout);
//     if (DBC_ERR_OK == ret) {
//       break;
//     }
//   }
//   return ret;
// }

// void Dbc_node::setTimeoutByMult(uint16_t mult) {
//   for (int i = 0; i < u16_DbcNum; i++) {
//     for (int j = 0; j < pt_DbcList[i].u16_tblSize; j++) {
//       ecal_com_dp_setTimeoutMultForMsg(&((pt_DbcList[i].pt_MsgTbl)[j]), mult);
//     }
//   }
// }

void Dbc_node::setLossCommHandler(std::function<void(uint32_t)> handler) {
  lossCommHandler = handler;
}

void Dbc_node::setLossCommCheckEnable(bool status) {
  losscomm_check_enable = status;
}

void Dbc_node::setTransmitHandler(
    std::function<int(uint16_t, uint32_t, uint8_t*, uint16_t)> handler) {
  transmitHandler = handler;
}

void Dbc_node::setLossCommRecoveryHandler(
    std::function<void(uint32_t)> handler) {
  lossCommRecoveryHandler = handler;
}

void Dbc_node::setCalcCrcHandler(
    std::function<uint32_t(const uint8_t*, uint16_t, uint8_t)> handler) {
  calcCrcHandler = handler;
}

void Dbc_node::loopHandler() {
  // TimerMsgType  msg;
  int rcvid;

  while (1) {
    // rcvid = MsgReceive(EventChid, &msg, sizeof(msg), NULL);
    // /* MsgReceive is unblocked when timer is alarmed*/
    // if (0 == rcvid)
    // {
    //     for (int i = 0; i < u16_DbcNum; i++)
    //     {
    //         ecal_com_dp_MainFunction(pt_DbcList[i].pt_MsgTbl,
    //         pt_DbcList[i].u16_tblSize);
    //     }
    // }

    // for (int i = 0; i < u16_DbcNum; i++) {
    //   ecal_com_dp_MainFunction(pt_DbcList[i].pt_MsgTbl,
    //                            pt_DbcList[i].u16_tblSize);
    // }
    // TODO(terry) : set linux timer
  }
}

// void Dbc_node::setTimer(int timer_value)
// {
//     struct itimerspec itime;
//     /* set timer */
//     itime.it_value.tv_sec  = GET_SEC(timer_value);
//     itime.it_value.tv_nsec = GET_NSEC(timer_value);
//     /* set to non-zero for circular timer */
//     itime.it_interval.tv_sec = GET_SEC(timer_value);
//     itime.it_interval.tv_nsec = GET_NSEC(timer_value);
//     /* start timer if the values in itime is not zero*/
//     timer_settime(TimerId, 0, &itime, NULL);
// }

void Dbc_node::start() {
  // if (!mainThread.joinable()) {
  //   mainThread = std::thread(&Dbc_node::loopHandler, this);
  // }
  // TODO(terry) : start thread
  b_TxStatus = true;
  b_RxStatus = true;
  // setTimer(u16_MainPeriod);
}

void Dbc_node::stop() {
  b_TxStatus = false;
  b_RxStatus = false;
  // setTimer(0);
  // TODO(terry) : start thread
}

void Dbc_node::setRxStatus(bool setVal) { b_RxStatus = setVal; }

void Dbc_node::setTxStatus(bool setVal) { b_TxStatus = setVal; }

void Dbc_node::ecal_com_dp_Init(const DbcParserMsgTblType* pt_MsgTbl,
                                uint16_t u16_tblSize) {
  uint16_t u16t_loop;

  for (u16t_loop = (uint16_t)0; u16t_loop < u16_tblSize; u16t_loop++) {
    (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u8_RxFlag = (uint8_t)OFF;
    (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_AgeCounter = (uint32_t)0;
    (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_RecoveryCnt = (uint16_t)0;
    (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_CrcErrCnt = (uint32_t)0;
    (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_LostCommCnt = (uint32_t)0;
    (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_MsgCntErrCnt = (uint32_t)0;
    (pt_MsgTbl[u16t_loop].pt_MsgStatus)->firstFrameFlag = true;
    (pt_MsgTbl[u16t_loop].pt_MsgStatus)->txEnFlag = true;
    ecal_com_dp_setTimeoutMultForMsg(&pt_MsgTbl[u16t_loop],
                                     TIMEOUT_DEFAULT_MULT);
    ecal_com_dp_setMsgCntMaxForMsg(&pt_MsgTbl[u16t_loop]);
  }
}

void Dbc_node::ecal_com_dp_setTimeoutMultForMsg(
    const DbcParserMsgTblType* pt_Msg, uint16_t u16_mult) {
  if ((DP_MSGDIR_RX == pt_Msg->u8_Dir) &&
      (DP_PERIODICMACRO == pt_Msg->u8_SendType) && (pt_Msg->u16_Period != 0)) {
    (pt_Msg->pt_MsgStatus)->u32_Timeout =
        (uint32_t)(u16_mult * pt_Msg->u16_Period);
    /* there is a min value for u32_Timeout */
    if ((pt_Msg->pt_MsgStatus)->u32_Timeout < TIMEOUT_MIN_VALUE) {
      (pt_Msg->pt_MsgStatus)->u32_Timeout = TIMEOUT_MIN_VALUE;
    }
  } else {
    (pt_Msg->pt_MsgStatus)->u32_Timeout = 0;
  }
}

void Dbc_node::ecal_com_dp_setMsgCntMaxForMsg(
    const DbcParserMsgTblType* pt_Msg) {
  uint16_t msgCntIndex = pt_Msg->u16_MsgCntIndex;
  if (DP_MSG_INVALID_INDEX != msgCntIndex) {
    const DbcParserSignalTblType* pt_msgCntPtr =
        &((pt_Msg->pt_SignalTbl)[msgCntIndex]);
    (pt_Msg->pt_MsgStatus)->u16_MsgCntMax = (1 << pt_msgCntPtr->u8_DataLen) - 1;
  } else {
    (pt_Msg->pt_MsgStatus)->u16_MsgCntMax = 0;
  }
}

// int Dbc_node::DisablePeriodicTransmitMsg(uint32_t canId) {
//   int ret = -1;
//   for (int i = 0; i < u16_DbcNum; i++) {
//     for (uint16_t u16t_loop = (uint16_t)0;
//          u16t_loop < pt_DbcList[i].u16_tblSize; u16t_loop++) {
//       if (DP_MSGDIR_TX == pt_DbcList[i].pt_MsgTbl[u16t_loop].u8_Dir) {
//         if (canId == pt_DbcList[i].pt_MsgTbl[u16t_loop].u32_CanId) {
//           pt_DbcList[i].pt_MsgTbl[u16t_loop].pt_MsgStatus->txEnFlag = false;
//           ret = 0;
//         }
//       }
//     }
//   }
//   return ret;
// }

int Dbc_node::DirectTransmitMsg(uint16_t channel, uint32_t canId, uint8_t* buff,
                                uint16_t dlc) {
  int ret = -1;
  if ((transmitHandler != nullptr) && (buff != nullptr)) {
    std::lock_guard<std::mutex> lock(directTxMutex);
    ret = transmitHandler(channel, canId, buff, dlc);
  }
  return ret;
}

void Dbc_node::ecal_com_dp_MainFunction(int channel,
                                        const DbcParserMsgTblType* pt_MsgTbl,
                                        uint16_t u16_tblSize) {
  /*if lossCommHandler is null, there is no need to do the timeout process
   if transmitHandler is null, there is no need to transmit tx messages*/
  bool rxProcess = (b_RxStatus && lossCommHandler);
  bool txProcess = (b_TxStatus && transmitHandler);

  if ((!rxProcess) && (!txProcess)) {
    return;
  }

  uint64_t u64t_time;
  u32_TimerCnt++;
  u64t_time = ((uint64_t)u32_TimerCnt) * ((uint64_t)u16_MainPeriod);

  for (uint16_t u16t_loop = (uint16_t)0; u16t_loop < u16_tblSize; u16t_loop++) {
    if (DP_PERIODICMACRO != pt_MsgTbl[u16t_loop].u8_SendType) {
      continue;
    }

    if (DP_MSGDIR_RX == pt_MsgTbl[u16t_loop].u8_Dir) {
      if ((rxProcess) && losscomm_check_enable &&
          ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_Timeout > 0)) {
        /* check losscomm enable flag */
        if ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_AgeCounter <=
            (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_Timeout) {
          ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_AgeCounter) +=
              u16_MainPeriod;
          /* when u32_AgeCounter reaches u32_Timeout, we call it one timeout*/
          if ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_AgeCounter >
              (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_Timeout) {
            /*set u32_AgeCounter to 0 again to trigger more timeout for plusing
             * the u16_RecoveryCnt*/
            (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_AgeCounter = 0;
            /*u16_RecoveryCnt is 0 means this message is in good condition
             before, and timeout occurs now, which means we need to report the
             loss comm*/
            if (0 == (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_RecoveryCnt) {
              ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_LostCommCnt)++;
              lossCommHandler(pt_MsgTbl[u16t_loop].u32_CanId);
              (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_RecoveryCnt =
                  TIMEOUT_RECOVERY_CNT_MAX;
            } else {
              /*If timeour occurs when u16_RecoveryCnt is not 0, we need to
               * u16_RecoveryCnt++*/
              if ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_RecoveryCnt <
                  TIMEOUT_RECOVERY_CNT_MAX) {
                ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_RecoveryCnt)++;
              }
            }
          }
        }
      } else {
        if ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_AgeCounter != 0) {
          (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_AgeCounter = 0;
        }
      }
    } else {
      if (txProcess) {
        // TODO(terry.lu) : fix the periodic loop
        // if ((0 != pt_MsgTbl[u16t_loop].u16_Period) &&
        //     (u64t_time % pt_MsgTbl[u16t_loop].u16_Period == 0) &&
        //     (pt_MsgTbl[u16t_loop].pt_MsgStatus->txEnFlag == true)) {
        if ((pt_MsgTbl[u16t_loop].pt_MsgStatus->txEnFlag == true)) {
          std::lock_guard<std::mutex> lock(constructMutex);
          /*update message counter before calling
           * ecal_com_dp_ContructSignalMsg*/
          uint16_t msgCntIndex = pt_MsgTbl[u16t_loop].u16_MsgCntIndex;
          DbcParserCanType txPdu;
          if (msgCntIndex != DP_MSG_INVALID_INDEX) {
            const DbcParserSignalTblType* ptt_sigtbl =
                pt_MsgTbl[u16t_loop].pt_SignalTbl;
            ecal_com_dp_UpdateMsgCnt(
                (double*)ptt_sigtbl[msgCntIndex].ad_Addr,
                (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_MsgCntMax);
          }
          /*construct the message into txPdu*/
          ecal_com_dp_ContructSignalMsg(&pt_MsgTbl[u16t_loop], &txPdu);
          /*transmit the txPdu by calling DirectTransmitMsg */
          // DirectTransmitMsg(txPdu.u32_canid, txPdu.pt_buff, txPdu.u16_dlc);
          // TODO(terry.lu): add channel num
          DirectTransmitMsg(channel, txPdu.u32_canid, txPdu.pt_buff,
                            txPdu.u16_dlc);
        }
      }
    }
  }
}

void Dbc_node::ecal_com_dp_ParseAllMsg(const DbcParserMsgTblType* pt_MsgTbl,
                                       uint16_t u16_tblSize) {
  uint16_t u16t_loop;

  for (u16t_loop = (uint16_t)0; u16t_loop < u16_tblSize; u16t_loop++) {
    if (pt_MsgTbl[u16t_loop].u8_Dir == DP_MSGDIR_RX) {
      std::lock_guard<std::mutex> lock(
          (pt_MsgTbl[u16t_loop].pt_MsgStatus)->MsgMutex);
      if ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u8_RxFlag == (uint8_t)ON) {
        if (pt_MsgTbl[u16t_loop].u16_TblSize != 0) {
          ecal_com_dp_ParseMsg(&pt_MsgTbl[u16t_loop]);
        }
        (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u8_RxFlag = (uint8_t)OFF;
      }
    }
  }
}

void Dbc_node::ecal_com_dp_ParseSignal(const DbcParserSignalTblType* pt_sigtbl,
                                       const uint8_t* ptt_buffer) {
  assert(pt_sigtbl != nullptr);
  assert(ptt_buffer != nullptr);

  uint64_t u64t_value;
  int64_t s64t_value;
  double f32t_factor;
  double f32t_offset;
  double f32t_max;
  double f32t_min;
  double f32t_physval;
  double f64t_physval;
  uint8_t u8t_byteorder;
  uint16_t u16t_startpos;
  int8_t s8t_signallen;
  uint16_t u16t_bytepos;
  int8_t s8t_startbit;
  uint8_t u8t_tempdata;
  int8_t s8t_shiftbit;
  float t_value = 0.0f;
  uint32_t* ptt_value;
  uint64_t* ptt_value64;

  u64t_value = (uint64_t)0;
  f32t_factor = pt_sigtbl->f32_Factor;
  f32t_offset = pt_sigtbl->f32_Offset;
  f32t_max = pt_sigtbl->f32_Max;
  f32t_min = pt_sigtbl->f32_Min;
  u8t_byteorder = pt_sigtbl->u8_ByteOrder;
  u16t_startpos = pt_sigtbl->u16_StartBit;
  s8t_signallen = (int8_t)(pt_sigtbl->u8_DataLen);

  if (u8t_byteorder == DP_BYTEORDER_MOTOROLA) {
    u16t_bytepos = u16t_startpos / 8;
    s8t_startbit = (int16_t)u16t_startpos % 8;
    u8t_tempdata = ptt_buffer[u16t_bytepos] << (8 - s8t_startbit - 1);
    u8t_tempdata = u8t_tempdata >> (8 - s8t_startbit - 1);
    if (s8t_signallen - s8t_startbit - 1 >= 0) {
      s8t_signallen = s8t_signallen - s8t_startbit - 1;
      u64t_value = ((uint64_t)u8t_tempdata << s8t_signallen);
      s8t_signallen -= 8;
      while (s8t_signallen > 0) {
        u16t_bytepos++;
        u64t_value |= ((uint64_t)ptt_buffer[u16t_bytepos] << s8t_signallen);
        s8t_signallen -= 8;
      }
      if (s8t_signallen == 0) {
        u16t_bytepos++;
        u64t_value |= ((uint64_t)ptt_buffer[u16t_bytepos] << s8t_signallen);
      } else if (s8t_signallen < 0) {
        s8t_signallen = 0 - s8t_signallen;
        if (s8t_signallen < 8) {
          u16t_bytepos++;
          u64t_value |= ((uint64_t)ptt_buffer[u16t_bytepos] >> s8t_signallen);
        }
      }
    } else {
      if (s8t_signallen > 8 - s8t_startbit) {
        if (s8t_signallen - s8t_startbit > 0) {
          u64t_value |= ((uint64_t)ptt_buffer[u16t_bytepos] >> s8t_startbit);
          u16t_bytepos++;
          u8t_tempdata = ptt_buffer[u16t_bytepos]
                         << (s8t_startbit - s8t_signallen);
          u64t_value |=
              ((uint64_t)u8t_tempdata >> (s8t_startbit - s8t_signallen));
        } else {
          u8t_tempdata = (ptt_buffer[u16t_bytepos] << (8 - s8t_startbit - 1));
          u64t_value |= ((uint64_t)u8t_tempdata >> (8 - s8t_signallen));
        }
      } else {
        u64t_value |=
            ((uint64_t)u8t_tempdata >> (s8t_startbit + 1 - s8t_signallen));
      }
    }
  } else {
    u16t_bytepos = u16t_startpos / 8;
    s8t_shiftbit = (int16_t)u16t_startpos % 8;
    if (8 - s8t_shiftbit < s8t_signallen) {
      u64t_value |= ((uint64_t)ptt_buffer[u16t_bytepos] >> s8t_shiftbit);
      s8t_startbit = 8 - s8t_shiftbit;
      while (s8t_signallen - (uint8_t)s8t_startbit >= 8) {
        u16t_bytepos++;
        u64t_value |= ((uint64_t)ptt_buffer[u16t_bytepos] << s8t_startbit);
        s8t_startbit += 8;
      }
      u16t_bytepos++;
      u8t_tempdata = ptt_buffer[u16t_bytepos]
                     << (8 - s8t_signallen + (uint8_t)s8t_startbit);
      u8t_tempdata =
          u8t_tempdata >> (8 - s8t_signallen + (uint8_t)s8t_startbit);
      u64t_value |= ((uint64_t)u8t_tempdata << s8t_startbit);
    } else {
      s8t_startbit = 8 - s8t_shiftbit - s8t_signallen;
      u8t_tempdata = ptt_buffer[u16t_bytepos] << (s8t_startbit);
      u8t_tempdata = u8t_tempdata >> (s8t_startbit + s8t_shiftbit);
      u64t_value = (uint64_t)u8t_tempdata;
    }
  }
  switch (pt_sigtbl->u8_Type) {
    case DP_SIGNALTYPE_UINT8:
      break;
    case DP_SIGNALTYPE_UINT16:
      break;
    case DP_SIGNALTYPE_UINT32:
      break;
    case DP_SIGNALTYPE_SINT8:
      break;
    case DP_SIGNALTYPE_SINT16:
      break;
    case DP_SIGNALTYPE_SINT32:
      break;
    case DP_SIGNALTYPE_FLOAT:
      if (pt_sigtbl->u8_SignType == DP_SIGNTYPE_SIGNED &&
          pt_sigtbl->u8_DataLen > 1) {
        if (u64t_value >> (pt_sigtbl->u8_DataLen - 1) == 1) {
          s64t_value = 0 - (int64_t)(((~u64t_value + 1)
                                      << (64 - pt_sigtbl->u8_DataLen)) >>
                                     (64 - pt_sigtbl->u8_DataLen));
        } else {
          s64t_value = (int64_t)u64t_value;
        }
        f32t_physval = s64t_value * f32t_factor + f32t_offset;
        if (f32t_physval > f32t_max) {
          f32t_physval = f32t_max;
        } else if (f32t_physval < f32t_min) {
          f32t_physval = f32t_min;
        }
        *((double*)(pt_sigtbl->ad_Addr)) = f32t_physval;
      } else {
        f32t_physval = u64t_value * f32t_factor + f32t_offset;
        if (f32t_physval > f32t_max) {
          f32t_physval = f32t_max;
        } else if (f32t_physval < f32t_min) {
          f32t_physval = f32t_min;
        }
        *((double*)(pt_sigtbl->ad_Addr)) = f32t_physval;
      }
      break;
    case DP_SIGNALTYPE_IEEEFLOAT:
      ptt_value = (uint32_t*)&t_value;
      *ptt_value = (uint32_t)u64t_value;
      f32t_physval = (double)(t_value);
      *((double*)(pt_sigtbl->ad_Addr)) = f32t_physval;
      break;
    case DP_SIGNALTYPE_IEEEDOUBLE:
      ptt_value64 = (uint64_t*)&t_value;
      *ptt_value64 = (uint64_t)u64t_value;
      f64t_physval = (double)(t_value);
      *((double*)(pt_sigtbl->ad_Addr)) = f64t_physval;
      break;
    default:
      break;
  }
}

void Dbc_node::ecal_com_dp_ParseMsg(const DbcParserMsgTblType* pt_Msg) {
  uint32_t u32t_loop;
  const DbcParserSignalTblType* ptt_sigtbl;
  uint8_t* ptt_buffer;

  if (pt_Msg->u8_Dir != DP_MSGDIR_RX) {
    return;
  } else {
    ptt_buffer = pt_Msg->pt_Buffer;
    ptt_sigtbl = pt_Msg->pt_SignalTbl;
  }

  if (pt_Msg->u8_MsgParseType == DP_MSGKIND_NORMAL) {
    for (u32t_loop = 0; u32t_loop < pt_Msg->u16_TblSize; u32t_loop++) {
      ecal_com_dp_ParseSignal(&ptt_sigtbl[u32t_loop], ptt_buffer);
    }
  } else if (pt_Msg->u8_MsgParseType == DP_MSGKIND_GROUP) {
    uint8_t t_groupIdx = 0;
    uint16_t t_sigIdx = 0;
    for (u32t_loop = 0; u32t_loop < pt_Msg->u16_TblSize; u32t_loop++) {
      /*update t_sigIdx according to t_groupIdx when it's not the first signal*/
      if (u32t_loop != 0) {
        t_sigIdx = u32t_loop + pt_Msg->u16_TblSize * t_groupIdx;
      }

      ecal_com_dp_ParseSignal(&ptt_sigtbl[t_sigIdx], ptt_buffer);

      if (u32t_loop == 0) {
        t_groupIdx = (uint8_t)(*((double*)(ptt_sigtbl[u32t_loop].ad_Addr)));
      }
    }
  }
}

void Dbc_node::ecal_com_dp_UpdateMsgCnt(double* p_msgCnt, uint16_t msgCntMax) {
  if (*p_msgCnt >= (double)msgCntMax) {
    *p_msgCnt = 0;
  } else {
    (*p_msgCnt)++;
  }
}

int Dbc_node::check_counter_diff(uint16_t last, uint16_t received,
                                 uint16_t msgCntMax) {
  uint16_t diff = 0;

  if (received > last) {
    diff = received - last;
  } else {
    diff = received + (msgCntMax + 1) - last;
  }
  return diff;
}

int Dbc_node::ecal_com_dp_RxIndication(uint32_t u32t_canid,
                                       uint16_t u8t_datalen,
                                       const uint8_t* ptt_buffer,
                                       const DbcParserMsgTblType* pt_MsgTbl,
                                       uint16_t u16_tblSize) {
  uint16_t u16t_loop;
  int ret = DBC_ERR_ID_NOT_FOUND;

  if (b_RxStatus) {
    for (u16t_loop = (uint16_t)0; u16t_loop < u16_tblSize; u16t_loop++) {
      if ((u32t_canid == pt_MsgTbl[u16t_loop].u32_CanId) &&
          (pt_MsgTbl[u16t_loop].u8_Dir == DP_MSGDIR_RX)) {
        if ((pt_MsgTbl[u16t_loop].u8_SendType == DP_PERIODICMACRO) &&
            (lossCommRecoveryHandler)) {
          /*set u32_AgeCounter to 0 to make it being not timeout*/
          (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_AgeCounter = 0;
          if ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_RecoveryCnt > 0) {
            ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_RecoveryCnt)--;
            /*consider as loss comm recovery only when u16_RecoveryCnt decreased
             * to 0*/
            if (0 == (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_RecoveryCnt) {
              lossCommRecoveryHandler(u32t_canid);
            }
          }
        }

        std::lock_guard<std::mutex> lock(
            (pt_MsgTbl[u16t_loop].pt_MsgStatus)->MsgMutex);


        /*check the crc*/
        uint16_t crcIndex = pt_MsgTbl[u16t_loop].u16_CrcIndex;
        if ((DP_MSG_INVALID_INDEX != crcIndex) && (calcCrcHandler)) {
          const DbcParserSignalTblType* pt_crcPtr =
              &((pt_MsgTbl[u16t_loop].pt_SignalTbl)[crcIndex]);
          ecal_com_dp_ParseSignal(pt_crcPtr, ptt_buffer);
          uint32_t crcValueGot = (uint32_t)(*((double*)pt_crcPtr->ad_Addr));
          uint32_t crcCalc = calcCrcHandler(ptt_buffer, u8t_datalen,
                                            pt_crcPtr->u16_StartBit / 8) &
                             ((1 << (pt_crcPtr->u8_DataLen)) - 1);
          if (crcValueGot != crcCalc) {
            ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_CrcErrCnt)++;
            ret = DBC_ERR_CRC_ERR;
            break;
          }
        }


        /*check the msg counter*/
        if ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->firstFrameFlag == true) {
          /* report loss comm test result on first frame */
          if ((pt_MsgTbl[u16t_loop].u8_SendType == DP_PERIODICMACRO) &&
              (lossCommRecoveryHandler)) {
            lossCommRecoveryHandler(u32t_canid);
          }

          (pt_MsgTbl[u16t_loop].pt_MsgStatus)->firstFrameFlag = false;
        } else {
          uint16_t msgCntIndex = pt_MsgTbl[u16t_loop].u16_MsgCntIndex;
          if (DP_MSG_INVALID_INDEX != msgCntIndex) {
            const DbcParserSignalTblType* pt_msgCntPtr =
                &((pt_MsgTbl[u16t_loop].pt_SignalTbl)[msgCntIndex]);
            double lastCnt = *((double*)pt_msgCntPtr->ad_Addr);

            ecal_com_dp_ParseSignal(pt_msgCntPtr, ptt_buffer);
            double rxedCnt = *((double*)pt_msgCntPtr->ad_Addr);

            uint16_t counterDiff = check_counter_diff(
                lastCnt, rxedCnt,
                (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u16_MsgCntMax);

            if (counterDiff == 1) {
              // OK
              ret = DBC_ERR_OK;
            } else {
              // ERROR
              ((pt_MsgTbl[u16t_loop].pt_MsgStatus)->u32_MsgCntErrCnt)++;
              ret = DBC_ERR_MSG_CNT_ERR;
              break;
            }
          }
        }


        /*copy the received data into message storage*/
        memcpy(pt_MsgTbl[u16t_loop].pt_Buffer, ptt_buffer, u8t_datalen);
        if (pt_MsgTbl[u16t_loop].u8_MsgParseType == DP_MSGKIND_NORMAL) {
          (pt_MsgTbl[u16t_loop].pt_MsgStatus)->u8_RxFlag = (uint8_t)ON;
        } else {
          /*Parse the message directly when it's DP_MSGKIND_GROUP type*/
          ecal_com_dp_ParseMsg(&pt_MsgTbl[u16t_loop]);
        }
        ret = DBC_ERR_OK;
        break;
      }
    }
  } else {
    ret = DBC_ERR_RX_DISABLE;
  }

  return ret;
}

void Dbc_node::ecal_com_dp_ContructOneSignal(
    const DbcParserSignalTblType* ptt_sigtbl, uint8_t* u8t_buffer) {
  uint64_t u64t_value;
  int64_t s64t_value;
  double f32t_factor;
  double f32t_offset;
  double f32t_physval;
  float f4t_physval;
  double f8t_physval;
  uint16_t u16t_startpos;
  uint8_t u8t_datalen;
  uint8_t u8t_byteorder;
  int8_t s8t_shiftbit;
  uint16_t u16t_bytepos;
  uint8_t u8t_tempdatalen;
  uint64_t u64t_temp_1;
  uint32_t* ptt_val;
  uint64_t* ptt_val64;

  u16t_startpos = ptt_sigtbl->u16_StartBit;
  u8t_datalen = ptt_sigtbl->u8_DataLen;
  u8t_byteorder = ptt_sigtbl->u8_ByteOrder;
  f32t_factor = ptt_sigtbl->f32_Factor;
  f32t_offset = ptt_sigtbl->f32_Offset;
  u64t_value = (uint64_t)0;
  switch (ptt_sigtbl->u8_Type) {
    case DP_SIGNALTYPE_UINT8:
      break;
    case DP_SIGNALTYPE_UINT16:
      break;
    case DP_SIGNALTYPE_UINT32:
      break;
    case DP_SIGNALTYPE_SINT8:
      break;
    case DP_SIGNALTYPE_SINT16:
      break;
    case DP_SIGNALTYPE_SINT32:
      break;
    case DP_SIGNALTYPE_FLOAT:
      f32t_physval = *((double*)(ptt_sigtbl->ad_Addr));
      if (ptt_sigtbl->u8_SignType == DP_SIGNTYPE_SIGNED &&
          ptt_sigtbl->u8_DataLen > 1) {
        /*Add 0.5 to round f32t_physval value*/
        s64t_value =
            (int64_t)((f32t_physval - f32t_offset) / f32t_factor + 0.5);
        if (s64t_value >= 0) {
        } else {
          s64t_value = (s64t_value & 0x7FFFFFFF);
          s64t_value =
              s64t_value | (((int64_t)1) << (ptt_sigtbl->u8_DataLen - 1));
          u64t_temp_1 = 0;
          for (int si = 0; si < ptt_sigtbl->u8_DataLen; si++) {
            u64t_temp_1 = (u64t_temp_1 << 1) | 1;
          }
          s64t_value = s64t_value & u64t_temp_1;
        }
        u64t_value = (uint64_t)s64t_value;
      } else {
        /*Add 0.5 to round f32t_physval value*/
        u64t_value =
            (uint64_t)((f32t_physval - f32t_offset) / f32t_factor + 0.5);
      }
      break;
    case DP_SIGNALTYPE_IEEEFLOAT:
      f4t_physval = *((float*)(ptt_sigtbl->ad_Addr));
      ptt_val = (uint32_t*)&f4t_physval;
      u64t_value = *ptt_val;
      break;
    case DP_SIGNALTYPE_IEEEDOUBLE:
      f8t_physval = *((double*)(ptt_sigtbl->ad_Addr));
      ptt_val64 = (uint64_t*)&f8t_physval;
      u64t_value = *ptt_val64;
      break;
    default:
      break;
  }
  if (u8t_byteorder == DP_BYTEORDER_MOTOROLA) {
    u16t_bytepos = u16t_startpos / 8;
    s8t_shiftbit =
        (int8_t)((int8_t)u8t_datalen - (int8_t)(u16t_startpos % 8) - (int8_t)1);
    if (s8t_shiftbit >= 0) {
      u8t_buffer[u16t_bytepos] =
          u8t_buffer[u16t_bytepos] | ((uint8_t)(u64t_value >> s8t_shiftbit));
    } else {
      u8t_buffer[u16t_bytepos] = u8t_buffer[u16t_bytepos] |
                                 ((uint8_t)(u64t_value << (0 - s8t_shiftbit)));
    }
    while (s8t_shiftbit > 0) {
      s8t_shiftbit -= 8;
      u16t_bytepos++;
      u8t_datalen -= 8;
      if (s8t_shiftbit >= 0) {
        u8t_buffer[u16t_bytepos] =
            u8t_buffer[u16t_bytepos] | ((uint8_t)(u64t_value >> s8t_shiftbit));
      } else {
        u8t_buffer[u16t_bytepos] =
            u8t_buffer[u16t_bytepos] |
            ((uint8_t)(u64t_value << (0 - s8t_shiftbit)));
      }
    }
  } else {
    u16t_bytepos = u16t_startpos / 8;
    s8t_shiftbit = (int8_t)(u16t_startpos % 8);
    u8t_buffer[u16t_bytepos] =
        u8t_buffer[u16t_bytepos] | ((uint8_t)(u64t_value << s8t_shiftbit));
    u8t_tempdatalen = 8 - (uint8_t)s8t_shiftbit;
    while (u8t_datalen - u8t_tempdatalen > 0) {
      u16t_bytepos++;
      u8t_buffer[u16t_bytepos] =
          u8t_buffer[u16t_bytepos] | ((uint8_t)(u64t_value >> u8t_tempdatalen));
      u8t_tempdatalen += 8;
    }
  }
}

void Dbc_node::ecal_com_dp_ContructSignalMsg(
    const DbcParserMsgTblType* pt_MsgTbl, DbcParserCanType* stt_pdu) {
  if (pt_MsgTbl->u8_Dir != DP_MSGDIR_TX) {
    return;
  }
  uint16_t u16t_msglen = pt_MsgTbl->u16_MsgLen;
  uint8_t* u8t_buffer = pt_MsgTbl->pt_Buffer;
  uint16_t crcIndex = pt_MsgTbl->u16_CrcIndex;

  memset(u8t_buffer, 0, u16t_msglen);
  /*construct all the signals except CRC*/
  for (uint16_t u16t_loop = 0; u16t_loop < pt_MsgTbl->u16_TblSize;
       u16t_loop++) {
    if (u16t_loop != crcIndex) {
      ecal_com_dp_ContructOneSignal(&(pt_MsgTbl->pt_SignalTbl[u16t_loop]),
                                    u8t_buffer);
    }
  }

  /*construct CRC at last*/
  if ((DP_MSG_INVALID_INDEX != crcIndex) && (calcCrcHandler)) {
    const DbcParserSignalTblType* pt_crcPtr =
        &((pt_MsgTbl->pt_SignalTbl)[crcIndex]);
    *((double*)pt_crcPtr->ad_Addr) =
        calcCrcHandler(u8t_buffer, u16t_msglen, (pt_crcPtr->u16_StartBit) / 8) &
        ((1 << (pt_crcPtr->u8_DataLen)) - 1);
    ecal_com_dp_ContructOneSignal(pt_crcPtr, u8t_buffer);
  }

  stt_pdu->u32_canid = pt_MsgTbl->u32_CanId;
  stt_pdu->u16_dlc = u16t_msglen;
  stt_pdu->pt_buff = u8t_buffer;
}

const DbcParserMsgTblType * Dbc_node::ecal_com_dp_GetMsgById(const DbcParserMsgTblType * pt_MsgTbl,
                                                       uint16_t u16_tblSize,
                                                       uint32_t canId)
{
    const DbcParserMsgTblType * retMsg = nullptr;
    for ( uint16_t u16t_loop = (uint16_t)0 ; u16t_loop < u16_tblSize; u16t_loop++ )
    {
        if ( canId == pt_MsgTbl[u16t_loop].u32_CanId )
        {
            retMsg = &pt_MsgTbl[u16t_loop];
            break;
        }
    }
    return retMsg;
}

int Dbc_node::ecal_com_dp_setTimeoutByVal(const DbcParserMsgTblType * pt_MsgTbl, uint16_t u16_tblSize,
                                          uint32_t canId, uint32_t setVal)
{
    const DbcParserMsgTblType* foundMsg = ecal_com_dp_GetMsgById(pt_MsgTbl,u16_tblSize,canId);
    int ret;
    if (foundMsg)
    {
        if ( (DP_MSGDIR_RX == foundMsg->u8_Dir)
          && (DP_PERIODICMACRO == foundMsg->u8_SendType) )
        {
            (foundMsg->pt_MsgStatus)->u32_Timeout = setVal;
            (foundMsg->pt_MsgStatus)->u32_AgeCounter = 0;
        }
        ret = DBC_ERR_OK;
    }
    else
    {
        ret = DBC_ERR_ID_NOT_FOUND;
    }
    return ret;
}

void Dbc_node::ecal_com_dp_setTimeoutByMult(const DbcParserMsgTblType * pt_MsgTbl, uint16_t u16_tblSize,
                                            uint16_t u16_mult)
{
    for ( uint16_t u16t_loop = (uint16_t)0 ; u16t_loop < u16_tblSize; u16t_loop++ )
    {
        ecal_com_dp_setTimeoutMultForMsg(&pt_MsgTbl[u16t_loop], u16_mult);
    }
}

uint32_t Dbc_node::calcCrcValDefault(const uint8_t* data,uint16_t length,uint8_t crcIndex)
{
    uint8_t CRC = 0xFF;
    int f, b;
    for (f = 0; f < length; f++)
    {
        if (f != crcIndex)
        {
            CRC ^= data[f];
            for (b = 0; b < 8; b++)
            {
                if ((CRC & 0x80) != 0)
                {
                    CRC <<= 1;
                    CRC ^= 0x1D;
                }
                else
                    CRC <<= 1;
            }
        }
    }
    return (uint32_t)(~CRC);
}
