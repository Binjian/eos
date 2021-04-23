#./get_info.py ./VBU0401_02_a_Rep.a2l -o ./vbu.json -p TQD_trqTrqSetECO_MAP_v CrCtl_facDragSpeed_MAP_v TQD_flgBrkRgnConfig_C
./get_info.py VBU0414_Rep.a2l -o ./vbu.json -p TQD_trqTrqSetECO_MAP_v
./xcp_send.py vbu.json -o out.json
