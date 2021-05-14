#!/usr/bin/env python3

import argparse
import json
# import system.file as file


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("a2l", help="a2l file path")
    parser.add_argument('-p', '--parameters', nargs='+',
                        help='<Required> list of parameter names', required=True)
    parser.add_argument('-o', '--output',
                        help='<Required> output json file name', required=True)
    parser.add_argument('-i', '--init',
                        help='<Required> init json file name', required=True)
    args = parser.parse_args()
    return args

def find_word_in_line(line,word):
    for w in line.split():
        if w == word:
            return True
    return False

def find_line_infile(file,*str):
    while True:
        line = file.readline()
        if not line:
            print(line)
            break
        found_all = True
        for s in str:
            found_s = find_word_in_line(line,s)
            if not found_s:
                found_all = False
                break
        if found_all:
            return line
    return None

def find_line_index(lines,*str):
    for i in range(len(lines)):
        found_all = True
        for s in str:
            found_s = find_word_in_line(lines[i],s)
            if not found_s:
                found_all = False
                break
        if found_all:
            return i
    return -1

def find_next_block(file,block_name):
    block = []
    while True:
        end_of_block = False
        line = find_line_infile(file, '/begin', block_name)
        if not line:
            break
        block.append(line)
        while True:
            line = file.readline()
            block.append(line)
            if find_word_in_line(line, block_name):
                end_of_block = True
                break
        if end_of_block:
            break
    if len(block) == 0:
        return None
    else:
        return block

#def find_next_characteristic(file):
#    block = []
#    while True:
#        end_of_block = False
#        line = find_line_infile(file,"/begin","CHARACTERISTIC")
#        if not line:
#            break
#        block.append(line)
#        while True:
#            line = file.readline()
#            block.append(line)
#            if find_word_in_line(line,"CHARACTERISTIC"):
#                end_of_block = True
#                break
#        if end_of_block:
#            break
#    if len(block) == 0:
#        return None
#    else:
#        return block

# example line :
#      /* Name                   */      TQD_pctAccPdl_MAP_x
#      /* ECU Address            */      0x7001fc94
#      /* Record Layout          */      Lookup2D_FLOAT32_IEEE

#    /begin RECORD_LAYOUT Lookup2D_FLOAT32_IEEE
#      FNC_VALUES  1 FLOAT32_IEEE COLUMN_DIR DIRECT
#    /end   RECORD_LAYOUT

def find_characteristic_info(block,name):
    idx = find_line_index(block,name,"Name")
    if idx < 0:
        return None, None, None
    # get address
    address_idx = find_line_index(block, "ECU", "Address")
    address = ''
    if address_idx > 0:
        # last
        address = block[address_idx].split()[-1]
        # delete '0x' in the front
        address = address[2:]
    # get layout
    record_layout_idx = find_line_index(block, "Record", "Layout")
    record_layout = ''
    if record_layout_idx > 0:
        record_layout = block[record_layout_idx].split()[-1]
    # get dimension
    dim = []
    if find_line_index(block,"AXIS_DESCR") > 0:
        idx = 0
        new_idx = 0
        while new_idx >= 0:
            new_idx = find_line_index(block[idx:], "Number", "Axis")
            if new_idx < 0:
                break
            x = block[idx+new_idx].split()[-1]
            dim.append(int(x))
            idx = idx + new_idx + 1
    else:
        dim.append(1)
    return address, dim, record_layout

def find_layout_info(block):
    name = block[0].split()[-1]
    info_words = block[1].split()
    value_type = info_words[2]
    index_mode = info_words[3]
    return name, value_type, index_mode

def get_init_value(init_json_path, query_name):
    # 1 read example json
    f = open(init_json_path, 'r')
    json_obj = json.load(f)
    f.close()

    # 2 return value if valid value exist
    for item in json_obj['data']:
        name = item['name']
        if query_name == name:
            length = 1
            for d in item['dim']:
                length = length*int(d)
            value = item['value']
            value_length = int(item['value_length'])
            if len(value) == length*value_length*2:
                return value
    return None

g_config = {'channel': 3, 'download_can_id': '630', 'upload_can_id': '631'}

# bytes for each type
g_value_length = {'FLOAT32_IEEE': 4}

if __name__ == '__main__':
    args = parse_arg()
    print('a2l filename : ' + args.a2l)
    print('search parameters : ')
    print(args.parameters)

    f = open(args.a2l,'r')

    layouts = {}
    # get value type table
    while True:
        block = find_next_block(f, 'RECORD_LAYOUT')
        if not block:
            break
        name, value_type, index_mode = find_layout_info(block)
        layouts[name] = {'value_type': value_type, 'index_mode:': index_mode}

    json_obj = {'config': g_config, 'data': []}

    f.seek(0)
    while True:
        block = find_next_block(f, 'CHARACTERISTIC')
        if not block:
            break
        for param in args.parameters:
            address, dim, record_layout = find_characteristic_info(
                block, param)
            if address:
                value_type = layouts[record_layout]['value_type']
                value_length = g_value_length[value_type]
                length = 1
                for i in range(len(dim)):
                    length = length*dim[i]
                value = get_init_value(args.init, param)
                if value != None and 2*length*value_length != len(value):
                    print("init value length does not math")
                    value = None
                if value == None:
                    value = '00'*length*value_length
                item = {'name': param, 'address': address, 'dim': dim,
                        'value_type': value_type, 'value_length': value_length, 'value': value}
                json_obj['data'].append(item)

    json_str = json.dumps(json_obj)

    f.close()

    print(json_str)
    f = open(args.output,'w')
    f.write(json_str)
    f.close()
