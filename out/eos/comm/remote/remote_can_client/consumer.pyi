from rocketmq.client import PullConsumer as PullConsumer

def validate_params(json_dic, start_time, need_vin: bool = ...): ...
def command_handle(json_dic: dict): ...
def send_data(data) -> None: ...
def handle_msg(data_dic: dict, code: int): ...
