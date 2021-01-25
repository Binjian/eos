import pygame
 
pygame.init()  # 初始化所有pygame模块
 
# 创建游戏主窗口 480 * 700
screen = pygame.display.set_mode((480, 700))
 
# 创建时钟对象 (可以控制游戏循环频率)
clock = pygame.time.Clock()
 
# 游戏循环 -> 意味着游戏的正式开始！
while True:
 
    # 通过时钟对象指定循环频率
    clock.tick(60)   # 每秒循环60次
 
    # 监听用户事件
    for event in pygame.event.get():
 
        # 判断用户是否点击了关闭按钮
        if event.type == pygame.QUIT:
            print("游戏退出...")
 
            pygame.quit()  # 卸载所有pygame模块
 
            exit()   # 直接终止当前正在执行的Python程序
 
 
    # 游戏代码,绘制图像。。。。。。
 
    pygame.display.update() 
