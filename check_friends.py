"""
该文件被main.py文件调用，用于进行友军保护。
"""
import numpy as np

class check_friends():
    """
    description: 检测友军
    """
    def __init__(self, ser, color, run_mode, engine_version):
        """
        description: 初始化变量以及尝试获取友军信息。
        param:
            ser:    串口信息。
        """
        self.color = color
        self.friends = []
        self.check_fr = 0
        self.RUN_MODE = run_mode
        self.ENGINE_VERSION = engine_version


        # 尝试10次通讯获取，防止1次获取失败。
        for i in range(10):
            if (self.check_fr):
                break
            self.get_color_and_friends(ser)

    def get_color_and_friends(self, ser):
        """
        description: 与电控通讯，获取友军颜色与友军id。
        param:
            ser:    串口信息。
        """
        try:
            ser.write(b'\x45')
        except:
            print("Wrong Open Serial!")
        
        # TODO:还需要添加风车的编号，已经打过的风车和灰色风车都标记为友军
        # TODO:目前的想法：1.红蓝色已击打看作一个标签进行训练识别   2.红蓝色已击打分为两类训练识别

        if self.RUN_MODE:
            print(f"Recieve: {ser.read()}")
        # 根据我方红蓝方的设定，进行友军识别
        if ser.read() == b'\xff' or self.color == 1:
            self.color = 1  # red
            self.friends = [0, 3, 6, 9, 12, 15] if self.ENGINE_VERSION == 7 else [0, 1, 2, 3]
        elif ser.read() == b'\xaa' or self.color == 2:
            self.color = 2  # blue
            self.friends = [1, 4, 7, 10, 13, 16] if self.ENGINE_VERSION == 7 else [4, 5, 6, 7]
        if self.RUN_MODE:
            print(f"Friend id: {self.friends}") if self.friends else print("No friend id!")

        # 如果是友军而且友军列表成功添加，那么友军标记边变1，并且友军列表添加死亡的敌人
        fr = []
        if self.check_fr == 0 and self.friends:
            fr = self.friends
            self.check_fr = 1
        self.friends_list = fr + ([2, 5, 8, 11, 14, 17, 19, 20, 21] if self.ENGINE_VERSION == 7 else [8, 9, 10, 11])

    def get_nonfriend_from_all(self, all, friends):
        """
        description: 获取非友军相关参数。
        param:
            all:    全部参数。
            friend: 友军参数。
        return:
            非友军参数。
        """
        new = []
        for i in all.numpy().tolist():
            if i not in (friends):
                new.append(i)
        return np.array(new)

    def get_enemy_info(self, result_boxes):
        """
        description: 处理识别的box，输出敌军的box信息。
        param:
            result_boxes:    boxes类。
        return:
            只含敌军的boxes类。
        """
        # 分别代表友军的box、box置信度、box的id
        exit_friends_boxes = []
        exit_friends_scores = []
        exit_friends_id = []
        friends_id = []
        for ii in range(len(result_boxes.classid)):
            if int(result_boxes.classid.numpy()[ii]) in self.friends_list:
                friends_id.append(int(result_boxes.classid.numpy()[ii]))
                exit_friends_boxes.append(result_boxes.boxes[ii])
                exit_friends_scores.append(result_boxes.scores[ii])
                exit_friends_id.append(result_boxes.classid[ii])
        if self.RUN_MODE:
            print(f"Friend Id: {friends_id}") if friends_id else print("No friend id!")
        enemy_list_index = []

        # 获取敌军的列表以及id
        try:
            for i in result_boxes.classid.numpy():
                if int(i) not in friends_id:
                    dex_tem = ((np.where(result_boxes.classid.numpy() == i))[0][0])
                    enemy_list_index.append(dex_tem)
        except:
            "g"

        if self.RUN_MODE:
            print(f"Enemy Id: {enemy_list_index}") if enemy_list_index else print("No enemy id!")

        ourbox = []
        for dex in enemy_list_index:
            ourbox.append(result_boxes.boxes[dex].numpy())

        result_boxes.boxes = ourbox
        result_boxes.scores = self.get_nonfriend_from_all(result_boxes.scores, exit_friends_scores)  # 置信度处理
        result_boxes.classid = self.get_nonfriend_from_all(result_boxes.classid, exit_friends_id)    # id处理

        if self.RUN_MODE:
            print(f"Nowboxes: {result_boxes.boxes}")
            print(f"Nowscore: {result_boxes.scores}")
            print(f"Nowid: {result_boxes.classid}")
        return result_boxes