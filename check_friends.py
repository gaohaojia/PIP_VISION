"""
该文件被main.py文件调用，用于进行友军保护。
"""
import numpy as np

class Boxes():
    def __init__(self, boxes, scores, classid) -> None:
        """
        param:
            boxes:   boxes位置信息。
            scores:  boxes的置信度。
            classid: boxes的id。
        """
        self.boxes = boxes      
        self.scores = scores    
        self.classid = classid  

class check_friends():
    """
    description: 检测友军
    """
    def __init__(self, color) -> None:
        if color == 1:
            friends = [0, 3, 6, 9, 12, 15]
        else:
            friends = [1, 4, 7, 10, 13, 16]
        self.friends_list = friends + [2, 5, 8, 11, 14, 17, 18, 19, 20]

    def get_nonfriend_from_all(self, all, friends) -> np.ndarray:
        """
        description: 获取非友军相关参数。
        param:
            all:    全部参数。
            friend: 友军参数。
        return:
            非友军参数。
        """
        new = []
        for i in all.tolist():
            if i not in (friends):
                new.append(i)
        return np.array(new)

    def get_enemy_info(self, result_boxes) -> Boxes:
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
            if int(result_boxes.classid[ii]) in self.friends_list:
                friends_id.append(int(result_boxes.classid[ii]))
                exit_friends_boxes.append(result_boxes.boxes[ii])
                exit_friends_scores.append(result_boxes.scores[ii])
                exit_friends_id.append(result_boxes.classid[ii])
        enemy_list_index = []

        # 获取敌军的列表以及id
        try:
            for i in result_boxes.classid:
                if int(i) not in friends_id:
                    dex_tem = ((np.where(result_boxes.classid == i))[0][0])
                    enemy_list_index.append(dex_tem)
        except:
            pass

        result_boxes.boxes = [result_boxes.boxes[i].tolist() for i in enemy_list_index]
        result_boxes.scores = self.get_nonfriend_from_all(result_boxes.scores, exit_friends_scores)  # 置信度处理
        result_boxes.classid = self.get_nonfriend_from_all(result_boxes.classid, exit_friends_id)    # id处理
        
        return result_boxes