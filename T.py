from PyQt6 import QtCore, QtGui, QtWidgets


# 提升table类,该类为图像显示类，目的是控制该Lable下的鼠标移动
# 是否可以利用该类在创建一个对比类
# 此处决定了resize显示的问题

class TLabel(QtWidgets.QLabel):
    xy0 = [0, 0]  # 记录了鼠标初始位置
    xy1 = [0, 0]  # 文本框结束位置
    flag = False
    img_pos = [0, 0, 0, 0]
    flag_switch = False

    # 鼠标点击事件，获取选中点的(x0, y0)原始佐伯安排
    def mousePressEvent(self, event):
        self.flag = True
        self.img_pos = [0, 0, 0, 0]
        self.update()
        if self.flag and self.flag_switch:
            self.xy0[0] = event.position().x()
            self.xy0[1] = event.position().y()

    # 鼠标释放事件，松开鼠标
    def mouseReleaseEvent(self, event):
        self.flag = False

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.flag and self.flag_switch:
            self.xy1[0] = event.position().x()
            self.xy1[1] = event.position().y()
            if self.xy0[0] < self.xy1[0]:
                x = self.xy0[0]
            else:
                x = self.xy1[0]
            if self.xy0[1] < self.xy1[1]:
                y = self.xy0[1]
            else:
                y = self.xy1[1]
            w = abs(self.xy1[0] - self.xy0[0]) # 获取width
            if x + w > self.width(): # 如果绘制超出框
                w = self.width() - x
            h = abs(self.xy1[1] - self.xy0[1])
            if y + h > self.height():
                h = self.height() - y
            if x < 0:
                x = 0
                w = self.xy0[0]
            if y < 0:
                y = 0
                h = self.xy0[1]
            self.img_pos = list(map(int, [x, y, w, h]))
            self.update()

    # 绘制事件，根据该类绘制矩形框
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.flag_switch:
            rect = QtCore.QRect(self.img_pos[0], self.img_pos[1], self.img_pos[2], self.img_pos[3])
        else:
            rect = QtCore.QRect(0, 0, 0, 0)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.red, 2, QtCore.Qt.PenStyle.SolidLine))
        painter.drawRect(rect)
