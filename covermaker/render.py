import os

import cv2
import numpy
from PIL import ImageDraw, Image, ImageOps, ImageFont
from covermaker.layout import layout_text
from utils import draw_box_txt_fine
import numpy as np


# 当前页渲染类，包含绘制文本，绘制box和保存的功能
class Render(object):
    def __init__(self, img_obj):
        self._img = Image.fromarray(cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB))  # 存储Image实例，属性为私有
        self._img_draw = ImageDraw.Draw(self._img)  # 存储draw实例，私有

    def draw(self, text, text_conf_section, show_text_box=False):
        '''根据配置绘制文本

        Args:
            text (str): 待绘制的文本
            text_conf_section (Section object): 要绘制的文本的配置项
            show_text_box (bool, optional): 是否显示文本框的区域。文本框将绘制成红色
        '''
        # 绘制文本
        self._draw_text(text, text_conf_section)
        # 绘制文本框
        show_text_box and self._draw_text_box(text_conf_section.box)  # python语句  bool and 函数体(arg*) 若bool为真，则执行函数体内容
        return cv2.cvtColor(numpy.asarray(self._img), cv2.COLOR_RGB2BGR)  # 返回CV图像

    # 实际绘制文本的逻辑
    def _draw_text(self, text, section):
        if not text:
            return
        # 对文本进行排版, 如果设置的字体大小小于5则启用大小适应
        if section.font_size <= 5:
            print('嵌字模式1')
            if len(text.splitlines()) > 1:  # 按照'\n'经行分行，如何识别需要分行未定义，如果行数大于1
                # 绘制文本分为两种情况：
                # 情况一：需要缩小大小的多行文本(包含'\n')，逐行绘制，字体大小无法控制,且只能对换行进行缩小，无法进行缩小
                # 情况二： 单行文本，能利用该bbox实现完全自适应大小字体。
                if section.dir == 'h':  # 如果是水平文本
                    font_size = int(
                        section.box.h / len(text.splitlines()) / (1 + section.line_spacing_factor))  # 字体大小为BOX宽度/行数/行间距
                    # ypos= int(font_size * section.line_spacing_factor + section.box.lt[1]) # 的字体的开始位置，与顶行有距离
                    # xpos= int(section.box.w / 2 + section.box.lt[0])
                    ypos = int(font_size * section.line_spacing_factor + section.box.lt[1])  # 的字体的开始位置，与顶行有距离
                    xpos = int(section.box.lt[0])
                    font_file = os.path.join(os.path.dirname(__file__), 'fonts', section.font)  # 获取字体文件的相对目录
                    font = ImageFont.truetype(font_file, font_size)  # 创建font实例
                    # 逐行绘制
                    for c in text.splitlines():
                        self._img_draw.text(xy=(xpos, ypos),
                                            text=c,
                                            fill=section.color,
                                            font=font,
                                            anchor='lt',
                                            align='left',
                                            stroke_width=section.stroke_width,
                                            stroke_fill=section.stroke_fill)
                        ypos += int(font_size * (1 + section.line_spacing_factor))
                else:  # 如果是垂直文本
                    # 对文本进行排版
                    font_size = int(section.box.w / len(text.splitlines()) / (1 + section.line_spacing_factor))
                    ypos = int(section.box.lt[1])
                    xpos = int(section.box.rb[0] - font_size * (1 + section.line_spacing_factor) / 2)
                    font_file = os.path.join(os.path.dirname(__file__), 'fonts', section.font)
                    font = ImageFont.truetype(font_file, font_size)
                    # 逐列绘制
                    for c in text.splitlines():
                        self._img_draw.text(xy=(xpos, ypos),
                                            text=c,
                                            direction='ttb',  # 经行垂直绘制
                                            fill=section.color,
                                            font=font,
                                            anchor='lt',
                                            align='left',
                                            stroke_width=section.stroke_width,
                                            stroke_fill=section.stroke_fill)
                        xpos -= int(font_size * (1 + section.line_spacing_factor))

            else:  # 如果只有一行情况
                # 未手动设置排版， 返回最大能利用该box区域的字体大小size,大小也会自由缩小，和分行后的line[]
                layout = layout_text(text, section)
                # 绘制
                for c, pos, section.degree, size in layout.iter_letters():  # 遍历每一个列每一个字，获取他们的排版信息用于绘制
                    self._img_draw.text(pos,
                                        c,
                                        fill=section.color,
                                        font=layout.font,
                                        stroke_width=section.stroke_width,
                                        stroke_fill=section.stroke_fill)
        else:
            # 此处进行特定字体的绘制,返回的是整张绘制好的图片
            print('嵌字模式2')
            img_cv = cv2.cvtColor(np.array(self._img), cv2.COLOR_RGB2BGR) # 注意此处所填的位置应该是涂白的
            img_cv_text = draw_box_txt_fine((self._img.width, self._img.height), text, './covermaker/fonts/'+section.font,
                                            text_size_ocr=section.font_size,
                                            is_multiline=True,
                                            multi_box=(section.box.lt[0],section.box.lt[1],section.box.w,section.box.h))
            img_cv = cv2.bitwise_and(img_cv,img_cv_text)
            self._img = Image.fromarray(img_cv)

    def _draw_text_box(self, box):
        self._img_draw.rectangle((box.lt, box.rb), outline='#ff0000', width=2)

    def save(self, file_path):
        self._img.save(file_path)
