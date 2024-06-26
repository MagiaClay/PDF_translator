import math
import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from paddleocr import PaddleOCR
from utils import pyMuPDF_fitz, get_merged_pdf,del_dir

from multiprocessing import freeze_support
import translators as ts
import cv2
import layoutparser as lp
import pandas as pd

# 定义全局变量
image_width = 0  # 待处理图片宽度
image_height = 0  # 待处理图片高度
text_size_ocr = 16  # 标准字体大小
origin_text_list = []
translated_text_list = []


class ocr_box:
    txt = ''
    box = None

    #  return y0, y1, x0, x1  # 返回一个标准化的bbox
    def __init__(self, box, txt):
        self.box = box
        self.txt = txt


class layout_box:
    txt = ''
    ocr_box_list = []
    ocr_box_list_size = 0
    Bbox = None

    def __init__(self, box, list):
        self.Bbox = box
        self.ocr_box_list = list

    # 添加OCR box到res下
    def append_ocr_box(self, box):
        self.ocr_box_list.append(box)
        self.txt = self.txt + ' ' + box.txt
        self.ocr_box_list_size += 1
        return self.ocr_box_list_size

    def is_true_text(self):
        global image_width
        if image_width > 0:
            max_margin = image_width / 5
        else:
            max_margin = 50
        x_list = []  # 获取每一个list的X0
        if self.ocr_box_list_size >= 2:  # 条件一：必须有两个box及以上
            for idx, b in enumerate(self.ocr_box_list):  # 获取每一个b的(x0)
                x_list.append(b.box[0])  # 实际[x0,y0,x1，y1]坐标系要对应，获得x[0]坐标
            for x in x_list:
                for y in x_list:
                    # 此处对相对距离进行量化，如果Layout内每一句OCRbox开始坐标点相差过大，则该text不是一个自然段的
                    if abs(x - y) >= max_margin:  # 像素大小需要根据文档长度进行量化
                        return False  # 条件二：X轴值相差超过
        else:
            return False  # 如果只包含一个文本则直接返回false
        return True


# 判断box1 是否在box2内,用质心判断
def is_in(box1, box2):  # 传入的参数(x0,y0,x1,y1)
    xa0, ya0, xa1, ya1 = box1
    xb0, yb0, xb1, yb1 = box2
    # (a,b)为box1 的质心
    a = (xa1 - xa0) / 2 + xa0
    b = (ya1 - ya0) / 2 + ya0
    # print((a,b))
    # print(box2)
    # 质心在box2围城的区域之中
    if (a >= xb0 and a <= xb1) and (b >= yb0 and b <= yb1):
        return True
    else:
        return False


# 计算图像中标准字体大小,只针对
def average_text_size(ocr_list):
    size_list = []
    temp_size = 0
    global text_size_ocr
    for ocr in ocr_list:
        height = int(abs(ocr.box[3] - ocr.box[1]))
        font_size = round(height * 0.85)
        # 对字体大小进行量化
        font_size = font_size - (font_size % 3)  # 3倍量化
        if 10 < font_size <= 20:
            font_size = 15
        elif 20 < font_size <= 36:
            font_size = 25
        size_list.append(font_size)
    for size in size_list:
        temp_size += size
    size_count = np.bincount(size_list)
    # 跟新全局变量
    text_size_ocr = np.argmax(size_count) + 1


def draw_ocr_box_txt_one_pic(image,
                             boxes,
                             txts,
                             text_blocks=None,
                             scores=None,
                             drop_score=0.8,  # 准确度门限
                             font_path="./doc/simfang.ttf"):
    # 全局变量更新
    global image_width, image_height
    image_height = image.height
    image_width = image_width
    h, w = image.height, image.width
    img_left = image.copy()
    # print("该图像目前1/4宽为："+str(w/6))
    # 新建一张白色的图
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255  # 初始化一个同样大小的图片
    random.seed(0)

    # ImageDraw模块提供了图像对象的简单2D绘制。用户可以使用这个模块创建新的图像，注释或润饰已存在图像，为web应用实时产生各种图形。 Draw()创建一个可以在给定图像上绘图的对象。
    # img_left_array = np.array(img_left)  # 分支一
    img1_left_cv2 = cv2.cvtColor(np.array(img_left), cv2.COLOR_RGB2BGR)
    draw_left = ImageDraw.Draw(img_left)  # 直接进行绘画

    bbox_codin = [b.coordinates for b in text_blocks]  # 取出了所有bbox
    # print(bbox_codin)
    layout_list = []  # 传入的参数(x0,y0,x1,y1)
    ocr_list = []  # 用于存储需要进行绘制的OCR
    # 初始化layout_list
    for layout in bbox_codin:
        ocr_save_list = []  # 防止List驻留导致的地址相同恶行Bug
        layout_instance = layout_box(layout, ocr_save_list)
        layout_list.append(layout_instance)
    # 进行一次便利，将OCR_list 和 Layout_list 进行初始化，进行标准字体确定和margin确定
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:  # 过滤准确度低的文本信息
            continue
        y0, y1, x0, x1 = rectangle_normalization_bbox(box)  #  return y0, y1, x0, x1  # 返回一个标准化的bbox
        ocr_box_instance = ocr_box((y0, x0, y1, x1), txt)  # 创建一个实例【y0,x0,y1,x1】->实际[x0,y0,x1，y1]坐标系要对应
        ocr_list.append(ocr_box_instance)  # 将实例0入列表
        for layout in layout_list:
            if is_in(ocr_box_instance.box, layout.Bbox):
                # 如果该OCRBOX在Layoutbox内,加入到子类
                layout.append_ocr_box(ocr_box_instance)  # 一次匹配成功后所有的layout都添加了该类？？
                ocr_list.pop(-1)  # 弹出该ocrbox实例，此处正确
    # 统计字体大小
    average_text_size(ocr_list)
    #选取需要的Layout_list_temp，用于处理自然段文本需要
    layout_list_temp = []  # 用与存取正确的layout
    for layout in layout_list:
        if layout.is_true_text():  # 该函数是自然段
            # print("是自然段文本，且内部OCR长度为：" + str(layout.ocr_box_list_size))
            layout_list_temp.append(layout)
        else:
            # 需要在Ocrlist里面push其内部的OCR
            for ocr in layout.ocr_box_list:
                ocr_list.append(ocr)
        # 得到的layout_list_temp[] 用于文本段的绘制 和 ocr_list，用于剩下需要单独绘制的OCR
    # 填涂背景与OCR的书写
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:  # 过滤准确度低的文本信息
            continue
        # 填充白色低（但想根据其背景进行识别），需要获取原图片，再把原图片经行填充
        # TODO
        # 绘图的时候将bbox转为对应的标准矩形
        box_reg = rectangle_normalization_bbox(box)  # X0
        # 截取该矩形图片
        img1_left_cv2_temp = img1_left_cv2[box_reg[2]:box_reg[3], box_reg[0]:box_reg[1]]  # 注意此处是【w,h】

        blue, green, red = bincount_1(img1_left_cv2_temp)
        draw_left.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            fill=(red, green, blue),  # (red,green,blue)
            outline=None)

        # 当前ocr的[x0,y0]是否在单句list里面
        for ocr in ocr_list:
            # 如果在，则正常绘制
            if (ocr.box[0] == box_reg[0]) and (ocr.box[1] == box_reg[2]):
                # 填字的时候仍然使用的是曲边box
                img_right_text = draw_box_txt_fine((w, h), txt, font_path, 'cn', box=box)  # 返回的是一个个包含文本信息的图片，整张图片的相对位置
                # 此处进行对修改后的副本经行文字融合
                img_right = cv2.bitwise_and(img_right, img_right_text)  # 右图位图迭代合并拼接
    # 此处单独利用layout_list[]绘制段落
    for layout in layout_list_temp:
        width = layout.Bbox[2] - layout.Bbox[0]
        height = layout.Bbox[3] - layout.Bbox[1]
        img_right_text = draw_box_txt_fine((w, h), layout.txt, font_path, 'cn', is_multiline=True,
                                           multi_box=(
                                               layout.Bbox[0], layout.Bbox[1], width,
                                               height))  # 返回的是一个个包含文本信息的图片，整张图片的相对位置
        img_right = cv2.bitwise_and(img_right, img_right_text)  # 右图位图迭代合并拼接

    img_right = Image.fromarray(img_right)
    con_enhancer = ImageEnhance.Contrast(img_right)
    img_right = con_enhancer.enhance(1.1)  # 对比度增强
    # img = Image.fromarray(cv2.cvtColor(img1_left_cv2, cv2.COLOR_BGR2RGB))
    img = img_left
    # 新建一张图，设置RGB色彩(255, 255, 255)，设置宽度和高度，这里设置2倍的宽是为了粘贴2张图像
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    # 把图像粘贴在img_show上
    img_show.paste(img, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))

    # # 图片合并
    box_1 = (0, 0, w, h)
    box_2 = (w, 0, 2 * w, h)
    img_left = img_show.crop(box_1)
    img_right = img_show.crop(box_2)
    img_merge = Image.blend(img_left, img_right, 0.5)
    con_enhancer = ImageEnhance.Contrast(img_merge)
    img_merge = con_enhancer.enhance(2.5)  # 对比度增强
    # 此处进行图像缩小

    img_merge = cv2.cvtColor(np.asarray(img_merge), cv2.COLOR_RGB2BGR)
    scale_num = 0.5 # 缩小系数为.0.5
    dist_size = (int(scale_num * img_merge.shape[0]), int(scale_num * img_merge.shape[1]))
    cv_image = resize_keep_aspectratio(img_merge, dist_size)  # 对文件进行等缩小
    img_merge = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    return img_merge  # 返回Image对象
    # return img_show


def bincount_1(pic_path):
    img = pic_path

    blue_list = []
    green_list = []
    red_list = []

    for x in range(img.shape[0]):  # 图片的高
        for y in range(img.shape[1]):  # 图片的宽
            px = img[x, y]  # 获得坐标
            blue_list.append(img[x, y][0])
            green_list.append(img[x, y][1])
            red_list.append(img[x, y][2])
            # print(px)  # 这样就能得到每个点的bgr值
    blue_count = np.bincount(blue_list)
    green_count = np.bincount(green_list)
    red_count = np.bincount(red_list)

    blue = np.argmax(blue_count) + 1
    green = np.argmax(green_count) + 1
    red = np.argmax(red_count) + 1

    return blue, green, red


def rectangle_normalization_bbox(box):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))  # 高
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))  # 长
    y0 = int(box[0][0])
    y1 = int(box[0][0] + box_width)
    x0 = int(box[0][1])
    x1 = int(box[0][1] + box_height)

    return y0, y1, x0, x1  # 返回一个标准化的bbox


def draw_box_txt_fine(img_size, txt, font_path="./fonts/simfang.ttf", lang='en', is_multiline=False, box=None,
                      multi_box=None):  # multi_box(width,height)
    if box:
        box_height = int(
            math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))  # height
        box_width = int(
            math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))  # width
    else:
        box_height = multi_box[3]
        box_width = multi_box[2]

    # 进行文本翻译
    if (lang == 'cn'):
        if txt:
            txt = address_font(txt)

    if box_height > 2 * box_width and box_height > 30:  # 非常长的文本框
        img_text = Image.new('RGB', (box_height, box_width), (255, 255, 255))  # 白色文本
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            # 对中文字体经行缩小 如果需要输出英文文档需要重新更改
            font, sz, space_txt, is_overflow = create_font(txt, (box_height, box_width), font_path)
            draw_text.multiline_text([0, 0], space_txt, fill=(0, 0, 0), font=font)  # 从(0,0）开始绘制
        img_text = img_text.transpose(Image.ROTATE_270)  # 进行图像旋转270°？
    else:
        if is_multiline:
            img_text = Image.new('RGB', (box_width, box_height), (255, 255, 255))
            draw_text = ImageDraw.Draw(img_text)
            if txt:
                font, sz, space_txt, is_overflow = create_font(txt, (box_width, box_height),
                                                               font_path, is_multiline=is_multiline)
                draw_text.multiline_text([0, 0], space_txt, fill=(0, 0, 0), font=font)  # 从(0,0）开始绘制
        else:
            img_text = Image.new('RGB', (box_width, box_height), (255, 255, 255))
            draw_text = ImageDraw.Draw(img_text)
            if txt:
                font, sz, space_txt, is_overflow = create_font(txt, (box_width, box_height), font_path)
                if is_overflow:
                    img_text = Image.new('RGB', (sz[0], sz[1]), (255, 255, 255))  # 重新绘图
                    draw_text = ImageDraw.Draw(img_text)
                    draw_text.multiline_text([0, 0], space_txt, fill=(0, 0, 0), font=font)  # 从(0,0）开始绘制
                else:
                    draw_text.multiline_text([0, 0], space_txt, fill=(0, 0, 0), font=font)  # 从(0,0）开始绘制

    # 以下使用CV对图像进行透视变换
    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])  # 源标准位置box
    if box:
        pts2 = np.float32(
            [box[0], [box_width + box[0][0], box[0][1]], [box_width + box[0][0], box_height + box[0][1]],
             [box[0][0], box_height + box[0][1]]])  # 源标准位置box
    else:
        pts2 = np.float32(
            [[multi_box[0], multi_box[1]], [box_width + multi_box[0], multi_box[1]],
             [box_width + multi_box[0], box_height + multi_box[1]],
             [multi_box[0], box_height + multi_box[1]]])  # 源标准位置box
    # pts2 = np.array(box, dtype=np.float32)  # box的bix

    M = cv2.getPerspectiveTransform(pts1, pts2)  # 透视变换的变换矩阵

    img_text = np.array(img_text, dtype=np.uint8)  # 图像
    img_right_text = cv2.warpPerspective(  # 图像的透视变化
        img_text,
        M,
        img_size,  # 需要输出的图像大小
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))

    return img_right_text  # 图片是一个整张大小图片

# 此处进行翻译限制
def address_font(txt):
    freeze_support()
    # 限制文本
    for index, word in enumerate(origin_text_list):
        if txt == word:
            txt = translated_text_list[index]
            return txt
    # 纯数字字符
    is_num = True
    for c in txt:
        if c in string.ascii_letters:
            is_num = False
            break
    if is_num:
        txt = txt
    else:
        txt = ts.translate_text(txt, to_language='cn',translator='baidu')  # 段落翻译，并不是整体翻译

    return txt


def create_font(txt, sz, font_path="./fonts/simfang.ttf", is_multiline=False):  # 正常【width,height】
    # font_size = int(sz[1] * 0.85)  # 获得宽，中文文本可在此处调节
    is_overflow = False  # 是否越界
    if is_multiline:
        font_size = text_size_ocr  # 暂定正文文本为25
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        length = font.getlength(txt)
    else:
        font_size = round(sz[1] * 0.85)
        # 对字体大小进行量化
        font_size = font_size - (font_size % 3)  # 3倍量化
        if 10 < font_size < 36:
            font_size = text_size_ocr
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        length = font.getlength(txt)
    first_line = True
    # 存储当前句子
    space_txt = ''

    if not is_multiline and length >= sz[0]:  # 如果只有一行,且改行不是文本段洛
        font_size = int(font_size * sz[0] / length)  # 得到一个相对大小的比例（长*款/字长） # 出现字体非常非常小的情况时候 一般只适用于英文文本
        sz = [int(sz[0] * 1.5), int(sz[1])]  # [height, width]
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        is_overflow = True
        space_txt += txt
        return font, sz, space_txt, is_overflow  # 多返回值没问题

    if is_multiline:
        while str_count(txt)[0] >= int(sz[0]) // font_size:

            # length = font.getlength(txt)
            # a = str_count(txt)
            # b = int(sz[0]) // font_size
            temp = txt
            txt = temp[:int(sz[0]) // font_size] # 此处有问题，需要返回被省略的标准函数
            alpha_temp = str_count(txt)[1]
            # 此处重新调整换行判断
            if alpha_temp > 2:
                idx = (int(sz[0]) // font_size) + alpha_temp
                txt = temp[:idx]
                space_txt += txt + '\n'
                txt = temp[idx:]
                continue
            else:
                txt = txt

            # 重新统计txt句子里的标点符号，并经行重新阶段
            if first_line:
                new_txt = '' + txt  # 首行缩进
                first_line = False
            else:
                new_txt = txt  # 在此处确认是否要缩进
            space_txt += new_txt + '\n'
            txt = temp[int(sz[0]) // font_size:]
    space_txt += txt

    return font, sz, space_txt, is_overflow  # 多返回值没问题


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu =count_punc = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace(): # 此处没有计算标点符号
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        elif c in string.punctuation:
            count_punc += 1
        else:
            count_pu += 1
    # 返回当前句子长度，以及标点符号。
    return s_len - math.ceil((en_dg_count+count_punc) * 0.5), math.ceil((en_dg_count+count_punc) * 0.5)


# 等比例缩放算法，效果>resize的双三次插值
def resize_keep_aspectratio(image_src, dst_size):
    src_h, src_w = image_src.shape[:2]
    # print(src_h, src_w)
    dst_h, dst_w = dst_size

    # 判断应该按哪个边做等比缩放
    h = dst_w * (float(src_h) / src_w)  # 按照ｗ做等比缩放
    w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))

    h_, w_ = image_dst.shape[:2]
    # print(h_, w_)

    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_ + 1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_ + 1) / 2)

    value = [0, 0, 0]
    borderType = cv2.BORDER_CONSTANT
    # print(top, down, left, right)
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)

    return image_dst


if __name__ == '__main__':
    freeze_support()  # 异步线程可能会报RUntime Error
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`

    ocr = PaddleOCR(use_angle_cls=True, lang="en",
                    use_gpu=False, show_log=False)  # need to run only once to download and load model into memory
    model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                          threshold=0.4,
                                          label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                                          enforce_cpu=True,
                                          enable_mkldnn=True)
    # translator restrict
    origin_text_list = []
    translated_text_list = []
    data = pd.read_csv('D:/testPics/temp/translated.csv')
    for index, row in data.iterrows():
        row_Data_1 = row.iloc[0]
        row_Data_2 = row.iloc[1]
        origin_text_list.append(row_Data_1)
        translated_text_list.append(row_Data_2)

    # DATA
    file_path = 'D:/testPics/PDFtoPIC/'  # f翻译目录源文件,记得加'/'
    file_path_1 = 'D:/testPics/PDFtoPIC'
    file_resize_path = 'D:/testPics/OCR_translated_resize/'
    save_folder = 'D:/testPics/OCR_translated'
    font_path = './fonts/simfang.ttf'  # PaddleOCR下提供字体包
    pdfPath = 'D:/testPics/原文档.pdf' # 唯一需要改动的路径。只要保持D盘文

    # PDF转图片
    del_dir(file_path)
    pyMuPDF_fitz(pdfPath, file_path_1)

    pictures = os.listdir(path=file_path)
    pictures_resize = os.listdir(path=file_resize_path)
    if len(pictures_resize) == 0 :  # 如果未处理过则处理
        for pic in pictures:
            if pic.endswith('.jpg') or pic.endswith('.png'):
                dic_path = file_path + pic
                dic_path_resize = file_resize_path + pic
                # 经行双三次插值对图像进行放大
                cv_image = cv2.imread(dic_path, 1)
                scale_num = 2
                dist_size = (2 * cv_image.shape[0], 2 * cv_image.shape[1])
                cv_image = resize_keep_aspectratio(cv_image, dist_size)  # 对文件进行等比放大
                image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                image.save(dic_path_resize)  # 同名保存
        pictures_resize = os.listdir(path=file_resize_path)
    else:
        for pic in pictures_resize:
            dic_path_resize = file_resize_path + pic
            os.remove(dic_path_resize)
        for pic in pictures:
            if pic.endswith('.jpg') or pic.endswith('.png'):
                dic_path = file_path + pic
                dic_path_resize = file_resize_path + pic
                # 经行双三次插值对图像进行放大
                cv_image = cv2.imread(dic_path, 1)
                scale_num = 2
                dist_size = (2 * cv_image.shape[0], 2 * cv_image.shape[1])
                cv_image = resize_keep_aspectratio(cv_image, dist_size)  # 对文件进行等比放大
                image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                image.save(dic_path_resize)  # 同名保存
        pictures_resize = os.listdir(path=file_resize_path)
    del_dir(save_folder+'/')
    print('预处理完成。')
    for pic in pictures_resize:
        dic_path = file_resize_path + pic

        image = cv2.imread(dic_path)
        image = image[..., ::-1]
        # 检测
        layout = model.detect(image)  # 输出的结构如下图 需要获得的是bbox

        # 接上面代码
        # 首先过滤特定文本类型的区域
        text_blocks = lp.Layout([b for b in layout if
                                 b.type == 'Text'])  # 这种写法从中间开始度，从layout中取出b，如果b == 'Text'，则取出list[b],
        # 之后利用Layout函数，把其b中text——blocks取出

        image = Image.open(dic_path).convert('RGB')  # 打开图片 Image 格式
        result = ocr.ocr(dic_path, cls=True)
        # 获取result字典数据
        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        im_show = draw_ocr_box_txt_one_pic(image, boxes, txts, text_blocks, scores, drop_score=0.85,
                                           font_path=font_path)
        # im_show.show()
        # 保存
        path = save_folder + '/' + pic
        im_show.save(path)
        print(path + '处理完成！')
    # 图片转PDF
    get_merged_pdf(save_folder)


