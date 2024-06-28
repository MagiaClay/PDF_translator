import math
import cv2
from PIL import ImageFont, Image, ImageDraw
import numpy as np
import datetime
import os
import glob
import fitz
import PyPDF2
import requests
import random
import json
from hashlib import md5

# 防止字符串乱码
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'


def compute_iou(gt_box, b_box):
    '''
    计算交并比
    :param gt_box: box = [x0,y0,x1,y1] （x0,y0)为左上角的坐标（x1,y1）为右下角的坐标
    :param b_box:
    :return: 
    '''
    width0 = gt_box[2] - gt_box[0]
    height0 = gt_box[3] - gt_box[1]
    width1 = b_box[2] - b_box[0]
    height1 = b_box[3] - b_box[1]
    max_x = max(gt_box[2], b_box[2])
    min_x = min(gt_box[0], b_box[0])
    width = width0 + width1 - (max_x - min_x)
    max_y = max(gt_box[3], b_box[3])
    min_y = min(gt_box[1], b_box[1])
    height = height0 + height1 - (max_y - min_y)
    if width < 0 or height < 0:
        interArea = 0
    else:
        interArea = width * height
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    iou = interArea / boxAArea
    # iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


# 绘制字体函数
def draw_box_txt_fine(img_size, txt, font_path="./fonts/simfang.ttf", is_multiline=True, box=None,
                      multi_box=None, text_size_ocr=25):  # multi_box(width,height)
    def create_font(txt, sz, font_path="./fonts/simfang.ttf", text_size_ocr=25, is_multiline=False):  # 正常【width,height】
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
                txt = temp[:int(sz[0]) // font_size]  # 此处有问题，需要返回被省略的标准函数
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
        import string
        count_zh = count_pu = count_punc = 0
        s_len = len(s)
        en_dg_count = 0
        for c in s:
            if c in string.ascii_letters or c.isdigit() or c.isspace():  # 此处没有计算标点符号
                en_dg_count += 1
            elif c.isalpha():
                count_zh += 1
            elif c in string.punctuation:
                count_punc += 1
            else:
                count_pu += 1
        # 返回当前句子长度，以及标点符号。
        return s_len - math.ceil((en_dg_count + count_punc) * 0.5), math.ceil((en_dg_count + count_punc) * 0.5)

    if box:
        box_height = int(
            math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))  # height
        box_width = int(
            math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))  # width
    else:
        box_height = multi_box[3]
        box_width = multi_box[2]

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
                                                               font_path, is_multiline=is_multiline,
                                                               text_size_ocr=text_size_ocr)
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


# 填充图片
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


# 图片转PDF
def get_merged_pdf(imgs_path):
    # 将图片转换为PDF文件
    def convert_pic_pdf(img_path, pdf_path):
        for img in sorted(glob.glob(img_path)):
            doc = fitz.open()
            imgdoc = fitz.open(img)
            pdfbytes = imgdoc.convert_to_pdf()
            imgpdf = fitz.open("pdf", pdfbytes)
            doc.insert_pdf(imgpdf)
            if not os.path.exists(pdf_path):
                doc.save(pdf_path)

    save_pdf_path = os.path.join(imgs_path, 'pdf')
    img_names_list = os.listdir(imgs_path)
    pic_name_list = ['.jpg', '.png', '.bmp', '.jpeg', '.JPG', '.PNG', '.JPEG']
    img_n_list = [name for name in img_names_list if os.path.splitext(name)[-1] in pic_name_list]
    for img_n in img_n_list:
        img_p = os.path.join(imgs_path, img_n)
        pdf_n = os.path.splitext(img_n)[0] + '.pdf'
        pdf_p = os.path.join(save_pdf_path, pdf_n)
        if not os.path.exists(save_pdf_path):
            os.makedirs(save_pdf_path)
        convert_pic_pdf(img_p, pdf_p)

    # 将pdf合并成一个pdf
    pdf_names_list = os.listdir(save_pdf_path)
    pdf_names_list.sort(key=lambda x: int(x[7:-4]))  # 经行排序排序(image_xx.pdf)
    pdf_writer = PyPDF2.PdfWriter()
    for pdf_n in pdf_names_list:
        pdf_p = os.path.join(save_pdf_path, pdf_n)
        # print(pdf_n)
        pdf_obj = open(pdf_p, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_obj)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])
    combine_pdf_path = os.path.join(imgs_path, 'all_pics.pdf')
    pdf_output_file = open(combine_pdf_path, 'wb')
    pdf_writer.write(pdf_output_file)
    pdf_output_file.close()
    return combine_pdf_path


# PDF转图片
def pyMuPDF_fitz(pdfPath, imagePath):
    startTime_pdf2img = datetime.datetime.now()  # 开始时间
    # print("imagePath=" + imagePath)
    pdfDoc = fitz.open(pdfPath)
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
        # 此处若是不做设置，默认图片大小为：792X612, dpi=96
        zoom_x = 1.33333333  # (1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if not os.path.exists(imagePath):  # 判断存放图片的文件夹是否存在
            os.makedirs(imagePath)  # 若图片文件夹不存在就创建

        pix.pil_save(imagePath + '/' + 'images_%s.png' % pg)  # 将图片写入指定的文件夹内

    endTime_pdf2img = datetime.datetime.now()  # 结束时间
    # print('pdf2img时间=', (endTime_pdf2img - startTime_pdf2img).seconds)


def del_dir(dir):  # 注意要加上'/'
    if not os.path.exists(dir):
        return False
    if os.path.isfile(dir):
        os.remove(dir)
        return
    for i in os.listdir(dir):
        t = os.path.join(dir, i)
        if os.path.isdir(t):
            del_dir(t)  # 重新调用次方法
        else:
            os.unlink(t)
        # os.rmdir(dir)  # 递归删除目录下面的空文件夹


# 百度翻译
def baidu_translator(text='', from_lang='en', to_lang='zh'):
    # Set your own appid/appkey.
    appid = '20240628002087171'
    appkey = 'uN9S5RZGcBZFjgCmmiYg'

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = from_lang
    to_lang = to_lang
    query = text

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()  # json file
    # print(result['trans_result'])
    # for p in result['trans_result']:
    #     print(str(p['dst']))
    return result['trans_result']


if __name__ == "__main__":
    # 1、PDF地址
    pdfPath = 'D:/testPics/原文档.pdf'
    # 2、需要储存图片的目录
    imagePath = 'D:/testPics/PDFtoPIC'  # 不用加/
    pyMuPDF_fitz(pdfPath, imagePath)

    # 将图像保存为PDF, 保存路径为源目录
    get_merged_pdf(imagePath)
