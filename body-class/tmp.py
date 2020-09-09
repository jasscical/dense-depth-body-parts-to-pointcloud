from PIL import Image
import numpy as np

"""
    记原图为origin，处理以后的结果为res
    origin是512*424，res是250*250
    原本的处理流程大致是这样的：
    1.
        找到origin中人像所在矩形，将人站立的方向缩放到190像素，另一边的长度自适应，保持图片等比例
        颜色数据和透明度数据处理方式略不一样。颜色数据是使用lanczos2缩放
        透明度则使用nearest，然后只保留缩放以后透明度为1（完全不透明）的部分作为一个mask
        颜色会被二值化处理，mask会用来将透明部分设置为全黑色
        （其中，颜色有一些操作我不太确定到底是想干什么，应该和二值化的阈值相关）
    2. 
        现在190*190的图像再加上30像素边界，得到250*250的图片，就是处理结果
    
    我的逆处理过程是这样的：
    1. 找到res中人像所在矩形，记为box1
    2. 找到origin中人像所在矩形，记为box2
    3. 将box1缩放到box2大小
    4. 新建一张和origin一样大小的图片，将box1粘贴到上面，其位置和box2相对于origin的位置一致
    
    有一个可能的问题是，既然原图作了二值化处理，那是不是应该说，比起box1直接缩放到box2的大小
    将box1缩放到二值化以后人像的大小，这样会更加精确（因为二值化以后基本上图像边界会有变动）
    这里的问题是，原本的处理流程中，人像是先缩小再二值化处理的，所以原人像大小没有对应的二值化图片
    也就是说，不管二值化的阈值是怎么确定的，都是在190像素长度的这个图上决定的，没办法对应到原图上
    所以这里只能直接把box1拉伸到box2的大小
"""


def get_figure_box(img):
    """ 计算人像所在区域，其方法和classification_demo.m的一致 """
    source = img.split()

    # 因为图片实际上是单色的，用哪一个颜色都是一样的
    R = np.asarray(source[0])

    # axis=0 把所有元素加到一行上
    margin_1 = R.sum(axis=0)
    margin_2 = R.sum(axis=1)

    tmp = np.nonzero(margin_1)[0]
    left, right = tmp[0], tmp[-1]
    tmp = np.nonzero(margin_2)[0]
    top, bottom = tmp[0], tmp[-1]

    # 800x600 pixel image is written as (0, 0, 800, 600)
    # 即四维坐标是左闭右开的，右下方向的坐标需要加一
    return left, top, right + 1, bottom + 1


def white_to_transparent(img):
    """ img是一个灰度图。返回一个带透明度的PNG，灰度为0部分对应透明部分。"""
    gray = np.asarray(img)
    alpha = np.ones_like(gray) * 255
    alpha[gray == 0] = 0
    rgba = np.empty(img.size + (4,), dtype=np.int8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, 'RGBA')


def run(img):
    root_path = 'E:/TensorFlowProgram/jupyter-notebook/dense-depth-body-parts-master/body-class/' #body-class的path
    img_name = img

    res_img_path = root_path + 'pre_250_250/' + img_name
    res_img = Image.open(res_img_path)
    # 由单色图转化为带透明度的图片
    res_img = white_to_transparent(res_img)

    origin_img_path = root_path + 'input/' + img_name
    origin_img = Image.open(origin_img_path)

    # 将人像部分截出来
    res_figure_box = get_figure_box(res_img)
    res_region = res_img.crop(res_figure_box)

    # 得到原图人像大小
    origin_figure_box = get_figure_box(origin_img)
    origin_figure_box_width = origin_figure_box[2] - origin_figure_box[0]
    origin_figure_box_height = origin_figure_box[3] - origin_figure_box[1]

    # 将处理的后人像大小拉伸回原本的大小
    recovered_figure = res_region.resize((origin_figure_box_width, origin_figure_box_height))

    # 新建一张图片，将处理人像放到原图对应的位置
    recovered_res = Image.new('RGBA', origin_img.size)
    recovered_res.paste(recovered_figure, origin_figure_box)
    recovered_res.save(root_path + '/res_all/' + img_name)
    print(img_name + '   have transfered---------------')




import os


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print('files:', files)  # 当前路径下所有非目录子文件
        return files


if __name__ == '__main__':
    path = r"E:/TensorFlowProgram/jupyter-notebook/dense-depth-body-parts-master/body-class/pre_250_250/" # after matlab body-dense pre-process img
    file_name = file_name(path)
    for img_name in file_name:
        print(img_name + 'start transfer--------------------')
        run(img_name)
