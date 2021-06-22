# 检测面部标志。
# 旋转、缩放和平移第二个图像以适应第一个图像。
# 调整第二张图像中的色彩平衡以匹配第一张图像的色彩平衡。
# 将第二个图像的特征混合在第一个图像的顶部

import cv2
import dlib
import numpy

import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))           #脸部轮廓
MOUTH_POINTS = list(range(48, 61))          #嘴部轮廓
RIGHT_BROW_POINTS = list(range(17, 22))     #右眉毛
LEFT_BROW_POINTS = list(range(22, 27))      #左眉毛
RIGHT_EYE_POINTS = list(range(36, 42))      #右眼
LEFT_EYE_POINTS = list(range(42, 48))       #左眼
NOSE_POINTS = list(range(27, 35))           #鼻子
JAW_POINTS = list(range(0, 17))             #颚部

# 用于排列图像的点
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# 点从第二个图像覆盖在第一个。
# 元素将被覆盖。
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# 在颜色校正期间使用的模糊量，作为瞳孔距离。
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

# 该函数get_landmarks()以 numpy 数组的形式获取图像，并返回一个 68x2 元素矩阵，其每一行对应于输入图像中特定特征点的 x、y 坐标。
# 特征提取器 ( predictor) 需要一个粗略的边界框作为算法的输入。这是由传统的人脸检测器 ( detector) 提供的，它返回一个矩形列表，每个矩形对应于图像中的一个人脸。
# 要制作预测器，需要预先训练的模型。这样的模型可以 从 dlib sourceforge 存储库下载。
def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

# 遮罩用于选择图像 2 的哪些部分和图像 1 的哪些部分应显示在最终图像中
# get_face_mask()定义了一个例程来为图像和地标矩阵生成掩码。
# 它绘制了两个白色的凸多边形：一个围绕眼睛区域，一个围绕鼻子和嘴巴区域。
# 然后将蒙版的边缘向外羽化 11 个像素。羽化有助于隐藏任何剩余的不连续性。
# 为两个图像生成这样的面罩.将第二个掩码转换为图像 1 的坐标空间。
# 然后通过采用逐元素最大值将掩码合并为一个。结合两个掩码可确保图像 1 中的特征被覆盖，而图像 2 中的特征则显示出来。
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

# 将输入矩阵转换为浮点数。这是接下来的操作所必需的。
# 从每个点集减去质心。一旦为结果点集找到最佳缩放和旋转，质心c1 和c2可用于找到完整的解决方案。
# 同样，将每个点集除以其标准差。这消除了问题的缩放部分。
# 使用奇异值分解计算旋转部分。
# 以仿射变换矩阵的形式返回完整的变换。  
def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # 所求的R实际上是U*Vt所给R的转置
    # 要求矩阵位于左侧（使用列向量）。
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

# 将结果插入 OpenCV 的cv2.warpAffine函数以将第二个图像映射到第一个：
def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

# 两幅图像之间的肤色和光照差异导致重叠区域边缘出现不连续性
# 对第二张图片进行色彩纠正
# 此函数尝试更改 的颜色im2以匹配的颜色im1。它通过除以im2的高斯模糊im2，然后乘以 的高斯模糊来实现im1。
# 这里的想法是RGB 缩放颜色校正，但不是整个图像的恒定比例因子，每个像素都有自己的局部比例因子。
# 通过这种方法，可以在一定程度上解释两个图像之间的光照差异。
# 例如，如果图像 1 从一侧点亮，但图像 2 具有均匀的照明，则颜色校正后的图像 2 在未点亮的一侧也会显得更暗。
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # 防止除零错误.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) / im2_blur.astype(numpy.float64))

def main(pra1,im2,landmarks2): 
    im1, landmarks1 = read_im_and_landmarks(pra1)
    if (len(landmarks1)==0):
        print("no face or many faces")
        return im1,1
    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
     
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)
     
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
     
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    return output_im,0

    
def ex_run(img_path1, img_path2, out_path):
    im1, landmarks1 = read_im_and_landmarks(img_path1)
    im2, landmarks2 = read_im_and_landmarks(img_path2)

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                               landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                          axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    cv2.imwrite(out_path + '/output.jpg', output_im)