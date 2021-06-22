#  智能换脸软件

## 一、功能说明
已知一幅 A 的人脸图像或人脸视频，新输入一张 B 的人脸图像，将 A 的图像或者视频自动地换成 B 的人脸。

## 二、代码说明
> image_swap.py

### 1. 使用dlib来提取面部标志：
```c
PREDICTOR_PATH = "/home/matt/dlib-18.16/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
```
* 该函数`get_landmarks()`以 numpy 数组的形式获取图像，并返回一个 68x2 元素矩阵，其每一行对应输入图像中特定特征点的 x、y 坐标。

* 特征提取器 (`predictor`) 需要一个粗边界框作为算法的输入。这是由传统的人脸检测器 (`detector`) 提供的，它返回一个矩形列表，每个矩形对应于图像中的一个人脸。

* 要制作预测器，需要预先训练的模型。这样的模型可以 从 dlib sourceforge 存储库下载。

### 2. 使用 procrustes 分析对齐人脸
```c
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
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])
```
* 将输入矩阵转换为浮点数。这是接下来的操作所必需的。
* 从每个点集减去质心。一旦为结果点集找到最佳缩放和旋转，质心 c1  和 c2 可用于找到完整的解决方案。
* 同样，将每个点集除以其标准差。这消除了问题的缩放部分。
* 使用奇异值分解计算旋转部分。
* 以仿射变换矩阵的形式返回完整的变换。

### 3. 对第二张图象进行色彩校正
```c
COLOUR_CORRECT_BLUR_FRAC = 0.6
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += 128 * (im2_blur <= 1.0)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))
```
* 此函数尝试更改 im2 的颜色以匹配 im1 的颜色。它通过除以im2的高斯模糊，然后乘以 im1 的高斯模糊来实现im1。这里的想法是 RGB 缩放颜色校正，但不是整个图像的恒定比例因子，每个像素都有自己的局部比例因子。
* 通过这种方法，可以在一定程度上解释两个图像之间的光照差异。例如，如果图像 1 从一侧点亮，但图像 2 具有均匀的照明，则颜色校正后的图像 2 在未点亮的一侧也会显得更暗。
* 也就是说，这是一个相当粗糙的问题解决方案，合适大小的高斯核是关键。太小，第一张图像中的面部特征将显示在第二张图像中。对于重叠的像素，太大且内核偏离面部区域之外，并且会发生变色。这里使用了 0.6 * 瞳孔距离的内核。

### 4. 将第二张图象的特征混合到第一张图像上
```c
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]
FEATHER_AMOUNT = 11

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

mask = get_face_mask(im2, landmarks2)
warped_mask = warp_im(mask, M, im1.shape)
combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                          axis=0)
```
* get_face_mask()定义了一个例程来为图像和地标矩阵生成掩码。它绘制了两个白色的凸多边形：一个围绕眼睛区域，一个围绕鼻子和嘴巴区域。然后将蒙版的边缘向外羽化 11 个像素。羽化有助于隐藏任何剩余的不连续性。
* 为两个图像生成这样的面罩。使用与步骤 2 中相同的转换，将第二个掩码转换为图像 1 的坐标空间。
* 然后通过采用逐元素最大值将掩码合并为一个。结合两个掩码可确保图像 1 中的特征被覆盖，而图像 2 中的特征则显示出来。
> vedio_swap.py
### 1. 将视频分割成帧
```c
def extract_frames(video_path, dst_folder, index):
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        os._exit(1)
    count = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        #if count % EXTRACT_FREQUENCY == 0:
        save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
        cv2.imwrite(save_path, frame)
        index += 1
        count += 1
    video.release()
    print("Totally save {:d} pics".format(index-1))
    sum_pic=index-1
    return sum_pic
```
* 将 data 文件夹中的视频进行分割，分割后的图片保存到指定文件夹中

### 2. 将转换后的帧合成视频
```c
def picvideo(path,file):
    filelist = os.listdir(path) 
    if len(filelist)==0:
        return 
    height,width,layers=cv2.imread(path+"/"+filelist[0]).shape
    size=(width,height)
    print(size)    
    file_path = file+str(int(time.time())) + ".avi"
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    video = cv2.VideoWriter( file_path, fourcc, OUT_FREQUENCY, size )
    
    for item in range(1,len(filelist)+1):
        item = path + '/' + str(item)+'.jpg'
        #print(item)
        img = cv2.imread(item)  
        video.write(img)      
    video.release() 
```
* 帧率：1秒钟有n张图片写进去。控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次。如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
* picvedio 函数从指定文件夹中抽取出图片并合成出相应视频，由用户指定视频文件的存放位置
## 三、编译运行
> 环境配置
* 环境：Annaconda（win-64，version 4.3.30）
* 语言：Python（version 3.6.5）
* 编辑器：Visual Studio Code
* 关键库：dlib、cv2、numpy

> 文件配置

![1.png](https://pic.rmb.bdstatic.com/bjh/b0d16d61cefda587ebf7e5abd919e700.png)

A1 的人脸图像：1.jpg   
![image.png](https://pic.rmb.bdstatic.com/bjh/227877d6a4e718c50d9c205683b3fd20.png)
A2 的人脸视频：test.mp4    
![image.png](https://pic.rmb.bdstatic.com/bjh/468dfb4f66abb7cfc36af9d75234b8f1.png)
B 的人脸图像：test.jpg    
![image.png](https://pic.rmb.bdstatic.com/bjh/9ee6685e4b51643e3d40662b7f8cd7b8.png)

> 运行结果
* 运行步骤：
  * 将 test 文件夹放在 C 盘根目录下
  * 打开 vscode 进行代码编译
  ```
  PS C:\test> cd .\faceswap\
  PS C:\test\faceswap> activate
  PS C:\test\faceswap> python .\main.py
  ```
* 运行结果
  * 图形界面：  
  ![image.png](https://pic.rmb.bdstatic.com/bjh/a8923b1843868789682aeb6e43d5b9dc.png)
  * 选择测试照片和保存路径，开始替换
  ![image.png](https://pic.rmb.bdstatic.com/bjh/88fff82a81778eb063fffe590db80301.png)
  * 替换完成    
  ![image.png](https://pic.rmb.bdstatic.com/bjh/0dfc8bc3b70820152ac84c57ae68b98c.png)
  * 查看保存路径下的文件      
  A1 与 B 合成：ouput.jpg   
  ![image.png](https://pic.rmb.bdstatic.com/bjh/440d67e8f96453d1c77e2778fa5e25b4.png)
  A2 与 B　合成：avi    
  ![image.png](https://pic.rmb.bdstatic.com/bjh/712d95865ffb128fb8ad7325aa44392c.png)
  
  




