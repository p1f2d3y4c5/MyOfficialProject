import cv2
import numpy as np
import time
def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))
def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p
    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b
def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)
    #知道处理一张图片时间后下面三行可以解除注释状态2023-02-29：赵辑锦
    # cv2.imshow('20190708_Dark',Dark_Channel)    # 查看暗通道
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    #设定出一个m形状的三维立体全0张量
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    #将张量内的值限定在0到1之间，Y：输入的张量，0，最小值阈值，1：最大值阈值
    Y = np.clip(Y, 0, 1)

    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y


if __name__ == '__main__':
    tic = time.time()
    #<原函数
    m = deHaze(cv2.imread(r'E:\newUser2\AppliactionDataLocation\MyOfficialProject\Yolov5Cont'
                          r'ainerDetectionBasedOnImprovedTransformerAndDefoggingMethod\Yolov5_for_Py'
                          r'Torch_v7.0\dataset\snow1.jpg') / 255.0) * 255
    #原函数/>
    #将浮点型阵列转化为uint8阵列
    m = m.astype(np.uint8)
    #<change 9
    b, g, r = cv2.split(m)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    m = cv2.merge([b, g, r])
    toc = time.time()
    print('Using time is', toc - tic)
    #change9 />
    cv2.imwrite(r'E:\newUser2\AppliactionDataLocation\MyOfficialProject\Yol'
                r'ov5ContainerDetectionBasedOnImprovedTransformerAndDe'
                r'foggingMethod\Yolov5_for_PyTorch_v7.0\dcdheResult\dcdhe3.jpg', m)
    print('success')

