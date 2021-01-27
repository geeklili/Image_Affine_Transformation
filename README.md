# 介绍
项目是为了完成两个头像的转换而来

类似于胖虎变成吴亦凡

原理是面部识别+仿射变形

# 例子
获取两个图片，注意图片的尺寸必须保证一样   

![](img/girl.png)    

与

![](img/girl2.png)    

结果如下，这是个动态图，如果显示不出来，可以下载img/img_affine/combine_girl.gif查看效果，注：图片来自于网络

![](img/img_affine/combine_girl.gif)    


# 依赖库
dlib, opencv and numpy

注：
- 模型大小为95M，谨慎下载
- 输入图片的大小为200X200
