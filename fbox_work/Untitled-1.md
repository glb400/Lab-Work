# 实验室

## 生成对抗集

安装foolbox，使用Python3环境方可运行keras和fbox

环境

+ py3:sys.path.append('/home/wangguangrun/.local/lib/python3.5/site-packages')
+ py:sys.path.append('/home/wangguangrun/anaconda2/lib/python2.7/site-packages')

fbox tutorial:https://foolbox.readthedocs.io/en/latest/user/tutorial.html#visualizing-the-adversarial-examples

流程：

仿照/scripts编写批处理.py
-》
利用fbox生成对抗图片
-》
利用project测试对抗集效果：
https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet

### part1 编写批处理文件

（tmux属于伪终端，因此可以使用tmux运行py文件）

文件 = 导入图片 + fbox处理 + 生成图片路径保存

导入文件就是val文件夹下的image用image导入，该部分：

fbox处理：

生成图片保存：



### part2 利用resnet测试对抗集效果

测试使用这个proj，先看看它的作用：

修改路径导入文件
