# -*- coding: utf-8 -*-
"""
@Date  : 2021/6/24
@Desc  : 
"""
import os

packages = [
    "numpy",
    "matplotlib",
    # "sklearn",
    # "requests",
    # "pyinstaller",
    # "django",
    # "pyqt5",
    # "pandas",
    # "pygame",
]

mirrorUrls = [
    r"https://pypi.tuna.tsinghua.edu.cn/simple/",  # 清华大学
    r"http://pypi.tuna.tsinghua.edu.cn/simple/",  # 清华
    r"https://pypi.mirrors.ustc.edu.cn/simple/",  # 中国科学技术大学
    r"https://mirrors.aliyun.com/pypi/simple/",  # 阿里云
]


def autoInstallPackages(**kwargs):
    """
    Auto Install Packages
    autoInstallPackages(Isurl[,urlNum[,]])
    :param kwargs:Isurl、urlNum
    :return:None
    """
    Isurl = kwargs.pop("Isurl", True)
    urlNum = kwargs.pop("urlNum", 0)
    for package in packages:
        if Isurl:
            execute = "pip install " + package + " -i " + mirrorUrls[urlNum]
        else:
            execute = "pip install " + package
        try:
            os.system(execute)
            print("Successful")
        except:
            print("Failed Somehow")


if __name__ == '__main__':
    autoInstallPackages()
    # autoInstallPackages(Isurl=False)
