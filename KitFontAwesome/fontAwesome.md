### Font-Awesome

#### 调用工具包
#####  在markdown中接口

** 方法1 **

在markdown中任意位置放置如下代码：

```html

<head> 
    <script defer src="https://use.fontawesome.com/releases/v5.0.13/js/all.js"></script> 
    <script defer src="https://use.fontawesome.com/releases/v5.0.13/js/v4-shims.js"></script> 
</head> 
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.0.13/css/all.css">

```

其中最后一行是表示此文档需要导入 Font Awesome 最新版本 5.0.13（截止至 2018.06.11）的图标符号，
前面的四行是用于将 Font Awesome 4.x 版本的语句转化为 5.0.13 版本。这是因为在 2017 年年底发布的 5.0 版本中，
对 4.x 版本的大量图标符号的名称进行了重写，加上此四行就可以同时使用 4.x 和 5.x 版本的语句。

** 方法2 ** (<i class="fa fa-thumbs-up" aria-hidden="true">推荐</i>）

或者如下代码，来源不同
```html
<link rel="stylesheet" href="https://cdn.bootcss.com/font-awesome/4.7.0/css/font-awesome.css">
```

** 方法3 ** (未成功)

或者如下代码,加载自己下载好关于font-awesome的工具包：

```html
<link rel="stylesheet" href="path/to/font-awesome/css/font-awesome.min.css">
```

不影响markdown的正常阅读，一般将其放在末尾。

##### 在html中调用
 - 使用 CSS
 - 复制 `font-awesome` 目录到你的项目中
 - 在`<head>`处加载`font-awesome.min.css`如下。

```html

<link rel="stylesheet" href="path/to/font-awesome/css/font-awesome.min.css">
```

 - 查看 [案例][4] 以获取 Font Awesome 的使用方法。

练习文件 practice1.html[^3]

#### 工具包介绍
- 名称 `font-awesome`
- 本机下载的安装包的位置[^1]
- 常用文件`font-awesome.min.css`路径[^2]

#### 插入图标

之后就可以直接插入各类 Font Awesome 符号了，其基础用法是：

    <i class="fa fa-weixin" aria-hidden="true"></i>

或者更简单，(aria-hidden 只是[辅助阅读功能][1])

    <i class="fa fa-weixin"></i>

例如：

<i class="fa fa-weixin"></i>
<i class="fa fa-spinner fa-spin fa-fw"></i>
<i class="fa fa-home fa-fw" aria-hidden="true"></i>


#### 相关链接
- 巧用 Font Awesome 装点 Markdown 文档 https://neo.sspai.com/post/45217
- 两步-在项目中使用fontawesome字体图标 https://www.imooc.com/article/25157
- [aria-hidden][1] https://zhuanlan.zhihu.com/p/75211551
- font-awesome [图标库][2] <i class="fa fa-thumbs-up" aria-hidden="true"></i>
- font-awesome[官网][3] <i class="fa fa-thumbs-up" aria-hidden="true"></i>


---

*[aria-hidden]: 查阅资料的专业解释：现代的辅助技术能够识别并朗读由 CSS 生成的内容和特定的 Unicode 字符。
为了避免屏幕识读设备抓取非故意的和可能产生混淆的输出内容（尤其是当图标纯粹作为装饰用途时），
我们为这些图标设置了 aria-hidden=“true” 属性。 通俗点说就是为屏幕识读设备过滤无关信息。

[^1]: font-awesome的工具包路径F:\Python520\Tutorial\KitFontAwesome\font-awesome

[^2]:font-awesome/css/font-awesome.min.css

[^3]: practice1.html

[1]: https://zhuanlan.zhihu.com/p/75211551

[2]: http://www.fontawesome.com.cn/faicons/

[3]: http://www.fontawesome.com.cn

[4]: http://www.fontawesome.com.cn/examples/

<link rel="stylesheet" href="https://cdn.bootcss.com/font-awesome/4.7.0/css/font-awesome.css">
