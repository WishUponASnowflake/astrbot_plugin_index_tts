# astrbot_plugin_index_tts

# 前言

本插件是基于index-tts对AstrBot的语音转文字(TTS)补充

建议使用 `Conda` 构建本插件的虚拟环境

## 性能要求

建议显卡显存大于4Gib
>[!TIP]
>建议`AMD`用户使用`WSL`或`Linux`以使用`ROCm版torch`

## 环境配置

根据[Index TTS 官方文档](https://github.com/index-tts/index-tts)配置环境

...配置虚拟环境...

```bash
cd /path/to/the/plugin
cd index-tts
pip install -e .
```

也可使用本插件包含的requirements.txt进行安装

```bash
cd /path/to/the/plugin
pip install -r requirements.txt
```

如果发生版本冲突，以报错为准去下载合适版本的库

>[!TIP]
>若使用 `uv` 配置环境，则需使用
>```bash
>uv pip install xxx
>```
>但仍推荐使用 `Conda` 作为虚拟环境

>[!WARNING]
>本插件目前仅在python3.10上测试，如果为Python >=`3.11` && <=`3.12`，可能不稳定

## `AstrBot` 内配置

添加`OpenAI`的tts适配器

apikey默认`1145141919810`

超时建议`30`~`60`s

其他默认便可

## 插件自定义

插件目前可自定义：apikey、音频输入文件

音频输入文件在`path/to/the/plugin/sounds`下，自定义时仅用填文件全名即可

如需修改端口号,可在 `main.py` 和 `service.py` 中 找到 
`port = xxx`
字样，更改即可，但注意不要更改 `port` 的 `数据类型`

# 使用

## 正常使用

在聊天界面输入
```cli
/tts
```
>[!NOTE]
>此处 `/` 为你的自定义唤醒词，会根据你的自定义而改变

## 作为 `LLM Tool` 调用 

建议在提示词之中加入对于使用`LLM Tool`(A.K.A. `Function Calling`) 的提示

其余便不用管了

# 后记

时间原因，文档写的较潦草，如有不懂或报错，请issue

如果有想法，也欢迎issue

如果你认为这个插件对你有帮助，请star

# 版本历史

`1.0.0` -> `1.0.1`:

    (1)添加了LLM工具
    (2)使用了线程池进行语音生成
    (3)优化逻辑

# TODO

添加GitHub代理

添加对vllm版本的快捷使用

？？？

