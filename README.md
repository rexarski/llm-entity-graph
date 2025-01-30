# README

Simplified and refactored the codebase of [LYiHub/platform-war-public](https://github.com/LYiHub/platform-war-public).

## Changelog

```fish
uv venv --python 3.11.7
source .venv/bin/activate.fish
uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 datasets ollama pydantic einops
uv pip install -r requirements.txt # remove windows-curses
```

* Remove platform-debate-related codebase
* Install Ollama
* Update `model_kwargs={"device": "cpu"}` in `embedding_model.py`
* Create `data` folder and run `demo_dataset.py` to generate the `results.json`

<!-- ## Structure

API key配置文件是`config.py`。该程序基于moonshot-v1模型实现，需要在配置文件中填入从kimi开放平台申请的API key以正常运行。如果需要更换其他模型服务，需要同时修改`API_BASE_URL`和程序中相应调用大模型的部分（模型名称、特殊参数等）

知识图谱类的主文件是`knowledgeGraph.py`，具体实现主文件接口的组件类分别为`graph_entity.py`，`graph_search.py`，`graph_storage.py`，`graph_visualization.py`。用于RAG的嵌入模型配置文件为`embedding_model.py`。

使用到知识图谱的两个工具类是`knowledge_retriever.py`和`knowledgeGraphExtractor.py`，分别用于信息检索和知识图谱提取。

平台大战对话的主程序是`platform_war.py`，相关UI和Agent类分别为`platform_war_UI.py`和`chat.py`。

## 环境配置

由于向量数据处理部分使用的`faiss-gpu`暂时只支持CUDA加速，所以本项目目前只支持Windows/Linux系统运行。

**创建conda环境**

```
conda create -n platform_war python=3.11.7
conda activate platform_war
```

**启用gpu加速（可选）**

安装电脑显卡版本匹配的CUDA和PyTorch, 例（具体版本请按电脑配置修改）：

```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

然后用以下命令安装FAISS的gpu版本

```
conda install -c conda-forge faiss-gpu
```

**项目目录下安装依赖（必选）**

```
pip install -r requirements.txt
```

如果前一步没有安装CUDA，需要将`embedding_model.py`中的

```
model_kwargs={"device": "cuda"}
```

修改为

```
model_kwargs={"device": "cpu"}
```

**配置API（必选）**

该程序基于moonshot-v1模型实现，需要在配置文件中填入从kimi开放平台申请的API key以正常运行。API key配置文件是`config.py`。

## 图谱提取

`knowledgeGraphExtractor.py`可用于从指定格式的json文件中自动提取知识图谱。

在项目目录下新建data文件夹，放入需要提取的数据文件，命名为`result.json`（或者可以修改`knowledgeGraphExtractor.py`中的路径以使用其他文件名），然后运行`knowledgeGraphExtractor.py`。

`result.json`需要遵循以下json格式：

```
{
  "item_id": {                 // 项目ID作为key
    "title": string,           // 标题
    "clusters": [              // 评论簇数组
      {
        "comments": [          // 评论内容数组
          string,
          ...
        ]
      }
    ]
  },
  "item_id": {
    "title": string,
    "clusters": [
      {
        "comments": [
          string,
          ...
        ]
      }
    ]
  },
  ...
}

```

## 平台辩论

从原始数据提取完知识图谱后，会在项目目录下生成相应的知识图谱数据库。

可以将`platform_war.py`、`platform_war_UI.py`和`chat.py`中的`PLATFORM_NAME`和`PLATFORM_KNOWLEDGE_BASE`修改为对应的数据库名称和路径。

运行`platform_war.py`以开始平台辩论。

## 预提取数据库

如果想复现项目视频中的效果，需要单独下载三个平台的向量数据库。

百度网盘下载链接：

<https://pan.baidu.com/s/1Ki0Sym9dmM76e6ghR6P8jQ?pwd=j3ih> 提取码: j3ih

谷歌云盘下载链接：

<https://drive.google.com/drive/folders/1kaXPSTjVaI1LP9lPtu8XhCnqjgWVlajY?usp=sharing>

使用方法：

解压缩，并将`bilibili_knowledge_base`，`weibo_knowledge_base`，`zhihu_knowledge_base`三个文件放在项目目录下。

运行`platform_war.py`以开始平台辩论。

## 已知问题

* UI缺乏自适应，在不同尺寸的窗口中可能存在显示不全/错位的问题。
* 对话轮次超出屏幕范围后，新对话会与旧对话互相覆盖。

## 参考

微软GraphRAG项目 <https://github.com/microsoft/graphrag>

*** -->
