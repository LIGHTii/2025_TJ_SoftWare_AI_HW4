# 2025_TJ_SoftWare_AI_HW4

## A. 平台搭建

### 1. 注册与登录 ModelScope
访问 [ModelScope](https://www.modelscope.cn/home)，点击右上角进行用户注册。

### 2. 获取计算资源
注册并登录后，确保已绑定阿里云账号并开通了免费云计算资源，启动 CPU 服务器。

### 3. 启动 Notebook 并配置环境

#### 环境配置

##### Conda 环境操作：

```bash
cd /opt/conda/envs
# 若目录不存在：
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda --version
conda create -n qwen_env python=3.10 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen_env
```

##### 非 Conda（root）环境操作：跳过上述，直接安装依赖。

#### 安装依赖项

```bash
pip install -U pip setuptools wheel
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install \
    "intel-extension-for-transformers==1.4.2" \
    "neural-compressor==2.5" \
    "transformers==4.33.3" \
    "modelscope==1.9.5" \
    "pydantic==1.10.13" \
    "sentencepiece" \
    "tiktoken" \
    "einops" \
    "transformers_stream_generator" \
    "uvicorn" \
    "fastapi" \
    "yacs" \
    "setuptools_scm"
pip install fschat --use-pep517
```

##### 可选增强工具：
```bash
pip install tqdm huggingface-hub
```

---

## B. 大模型实践

### 1. 下载模型至本地

```bash
cd /mnt/data
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
```

### 2. 构建推理脚本并运行

```bash
cd /mnt/workspace
```

创建 `run_qwen_cpu.py` 文件：

```python
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

model_name = "/mnt/data/Qwen-7B-Chat"
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
).eval()

inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

运行脚本：

```bash
python run_qwen_cpu.py
```

---

## C.问答测试参考


1.请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少 

2.请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上 

3.他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道 

4.明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？

5.领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。

---
