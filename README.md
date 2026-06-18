# 赛题简介

本仓库是挑战杯人工智能赛道初赛的 baseline 仓库。赛题围绕 Intern-S 系列模型的数学智能体设计与推理创新展开，选手需要设计一个能够解决数学问题的推理智能体。智能体接收一道数学题文本，结合题目元信息进行推理，并输出最终答案。

选手的核心任务是实现：

```python
agent.solve(problem: str, metadata: dict) -> dict
```

平台会在正式评测时读取实际测试题，调用选手提交代码中的该入口函数，取得 `final_response`，并结合官方 judger 与标准答案进行判分。

本赛题鼓励选手探索不同的智能体设计方式，包括但不限于：

- 提示词设计与多轮推理
- 多候选生成、验证与选择
- 规划、反思、纠错、答案格式化
- 工具调用、检索、记忆或其它推理策略
- 面向数学题的专门解析、符号计算或后处理

最终评分主要依据 `final_response` 的答案正确性。`trace` 可用于记录智能体的推理与交互过程，便于异常排查、结果展示和同分情况下的设计质量参考。

## Baseline 仓库说明

本仓库提供一个 naive agent baseline，作用是帮助选手理解：

- 赛题内容与本地调试方式
- 输入数据格式
- 智能体入口函数
- 智能体输出格式
- 平台 runner 与选手代码之间的调用关系

Baseline 只是一个最小可运行示例，不限制选手的具体实现。选手可以保留、修改或替换其中的智能体逻辑，也可以新增模块、工具和依赖。但正式提交的仓库必须满足平台约定的入口规范。

### 必须包含的文件

选手提交的仓库根目录必须包含：

```text
user_agent.py
```

平台 runner 会从该文件中加载选手实现。其它文件可以根据需要添加，例如：

```text
requirements.txt
tools/
prompts/
utils/
```

但所有新增文件都应使用相对路径读取，不要依赖本地机器上的绝对路径。

### `user_agent.py` 必须遵守的规范

`user_agent.py` 中必须提供类：

```python
class ReasoningAgent:
    ...
```

平台会使用官方 client 初始化该类：

```python
from user_agent import ReasoningAgent

agent = ReasoningAgent(client=official_client)
```

因此 `ReasoningAgent` 至少需要支持如下构造方式：

```python
def __init__(self, client, *args, **kwargs):
    ...
```

其中 `client` 由平台提供。选手不要在代码中写死 API key，也不要假设本地存在固定的 API 配置文件。正式评测时，模型访问、限流、token 统计、超时控制等由官方 client 和平台 runner 统一管理。

评测时平台提供的 client 与 baseline 中 `llm_client.py` 的 `InternChatClient` 结构一致，可作为本地实现参考。选手可以参考其中的 `chat(messages, temperature, max_tokens)` 调用方式组织模型请求：

```python
response = client.chat(
    messages=[
        {"role": "user", "content": problem},
    ],
    temperature=0.2,
    max_tokens=4096,
)
```

正式评测 client 可能包含额外的资源统计、限流和安全控制逻辑。选手代码只应依赖公开约定的调用接口，不要依赖 client 内部私有字段。

`ReasoningAgent` 必须提供推理函数：

```python
def solve(self, problem: str, metadata: dict) -> dict:
    ...
```

输入参数含义如下：

- `problem`: 数学题题面文本。
- `metadata`: 题目元信息字典，至少可能包含 `idx`。正式评测时，metadata 的具体字段以平台 runner 为准。选手代码不得依赖 `answer`、标准答案或隐藏评测数据。

返回值必须是 `dict`，且必须包含非空字符串字段：

```python
{
  "final_response": "最终答案"
}
```

推荐同时返回 `trace`：

```python
{
  "final_response": "最终答案",
  "trace": [
    {"step": "plan", "content": "..."},
    {"step": "model_call", "content": "..."},
    {"step": "finalize", "content": "..."}
  ]
}
```

字段要求：

- `final_response` 必须是可读的最终答案，不能为空。
- `trace` 可选，建议为列表，记录关键推理步骤、模型调用摘要、候选答案、验证过程等。
- 返回内容必须可以被 JSON 序列化。
- 不要在 `trace` 中写入 API key、访问令牌、个人隐私信息或其它敏感内容。

### 实现自由度

选手可以使用 baseline 中的 lagent 示例，也可以完全不用 lagent。平台只要求 `user_agent.py` 暴露符合规范的 `ReasoningAgent` 和 `solve` 方法。

选手可以新增辅助函数、类和模块，也可以在 `ReasoningAgent` 内部维护状态。但需要注意：

- 不要依赖评测题之间的固定顺序。
- 不要假设多个题目一定在同一个进程中运行。
- 不要依赖本地绝对路径。
- 不要读取或构造隐藏测试集、标准答案或 judger 信息。
- 不要在代码中硬编码 API key。
- 不要输出恶意内容、执行破坏性操作或规避平台资源限制。

## 输入数据与输出样例

本地调试输入为 JSONL 文件，每行是一道题。每行至少包含：

```json
{"idx": 0, "problem": "题目文本"}
```

样例：

```json
{"idx": 0, "problem": "设$\\mathbb{F}_{81}$为$81$元的有限域。$T=\\{\\alpha\\in\\mathbb{F}_{81}|\\mathbb{F}_{81}=\\mathbb{F}_3(\\alpha)\\}$。求$T$中元素的个数。", "answer": "72", "subject": "抽象代数", "source": "sample"}
```

其中 `answer` 只会出现在样例数据中，用于选手本地对照调试。正式评测不会向 `solve` 传入标准答案。

本地 runner 会将每道题的结果保存为独立 JSON 文件：

```text
outputs/
  0.json
  1.json
  2.json
```

成功输出样例：

```json
{
  "idx": 0,
  "status": "success",
  "final_response": "72",
  "trace": [
    {
      "step": "solve",
      "content": "有限域 F_81 是 F_3 上 4 维扩张，生成整个扩张的元素个数为 81 - 9 = 72。"
    }
  ]
}
```

异常输出样例：

```json
{
  "idx": 0,
  "status": "error",
  "final_response": "",
  "error": {
    "type": "RuntimeError",
    "message": "错误信息"
  },
  "trace": []
}
```

本地调试时，如果某个 `idx.json` 已经存在且文件非空，runner 会跳过该题，便于中断后继续运行。

## 本地调试

安装依赖：

```bash
pip install -r requirements.txt
```

设置 API key：

```bash
export INTERN_API_KEY="sk-..."
```

运行样例：

```bash
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs
```

如果只想快速自测，也可以省略输入和输出目录。脚本会自动寻找默认样例数据，并把结果写入一个不会覆盖旧结果的时间戳目录：

```bash
python main.py
```

完整 MathSolve-Agent 批量入口也支持自动规划输出：

```bash
python -m math_prove.main \
  --input data/interns1_math_18domains_504.json \
  --model intern-s2-preview \
  --ablation official_stable
```

如果不传 `--output`，结果会进入 `outputs/runs/<timestamp>_<dataset>_<ablation>/`，其中包含 `results.jsonl`、`results.json`、`logs/` 和 `run_summary.json`。

Prompt 版本批量实验使用目录约定。每个版本放在 `prompts/<variant>/`，可包含以下 5 个文件：

```text
classify.txt
solve.txt
verify.txt
select.txt
extract.txt
```

空文件或缺失文件会回退到默认 prompt。运行多个版本：

```bash
python -m math_prove.run_prompt_sweep \
  --input data/interns1_math_18domains_504.json \
  --variants v1_baseline,v2_strict \
  --model intern-s2-preview \
  --ablation official_stable
```

如果不传 `--output-dir`，结果会进入 `outputs/prompt_sweeps/<timestamp>_prompt_sweep_<dataset>/`，每个版本独立子目录，根目录包含 `sweep_summary.json`。

本地 runner 默认并发数为 8。如需调整：

```bash
export LOCAL_MAX_CONCURRENCY=4
```

本地调试结果只用于选手自测，不代表正式评测分数。正式评测会使用隐藏测试集、官方 client、平台 runner 和官方 judger。

## Intern-S API 使用说明

选手可以使用报名挑战杯时填写的手机号注册书生 API 控制台：

```text
https://internlm.intern-ai.org.cn/api/document
```

API 控制台主要用于：

- 查看 API docs 和可用模型列表。
- 获取、创建或管理 API key。
- 申请更高的 RPM / TPM 流控。
- 查看调用量、配额和使用情况。

书生 API 控制台中的 API 均可免费使用。平台会对普通用户执行流控策略：

```text
RPM 30
TPM 150000
```

如果本地调试或实验需要更高流控，可以前往以下页面申请提升：

```text
https://internlm.intern-ai.org.cn/api/strategy
```

申请时请在备注中填写“挑战杯”。通常情况下，200 RPM 以内的流控提升会在 1-2 个工作日内生效。具体生效时间以 API 控制台状态为准。

选手可以使用 API 控制台下的任意可用模型作为智能体 base model，例如：

- `intern-s1`
- `intern-s1-pro`
- `intern-s2-preview`

本地使用 baseline runner 调试时，可以通过环境变量指定模型：

```bash
export INTERN_MODEL="intern-s2-preview"
```

如需调整 API endpoint，请以书生 API 控制台文档中的接口地址为准配置 `INTERN_API_BASE`。

提交仓库至评分系统时，选手可以选择智能体实际使用的模型。正式评测时，平台 runner 会通过官方 client 使用该模型进行调用。

评分系统 client 使用的 API 来自书生 API 控制台：

```text
https://internlm.intern-ai.org.cn/api
```

建议选手本地调试时也使用同一份 API 控制台，避免本地实验环境和正式评分环境之间产生模型行为、接口格式或网关策略差异。

## 赛事规则与常见问答 (FAQ)

为了帮助参赛队伍更好地理解《挑战杯》报名、组队以及作品提交的相关规则，特整理如下常见问题解答：

### 1. 赛程安排与作品提交时间
*   **组队报名时间**：2026/05/30 - 2026/06/30。参赛选手须登录“挑战杯”官网报名并上传学校盖公章的报名表，完成校级及发榜单位审核，逾期系统将自动关闭。
*   **作品提交时间**：2026/05/30 - 2026/09/04（初赛最晚提交截止时间为 2026-09-04 23:59）。建议至少提前 1 周完成提交，避免临近截止时间集中提交导致队列拥堵或失败。
*   **初赛评审时间**：2026/09/20 之前完成初审，确定入围终审擂台赛的晋级名单。
*   **终审擂台赛时间**：预计在 2026 年 11 月举行。

### 2. 参赛资格与人员限制
*   **已毕业学生能否参赛？**
    *   **已毕业学生不可参赛**。
    *   **2026年全日制毕业班学生**（专科生、本科生、硕士和博士研究生）可以参赛（以 2026 年 6 月 1 日以前正式注册的在校生身份为准）。
*   **国籍与学校限制**
    *   本赛题面向国内全日制非成人教育的普通高等学校在校专科生、本科生、硕士和博士研究生（不含在职研究生）。
    *   **留学生与港澳台地区在校学生可以参赛**，但所有参赛作品的申报主体/参赛主体必须为国内的高等院校或科研院所，不得以境外高校或科研院所名义申报。
*   **跨校/跨地域组队**
    *   可以跨专业、跨学校、跨单位、跨地域组队，但同一团队的所有成员均应符合相关年龄和身份要求。
    *   每件作品只能由 **1 所**高等院校或科研院所作为参赛主体进行申报。
*   **团队规模限制**
    *   每个参赛团队人数不超过 **10 人**。
    *   每件作品的指导教师不超过 **3 人**。

### 3. 评测与排名展示
*   **在线评测系统的实时排名会每天公开吗？**
    *   在线评测系统正式上线后，榜单会**实时更新**。
    *   评测系统上线前，如果通过邮件等方式进行临时评测，不保证按天更新排名。

---

## 提交方式

参赛队伍可以在报名结束后提交作品进行判分，提交开放时间预计为 2026-07-01 起，具体以赛事通知为准。选手需要基于 baseline 仓库准备自己的提交代码，并向判分系统提交仓库地址与 commit SHA。

推荐流程：

1. 获取 baseline 仓库代码。
2. 在自己的仓库中完成实现。
3. 确保仓库根目录包含符合规范的 `user_agent.py`。
4. 将代码推送到 GitHub 或平台指定的代码托管服务。
5. 在判分系统中完成仓库授权。
6. 向判分系统提交仓库地址和 commit SHA。

关于仓库可见性：

- 如果直接 fork 一个公开 baseline 仓库，fork 和提交内容通常也是公开可见的。
- 如果不希望自己的方案公开，建议将 baseline clone 到本地后，推送到自己的 private repository。
- 选手也可以保持仓库公开，但需自行承担方案被其他人看到的风险。

判分系统会以选手提交的 commit SHA 作为评测快照。请务必提交具体 commit SHA，而不是只提交分支名。分支名可能继续变化，commit SHA 才能保证结果可复现。

### 挑战杯官网材料提交

判分系统用于初赛榜单评测，挑战杯官网材料提交用于满足赛事报名与材料归档要求。两者都需要完成。

初赛截止前，请根据挑战杯官网要求，将自己的最终版本代码仓库与赛题要求的其它材料打包成 `.zip` 文件，提交到挑战杯官网，并发送邮件至：

```text
changshuai@pjlab.org.cn
```

建议压缩包内至少包含：

- 最终版本代码。
- `user_agent.py` 及所有运行所需的辅助文件。
- `requirements.txt` 或其它依赖说明。
- 一份说明文件，写明队伍信息、题目名称、最终仓库地址、最终分支名称、最终 commit hash 和选择使用的模型。

提交前建议检查：

- `user_agent.py` 可以被正常 import。
- `ReasoningAgent(client=official_client)` 可以正常初始化。
- `solve(problem, metadata)` 返回包含非空 `final_response` 的字典。
- 所有依赖都已写入 `requirements.txt` 或赛事另行指定的依赖文件。
- 代码中没有硬编码 API key、个人路径、临时文件路径或调试用标准答案。

## 自动判分系统

自动判分系统预计 7 月上线，具体上线日期以赛事通知为准。

判分系统上线前，如果选手需要提交作品并获取评测分数，可以将代码库打包成 `.zip` 文件发送邮件至：

```text
changshuai@pjlab.org.cn
```

邮件模板如下：

```text
邮件主题：[挑战杯初赛评测申请] 队伍名称 - 题目名称 - commit hash

收件人：changshuai@pjlab.org.cn

正文：
队伍名称：

题目名称：基于 Intern-S 系列模型的数学智能体设计与推理创新

报名成员信息（至少填写 1 名挑战杯官网内报名成员）：
- 姓名：
- 学校：
- 报名手机号：

需要评测的代码版本：
- 仓库地址：
- 分支名称：main
- commit hash：
- 选择使用的模型：例如 intern-s2-preview

附件：
- 代码库 zip 文件名：

备注：
- 如有特殊依赖、运行说明或需要说明的异常情况，请在这里填写。
```

邮件评测同样以邮件中写明的分支名称和 commit hash 为准。请确保附件中的代码版本与邮件正文中的 commit hash 对应，避免评测结果无法复现。

判分系统的基本工作流程如下：

1. 选手在系统中提交仓库地址与 commit SHA。
2. 系统检查仓库授权，并拉取对应 commit 的代码。
3. 系统在隔离环境中安装依赖并加载 `user_agent.py`。
4. 系统根据选手提交时选择的模型，使用官方 client 初始化 `ReasoningAgent`。
5. 平台 runner 读取正式评测题，调用 `agent.solve(problem, metadata)`。
6. 系统收集 `final_response`、`trace`、运行耗时、异常信息和资源使用情况。
7. 官方 judger 结合标准答案对结果进行判分。
8. 系统返回最终分数以及是否存在异常。

正式评测时：

- 平台会使用官方闭源测试集。
- 平台会使用来自书生 API 控制台的官方 client，不使用选手自带 API key。
- 平台会统一控制超时、并发、模型调用预算和资源限制。
- 平台不会向选手返回逐题题目、标准答案、judger 细节或逐题详细反馈。
- 如果代码无法安装、无法 import、入口函数不符合规范、运行超时、输出格式错误或触发安全策略，系统会标记异常。

## 初赛评分与晋级

初赛完全采取客观评分方式对所有参赛队伍进行排名。排名依据为初赛截止日期时，判分系统榜单上的队伍最终排名。

在初赛截止日期前，组委会会根据整体赛题完成质量、参赛队伍分数分布等情况，公布进入决赛的最低分数线。进入决赛的队伍数量少于 30 支，具体晋级名单以赛事官方通知为准。

## 提交次数限制

从判分系统上线日到初赛截止时间，每支队伍的提交次数限制为：

- 每天最多提交 2 次。
- 每周最多提交 10 次。

判分系统只会给出最终分数以及是否存在异常。请选手在本地充分调试后再提交正式评测，避免因格式错误、依赖缺失或超时浪费提交次数。

## 常见异常

以下情况可能导致判分异常：

- 仓库授权失败，系统无法拉取提交。
- 提交的是分支名而不是 commit SHA。
- `user_agent.py` 不存在。
- `ReasoningAgent` 类不存在或构造函数不接受 `client`。
- `solve` 方法不存在、签名不符合要求或抛出异常。
- 返回值不是字典。
- `final_response` 缺失、为空或不是字符串。
- 返回值无法 JSON 序列化。
- 依赖无法安装或版本冲突。
- 运行超时或资源使用超过平台限制。
- 代码依赖本地绝对路径、私有文件或未声明资源。
- 代码包含硬编码密钥、恶意行为或规避评测限制的逻辑。

## 实现建议

为了提高提交稳定性，建议选手：

- 保持入口接口简单稳定，把复杂逻辑封装到内部模块。
- 对模型输出做答案抽取和格式化，避免 `final_response` 过长或答案不明确。
- 对异常进行适度处理，保证单题失败不会影响整体运行。
- 控制模型调用次数和 token 使用，避免超时。
- 在本地使用样例数据完整跑通安装、初始化、推理和输出保存流程。
- 保留清晰的 `trace`，但不要记录敏感信息。

本仓库的 baseline 仅用于说明接口和基本思路。正式比赛欢迎选手提交更强、更稳定、更有创造性的数学智能体实现。
