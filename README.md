# PaperTrail · 带证据引用的论文阅读助手

上传文本型 PDF，按所选论文进行中英文检索、证据问答和跨论文对照。每条回答保留原文、文件页码、章节与文本块坐标，点击引用可以查看高亮原页。

这不是仅有上传和聊天的封装：项目包含独立解析进程、双栏排序、章节上下文索引、BM25 + E5 混合检索、pgvector 存储、引用校验和可重复评测。

## 已部署与实际验证

- 服务器目录：`/home/xf/practice/paper-reader`。
- 平台：服务器回环地址 `127.0.0.1:18100`，通过 SSH 转发访问。
- PostgreSQL 16.15 + pgvector 0.6.0，独立端口 15440；Python 3.12.3。
- CPU 上实际运行 `intfloat/multilingual-e5-small` 的 384 维量化 ONNX 嵌入。
- 本地 Ollama `qwen2.5:1.5b-instruct` 实际完成选证，返回原文引用。
- 16 项服务器自动化测试通过。
- 两篇明确标注为合成材料的 PDF、10 道手工编写问题：分别评测 keyword、vector、hybrid。最新原始数据见 `reports/evaluation.json`。小规模合成集成绩不能代表真实论文整体准确率。
- 真实论文《Attention Is All You Need》已导入，中文和英文数据集问题均定位至文件第 7 页；证据见 `reports/real-paper-check.json`。这只是两个端到端样例，不是真实论文评测集。

Docker Compose 配置已提供，但服务器账户无 Docker/sudo 权限，当前部署使用独立用户进程。没有运行 Docker 验收或线上 CI，也未录制演示视频。

## 直接使用服务器版

在 VS Code 已连接服务器后，打开 `/home/xf/practice/paper-reader`，在“端口”面板转发 **18100**。也可以在本机终端运行：

```bash
ssh -N -L 18100:127.0.0.1:18100 xf@192.168.9.100
```

打开 <http://127.0.0.1:18100>。本次演示已经建立临时隧道；如果端口已被占用，不要再重复建立。隧道断开后，应用仍在服务器运行，可以重新转发。

使用流程：上传 PDF → 勾选论文 → 输入问题 → 点击“查找证据” → 点击回答的页码 → 核对右侧高亮原文。选择 2–4 篇论文后可以生成数据集、指标、方法与局限的证据对照。

默认提供两篇合成测试论文和一篇公开真实论文。合成材料中的数据集和实验数值是虚构的，只供测试。

## 服务器管理

```bash
cd /home/xf/practice/paper-reader
.venv/bin/python scripts/server.py status
.venv/bin/python scripts/server.py start
.venv/bin/python scripts/server.py test
.venv/bin/python scripts/server.py evaluate
# 停止应用及本项目数据库，不删除文件：
.venv/bin/python scripts/server.py stop
```

`start` 启动独立数据库和应用。服务器重启后需手动执行，不包含系统自启动或进程守护。日志在 `runtime/app.log`、`runtime/postgres.log`；文件在 `data/pdfs`；数据库在 `runtime/pgdata`。随机生成的数据库凭据在权限 0600 的 `runtime/server-env.json`，不包含 SSH 密码，不提交 Git。

切换嵌入模型需要重建向量，不能混用不同模型：

```bash
# 保证数据库已启动，在暂停新上传/查询时执行
.venv/bin/python scripts/reindex_vectors.py
```

重建保留文本块 ID 和页码，已有引用定位不受影响。这个脚本不会重新解析 PDF，解析算法变化需要另行迁移文本块。

## 本地开发与复现

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn paper.api:create_app --factory --host 127.0.0.1 --port 18100
```

无环境变量时默认使用 SQLite 和词项哈希向量，界面会明确显示“非语义模型”。它便于离线开发，不等同于服务器的 E5 语义配置。

测试：`python -m pytest -q`。HTTP 评测：服务启动后运行 `python scripts/evaluate.py`。默认 PDF fixtures 已包含，也可通过 `python scripts/make_fixtures.py` 重新生成。

语义模型复现：

```bash
pip install -r requirements-semantic.txt
python scripts/download_model.py
# Linux/macOS 环境变量；Windows 可通过 $env: 设置同名变量
export EMBEDDING_MODEL=e5-onnx-int8-v1
export E5_MODEL_DIR="$PWD/runtime/e5"
export OLLAMA_URL=http://127.0.0.1:11434
export LLM_MODEL=qwen2.5:1.5b-instruct
```

模型下载使用固定仓库 revision 和 SHA-256 清单，仅下载官方 E5 文件。服务器已完成下载并校验；模型本体不放入 Git。若已有 hash 索引，不能仅改环境变量直接复用；重新导入新库或执行向量迁移。

本项目不是通用自由问答：模型选择与问题相关的原文片段，服务器验证引用 ID、论文、页码和逐字匹配，然后展示摘录；没有让模型自由生成不带原文支持的总结。未配置模型或输出无法校验时退回检索摘录，状态中显示来源；找不到可用证据时返回 no_evidence。

## 能力与边界

| 能力 | 实现与限制 |
|---|---|
| PDF 解析 | ≤25 MB、≤80 页、文本型、非加密；独立子进程，超时与 Linux 资源限制；不提供 OCR |
| 阅读顺序 | 按文本块坐标区分双栏和跨栏分隔，重复页眉页脚过滤；复杂图表、公式、多栏仍可能出错 |
| 切分 | 页内、块内切分，最多 1200 字符，200 字符重叠；高亮原始块，不能保证句级精确框 |
| 检索 | BM25 中文二元词/英文词 + E5 语义向量，RRF 排名融合；小库使用 pgvector 精确余弦检索，无 ANN 索引 |
| 引用 | 保存 chunk ID、paper ID、文件页码、原文、bbox；页码从 PDF 第 1 页开始，不等同论文印刷页码 |
| 对比 | 按字段检索原文证据，不自动宣称某论文更好，不把不同数据集的分数直接作优劣比较 |
| 缺证据 | 关键词/语义阈值筛选与模型选证；阈值是启发式，可能误召回或漏召回，仍需查看原文 |
| 数据管理 | SHA-256 去重、任务内论文范围限制、问答历史；当前无多用户登录、文件删除、配额和库清理 |

论文中的指令被当作不可信文本。模型返回的 quote 必须是候选块的真实子串，未知引用会被拒绝。但“引用文本存在”不代表“论证充分”或“正确回答了问题”；项目不把定位校验率称为答案准确率。

## 结构

```text
paper/api.py             上传、检索、问答、对照、引用原页与历史接口
paper/parser.py          文本块、章节、页码、双栏启发式
paper/parse_worker.py    受限解析子进程
paper/retrieval.py       BM25、E5/词项向量、pgvector、RRF
paper/answers.py         本地模型选证、逐字引用校验、降级
paper/db.py              论文、文本块向量、问答记录
paper/static/           中文操作页面
fixtures/               两份合成 PDF 与手工问题集
tests/                  自动化回归
scripts/                部署、重建向量、下载模型和评测
reports/                实际生成的验证结果
model-manifest.json     官方模型版本和文件校验值
```

主要接口：`POST /api/papers`、`POST /api/search`、`POST /api/ask`、`POST /api/compare`、`GET /api/papers/{id}/chunks`、`GET /api/papers/{id}/pages/{page}.png`、`GET /api/history`、`GET /api/evaluations`。完整参数见运行服务的 `/docs`。

技术依据：[PyMuPDF 文本块与坐标](https://pymupdf.readthedocs.io/en/latest/app1.html)、[pgvector](https://github.com/pgvector/pgvector)、[E5 模型](https://huggingface.co/intfloat/multilingual-e5-small)、[真实论文来源](https://arxiv.org/abs/1706.03762)。
