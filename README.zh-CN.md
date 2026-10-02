# DrugMCTS 工具版

独立于原实验项目的新工具，提供命令行、Python API、MCP 三种入口。输入 SMILES，检索相似分子，执行 MCTS，返回候选靶点、评分、证据和轨迹。默认使用模型服务，无需在工具进程里加载 Qwen 或使用 CUDA。

## 先跑通

```bash
git clone https://github.com/yaoge777/DrugMCTS-tool.git
cd DrugMCTS-tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mcp]'
drugmcts demo --trace --output outputs/demo.json
```

演示无需密钥。RDKit 检索和 MCTS 控制流会实际执行，但模型回答及靶点关联是明确标记的合成数据，不代表科学预测。

## 导入你现有的数据

在本目录下运行：

```bash
drugmcts prepare-legacy --legacy-dir /absolute/path/to/original/DrugMCTS --dataset drugbank --output local-data/drugbank.json
drugmcts prepare-legacy --legacy-dir /absolute/path/to/original/DrugMCTS --dataset kiba --output local-data/kiba.json
```

导入程序只读旧目录，不导入旧 Python 文件，也不会复制密钥、模型权重或实验答案。新生成的 `local-data/` 已加入 Git 忽略和打包排除规则。别人可以按 [知识库格式](docs/knowledge-base.md) 自备 JSON 数据，无需持有你的完整实验项目。

## 使用真实模型

修改 `examples/config.toml` 中的知识库路径、模型名称、服务地址。使用付费或鉴权服务时，把 `api_key_env` 设为 `DRUGMCTS_API_KEY`，由环境变量提供密钥。示例默认连接本机 `http://localhost:8000/v1`，不自带模型服务；配置里的模型名称会实际传给服务。

```bash
drugmcts doctor --config examples/config.toml
drugmcts predict --config examples/config.toml --smiles 'CC(=O)Oc1ccccc1C(=O)O' --top-k 5 --trace --output outputs/prediction.json
```

路径相对于配置文件解析。`doctor` 检查本地配置和数据，不会调用模型。`retrieve` 只检索；`predict` 会将查询结构及相关证据发给配置的模型服务。`top-k` 是使用者指定的数量，工具不会读取 ground truth 来决定返回数量。只有通过评分的靶点才会进入结果，可能少于请求数量。

## Python

```python
from drugmcts import DrugMCTS

tool = DrugMCTS.from_config("examples/config.toml")
result = tool.predict("CC(=O)Oc1ccccc1C(=O)O", top_k=5, include_trace=True)
print(result.model_dump_json(indent=2))
```

无需模型服务的演示改为 `tool = DrugMCTS.demo()`。

## MCP

```bash
drugmcts serve --config /absolute/path/DrugMCTS-tool/examples/config.toml
# 离线演示：drugmcts serve --demo
```

这是 stdio MCP 服务，应由 MCP 客户端启动；提供 `drugmcts_info`、`retrieve_similar_molecules`、`predict_targets`。客户端配置示例见 [英文 README](README.md#mcp)。真实预测可能持续数分钟，客户端超时需要覆盖配置中的搜索时间；服务每次只接受一个预测任务。

## 与论文版本的关系

核心仍然是检索、分子分析、分子筛选、相互作用分析、蛋白选择和双奖励 MCTS。修正了提示词/解析器不一致、重复补选、奖励分母不一致和模型配置无效等问题，并补充最终排名、错误信息及调用预算。

默认采用轻量的 Tanimoto 检索。需要双检索时安装 `.[mcp,hybrid]`，再把 `retrieval.mode` 改成 `hybrid`。新工具对相互作用分析、属性提取、失败处理和排名做了明确的工程选择，详见 [方法说明](docs/method.md)，不能直接宣称复现了论文表格。

评分表示模型选择一致性与 yes 投票比例，不是实测结合力或校准概率。旧数据以 PDB 结构为靶点代表，同时输出 UniProt 映射。原始实验目录未修改。MIT 许可证仅覆盖此工具代码及合成演示数据。
