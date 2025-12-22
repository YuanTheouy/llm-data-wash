llm-data-wash/
├── assets/                  # 存放演示图片或静态资源
├── configs/                 # 配置文件模板
│   ├── basic_clean.yaml     # 基础清洗配置
│   └── regen_vllm.yaml      # vLLM重生成配置
├── llm_data_wash/           # 核心源码包
│   ├── __init__.py
│   ├── core/                # 核心抽象基类
│   │   ├── __init__.py
│   │   ├── base_loader.py   # 数据加载基类
│   │   ├── base_filter.py   # 数据清洗基类
│   │   └── base_generator.py# 数据生成基类
│   ├── loaders/             # 数据加载模块
│   │   ├── __init__.py
│   │   ├── local_loader.py  # 本地 JSONL/Parquet 加载
│   │   └── hf_loader.py     # HuggingFace Datasets 加载
│   ├── filters/             # 数据清洗/过滤模块
│   │   ├── __init__.py
│   │   ├── basic_filters.py # 长度、正则、重复清洗
│   │   ├── quality_filter.py# 基于困惑度(PPL)或奖励模型(RM)的清洗
│   │   └── dedup.py         # MinHashLSH 去重（可选）
│   ├── generators/          # 数据生成/增强模块
│   │   ├── __init__.py
│   │   └── vllm_generator.py# 基于 vLLM 的重生成（封装你之前的代码）
│   └── utils/               # 工具模块
│       ├── __init__.py
│       ├── gpu_monitor.py   # GPU 监控 (pynvml)
│       ├── logger.py        # 统一日志处理
│       └── concurrency.py   # 线程池/进程池封装
├── scripts/                 # 执行脚本（CLI 入口）
│   ├── download_data.py
│   ├── run_pipeline.py      # 组装 Loader -> Filter -> Generator
│   └── start_vllm_server.sh # 辅助 shell 脚本
├── tests/                   # 单元测试
│   ├── test_filters.py
│   └── test_generators.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py                 # 或 pyproject.toml