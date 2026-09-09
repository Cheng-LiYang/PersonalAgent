# Personal Knowledge Agent

通过一个 Docker 容器运行的个人知识库 Agent，浏览器提供聊天、PDF/Markdown/TXT/DOCX 上传、增量索引、引用、人工审核和模型设置。保留 LangGraph 工作流、混合检索、记忆和运行记录。

## 启动

安装并启动 Docker Desktop（Linux containers），或在 Linux 安装 Docker Engine 与 Compose。在本目录执行：

```sh
docker compose up -d --build
```

打开 http://localhost:8080 。API 文档位于 http://localhost:8080/api/docs 。无需在电脑上安装 Python、Node 或模型。

默认使用轻量 Hash Embedding + BM25 和离线抽取式回答，不会自动下载大模型。要生成完整回答，在网页“模型设置”中填写 OpenAI-compatible 服务地址、模型和 API Key。默认离线模式不等于云端大模型的回答能力，Hash 检索也弱于语义模型。

在“知识库”页上传文件，或将资料放入宿主机 `KnowledgeBase/` 后，对容器路径 `/app/KnowledgeBase` 创建索引。不要填写 Windows 路径。宿主机 Ollama 的接口地址可填写 `http://host.docker.internal:11434/v1`，并确保 Ollama 允许容器访问。

## 配置与数据

可将 `.env.example` 复制为 `.env` 后修改端口等变量；已有 `.env` 时直接编辑，避免覆盖密钥。修改后执行 `docker compose up -d --build`。

- `config/docker.yaml`：运行配置；修改后执行 `docker compose restart`。
- `KnowledgeBase/`：挂载的资料目录，文件不会进入镜像。
- `agent-data`：Compose 管理的数据卷，保存索引、模型设置、对话与运行记录。
- `model-cache`：可选本地模型下载缓存。

原桌面版 `data/` 保留作为旧数据，不直接挂入 Linux 容器，以免 Windows 路径、旧向量维度和 DPAPI 密钥导致故障。首次 Docker 使用需重新配置模型并为原资料建立索引。Linux 中保存的 API Key 仅做 Base64 编码，应保护数据卷访问权限。

```sh
docker compose ps
docker compose logs -f --tail=100
docker compose down
```

普通 `down` 保留数据卷；不要使用 `down -v`，除非明确要删除容器数据。备份时先停止服务，再备份知识库目录及数据卷。

## 本地语义模型（可选）

默认镜像移除了 PyTorch 等大型依赖。需要 BGE 时，在 `.env` 设置 `INSTALL_LOCAL_MODELS=true`，并将 `config/docker.yaml` 的 `embedding.provider` 改成 `local`，然后重新构建。第一次使用会下载配置中的模型，需要联网、更多磁盘和内存。更换 Embedding 后需要重建索引；建议使用新的数据卷，避免复用不同模型的旧向量。本地模型扩展不属于默认构建验证范围。

## 分享给其他用户

分享本目录的 `backend/`、`web/`、`config/`、`Dockerfile`、`docker-compose.yml`、`requirements.txt`、`.dockerignore`、`.env.example` 和本说明，其他用户执行同一启动命令即可。不要分享自己的 `.env`、`data/` 或知识库内容；知识库目录缺失时 Compose 会创建。

也可分发已构建的镜像：

```sh
docker save -o personal-knowledge-agent.tar personal-knowledge-agent:latest
```

其他用户导入后可直接启动，无需源码：

```sh
docker load -i personal-knowledge-agent.tar
docker run -d --name personal-agent --restart unless-stopped --init -p 127.0.0.1:8080:8000 -v personal-agent-data:/app/data -v personal-agent-kb:/app/KnowledgeBase personal-knowledge-agent:latest
```

打开 http://localhost:8080 并上传自己的资料。镜像包含默认配置和网页资源。

默认只允许本机访问。需要可信局域网访问时，将 `.env` 中 `BIND_HOST` 改成 `0.0.0.0`，访问 `http://服务器IP:8080`。当前应用没有账号隔离和登录鉴权，一个实例的知识库及模型设置共享；不要直接暴露到公网。不同用户建议分别运行自己的容器。

## 精简后的源码

```text
backend/             Agent、检索、文档处理、API 与网页服务入口
web/                 Vue 源码与锁定的前端依赖（构建时使用）
config/docker.yaml   Docker 默认配置
Dockerfile           前端构建 + Python 运行镜像
docker-compose.yml   单容器启动与持久化
requirements.txt     Python 运行依赖
.env.example         环境变量模板
.dockerignore        构建白名单
README.md            部署与使用说明
```

桌面客户端、EXE 构建产物、开发虚拟环境、测试与评测源码、旧设计文档和旧部署配置已删除。运行时依赖的 Harness 预算、策略与工作流模块仍保留。`tmp/` 内部分旧测试缓存因 Windows 文件权限限制尚未删除，不会进入 Docker 镜像，也不参与运行。
