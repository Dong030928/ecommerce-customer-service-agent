# 售后工作流：LangGraph 原生 Checkpoint 与持久化恢复改造方案

日期：2026-10-05。状态：用户已确认，已完成编码与离线验证。实际实现和测试记录见 [售后 Checkpoint 验证记录](AFTER_SALE_CHECKPOINT_VERIFICATION.md)。以下保留确认时的设计方案；“当前实现”及“拟执行”均指改造前状态与当时的实施计划。

目标仓库：`D:\Code\Agent\ecommerce-customer-service-agent`。

拟提交目标：远端 `origin` 的 `main` 分支。确认本方案后再编码、测试、提交与推送，不提前执行 Git 写操作。

## 1. 改造结论与范围

本次属于售后模块的中等规模局部重构，不重写整个 Agent。采用 LangGraph 原生 Checkpointer、`interrupt()` 和 `Command(resume=...)`，实现真实图中断及审批恢复；同时使用 SQLite 持久化图状态、审批记录及模拟申请幂等结果。

默认面向单机、单个应用进程运行；通过同一工作流的恢复串行化处理该进程内的并发请求。不宣称实现多实例调度、分布式锁或真实支付的 exactly-once。

外部保留 `POST /chat`、`POST /chat/resume` 及现有主要字段。TaskPlanner、普通 Tool Calling、父子文档 RAG、会话记忆的业务实现不在本次重构范围内。

本次仍然只记录模拟售后申请，不接入真实退款/退货写接口，不新增工作人员账号认证系统。角色字段校验与真实身份认证必须继续区分。

## 2. 当前实现与需要解决的问题

已核对的主要源码：

- `backend/workflows/after_sale_workflow.py`：图无 Checkpointer；`stop_before_submission` 返回 `paused` 后连到 `END`；图返回后手动保存业务快照。
- `backend/state/checkpoints.py`：`CheckpointStore` 用内存字典保存快照与防重结果；`WorkflowResumer` 独立执行校验、查询和模拟申请记录，没有恢复原图。
- `backend/agents/customer_service_agent.py`：主 Agent 调用上述工作流，Resume 后生成公开 Trace 和成本摘要。
- `backend/main.py`：当前在模块级创建 Agent，尚无持久化连接生命周期管理。
- `tests/test_langgraph_workflow.py`：已有资格判断、错误凭证/角色、业务事实变化及重复审批等测试。

当前问题不是完全没有人工审批机制，而是：

1. `paused` 是业务状态，不是真正停在可恢复的图节点。
2. 重启丢失快照、恢复凭证关联与申请防重结果。
3. 图 State 包含 `HookManager` 等运行时对象，不适合原样持久化。
4. 当前工作流 ID 主要由会话和订单组成，同一订单再次申请需要明确区分执行实例。
5. 审批终态与重复/冲突决策需要统一持久化，不能只在响应中返回更新后的状态。

## 3. 目标业务链路

保留订单、物流、政策检索和资格判断，改变其后的执行方式：

```text
识别售后类型 → 读取订单 → 读取物流 → 检索政策 → 判断资格
    ├─ 缺参、事实失败或资格不满足 → 解释/澄清/阻断 → END
    └─ 满足退款/退货申请条件
         → prepare_approval：保存申请实例、审批凭证及事实快照
         → human_review：interrupt()，等待工作人员提交决策
              ├─ rejected → 持久化拒绝结果 → END
              ├─ needs_more_info → 更新待补充状态 → 再次等待审批
              └─ approved → 重查订单/物流 → 比较关键事实
                              ├─ 不一致或查询失败 → 持久化阻断结果 → END
                              └─ 一致 → 幂等记录模拟申请 → 保存完成结果 → END
```

节点名称是拟定实现名称，编码时可小范围调整。取消订单、赔付等现有不自动提交的边界保持不变。

`needs_more_info` 仅维持当前申请的等待状态及审批备注；本期不新增修改订单对象、重新录入材料或完整材料管理功能。需要改变申请对象/关键事实时重新创建申请并核验。

## 4. 数据与存储职责

### 4.1 原生图快照

使用 `SqliteSaver` 保存图 State、待执行节点和中断状态，拟默认路径：

```text
.runtime/after_sale_checkpoints.sqlite3
```

配置 `graph.compile(checkpointer=...)`，首次执行与恢复均使用同一个由服务端生成、关联到申请实例的 `thread_id`。客户端不能自行指定一个任意图线程来恢复。

### 4.2 业务审批与防重存储

使用独立 SQLite 业务存储，拟默认路径：

```text
.runtime/after_sale_approvals.sqlite3
```

职责包括申请关联、审批决策、终态结果、幂等记录及必要审计信息，不再自研一套图运行快照系统。拟包含：

- 申请表：`workflow_id`、`session_id`、`thread_id`、申请人身份、订单、操作类型、审批要求、状态、版本、时间、快照及 Schema 版本。
- 审批记录：服务端匹配的凭证信息、审批人标识、角色、决策、备注、时间、决策处理结果。
- 模拟申请表：唯一 `idempotency_key`、固定申请编号、提交时间及结果。

申请身份、凭证和业务快照均属受控数据，不输出到公共 Trace，不将本地数据库加入 Git。存储只保留必要身份关联，不保存模型 API Key、业务服务 Token 或完整无关聊天历史。恢复凭证的生成、存储与校验保持一致且不得在节点重放时换新；对外只沿用现有审批响应协议。

测试使用注入的临时数据库/内存资源，不共享真实 `.runtime` 文件。

### 4.3 两类存储不是一个跨库原子事务

图 Checkpointer 与业务数据库各自提交，不能宣称二者自动原子一致。通过业务结果防重、终态查询及失败重试补齐边界：

- 若业务模拟申请已落库、图后续快照未完成，重放提交节点必须返回同一申请编号。
- 若终态业务结果已保存、响应丢失，重复 Resume 返回保存的结果，不再次推进图。
- 若只是有业务关联但没有有效图快照/中断，则阻断恢复并返回可定位的原因，不伪造成功。
- 不做“节点结果不明时再生成一份新申请”的兜底。

## 5. 图 State 与运行时依赖改造

State 以可序列化业务数据为主：最小请求字段、申请对象、意图、工具记录、政策引用、资格判断、审批信息、冻结事实、决策及节点历史。

- `HookManager`、HTTP 客户端、PolicyService、数据库连接、锁等运行时对象放在工作流服务/请求执行上下文中，不写进 Checkpoint。
- 不要求把所有 Pydantic 对象都删除；优先建立显式的 `model_dump`/`model_validate` 边界，避免恢复依赖不可控的任意对象反序列化。
- 不引入全局可变的“当前请求/当前 Hook”，防止并发请求串状态。
- Resume 使用服务端保存的申请身份和订单对象，不重新接受客户端指定的业务对象。
- 保持原有 `ChatResponse`、`WorkflowSummary`、工具事实和引用的公开投影；不要把完整图快照直接返回。
- 给持久化 State 加 Schema 版本；遇到不支持的旧版本安全拒绝，不能错误解释后继续提交。

## 6. 申请实例、审批与恢复协议

### 6.1 独立申请实例

每次新售后申请生成独立实例标识和线程关联；同一会话、同一订单的两次申请不能误续同一线程。`workflow_id` 仍保持字符串字段，但不继续把现有确定性格式当成唯一实例保证。

重复 Resume 的幂等范围是同一申请实例；本期不宣称跨不同申请实例自动阻止同一订单的重复业务申请。

### 6.2 Resume 保持外部协议

保留：`session_id`、`workflow_id`、`resume_token`、`reviewer_id`、`reviewer_role`、`decision`、`reviewer_note`。

内部流程：

1. 按会话/申请定位业务记录，校验关联、恢复凭证、非空审批人和要求角色。
2. 对同一工作流恢复加执行互斥，状态检查必须在互斥范围内重新进行。
3. 查询已持久化终态：同一最终决策的重试返回已有结果；冲突决策拒绝，不把已批准变拒绝或把已拒绝变批准。
4. 非终态时确认对应原生图存在且正在等待审批，再调用 `graph.invoke(Command(resume=...), config)`。
5. 将图结果转为现有 `ChatResumeResponse`，继续输出 `resume_result`、`business_recheck` 及脱敏 Trace。

待补充信息不是终态，后续仍可审批。为支持重复 HTTP 请求，本期可用已有字段的规范化摘要识别相同待补充请求，不新增客户端必填幂等字段；这不等同于完整的材料版本管理协议。

错误请求与存储不可用必须得到结构化错误/受控服务错误，不向调用者暴露数据库路径、SQL、身份、Token 或堆栈。避免将存储失败误报为“订单事实变化”。

### 6.3 不冒充完整鉴权

本期保留角色及凭证检查，不新增真实工作人员账号/JWT/RBAC。`reviewer_id` 与 `reviewer_role` 仍要求来自可信管理入口；公开暴露部署时必须由认证服务注入并验证，而不能信任调用方自报角色。Resume Token 是流程恢复凭证，不是工作人员登录凭证。

## 7. 节点重放、业务复查和幂等

### 7.1 中断前后职责

原生恢复会重新执行被中断节点，所以：

- 申请实例、Token、幂等键和事实快照在 `prepare_approval` 阶段稳定建立并持久化；该阶段自身也应可重放。
- `human_review` 负责中断和读取决策；中断前不创建模拟申请、不重复生成 Token，不重复追加不可幂等的审计/节点历史。
- 待补充分支通过节点/条件边回到审批等待，不用在一个节点内无限堆叠 `interrupt()` 调用。
- 不捕获并吞掉 LangGraph 的中断信号，不能把正常中断降级成系统错误。

### 7.2 业务复查

保留现有七个字段：订单状态、支付状态、金额、履约状态、签收时间、可退属性、物流状态。

订单/物流任意读取失败就阻断；任意关键事实不一致就阻断。只在批准后复查，拒绝或待补充不产生业务申请。

本期不扩展为政策版本的全面重新评估，也不宣称彻底消除“复查与真实业务提交之间”的 TOCTOU；当前没有真实外部写操作。

### 7.3 模拟申请幂等

保留 SHA256 派生稳定幂等键的思路，输入绑定申请实例、操作类型及订单。业务表设置唯一约束，在短事务中完成记录/已有记录查询。

同一申请的重复批准及并发批准必须得到同一申请编号。数据库唯一约束是最终防重边界，进程锁主要负责防止同一图线程被并发推进。不要跨网络查询持有长数据库事务。

这只保证本项目模拟申请记录的防重，不等于外部支付 exactly-once。将来接真实业务写接口仍需其自身的幂等协议与结果查询。

## 8. 初始化、依赖与配置

- 增加并锁定与当前 `langgraph==1.1.10` 兼容的 `langgraph-checkpoint-sqlite` 版本，先做依赖解析与小型兼容性验证，不为了迁移无理由升级全部依赖。
- 使用当前版本支持的 `interrupt()`、`Command(resume=...)`、`invoke()` 和快照查询；不直接照搬新文档中要求更高版本的接口。
- 增加两个路径配置，拟命名 `AGENT_WORKFLOW_CHECKPOINT_PATH` 与 `AGENT_WORKFLOW_APPROVAL_STORE_PATH`，相对路径统一按仓库根目录解析。
- FastAPI 生命周期负责创建/关闭持久化连接与 Agent，避免模块导入时创建正式数据库；测试支持显式依赖注入及资源关闭。
- 初始化业务表，配置必要的 SQLite 锁等待与连接使用方式；只声明本期验证过的单进程并发范围。
- 默认运行路径继续处于已忽略的 `.runtime/`；启动失败不得静默退回内存而假装具有持久化能力。
- 旧进程内申请没有可迁移的持久化来源；切换前处理已有待审批申请或提示重新发起，不编造历史恢复能力。

## 9. 拟改文件

| 文件/模块 | 改造内容 |
|---|---|
| `backend/workflows/after_sale_workflow.py` | State 拆分、图中断、恢复后节点、分支与公开摘要 |
| `backend/state/checkpoints.py` | 改为恢复协议协调与结果投影；移除手写图快照职责，保留/抽出业务校验 |
| `backend/state/native_checkpoint.py`（新增） | 原生 Checkpointer 工厂、连接生命周期及线程执行协调 |
| `backend/state/approval_store.py`（新增） | SQLite 申请/审批/模拟申请存储、幂等和终态结果 |
| `backend/agents/customer_service_agent.py` | 持久化依赖注入、Resume 接入及 Trace/成本兼容 |
| `backend/main.py` | 应用生命周期与资源关闭 |
| `backend/config/settings.py`、`.env.example`、`requirements.txt` | 路径与依赖配置 |
| `backend/api/schemas.py` | 保留协议，修正文档说明；确有需要时才加可选字段 |
| `backend/approvals/hitl.py` | 修正“尚未开放 /chat/resume”等与现有实现不符的文案 |
| `tests/test_langgraph_workflow.py` | 更新原图节点与中断/恢复断言 |
| `tests/test_workflow_persistence.py`（新增） | 跨实例/子进程持久化、序列化与资源关闭测试 |
| `tests/test_workflow_resume.py`（新增） | 决策分支、防重、并发、终态冲突与故障重放测试 |
| `tests/test_workflow_api.py`（新增） | Resume HTTP 契约及应用生命周期测试 |
| 现有上下文、Trace、成本、评测相关测试/`backend/cases.yml` | 必要的字段/节点断言适配，不扩大业务能力 |
| `README.md`、`CHANGELOG.md`、`readmeImg/architecture.*` | 更新工作流说明与架构图，只描述实测通过的能力 |

仓库已有未跟踪 `docs/` 内容保留，不整体纳入本次提交。外部面试复习文档不在此次仓库实现范围内，可在功能完成后另行确认更新。

## 10. 测试与验收

### 10.1 必须通过的场景

1. 退款与退货的原有资格判断仍然正确；缺参/订单失败仍走短路。
2. 资格通过后真实存在待恢复的原生中断；尚未记录模拟申请。
3. 批准后走原图后续节点，产生唯一模拟申请编号。
4. 拒绝不提交；要求补充信息保持等待，之后可继续审批。
5. 错误会话、申请、凭证、角色或空审批人被阻断，且不查询业务/不推进图。
6. 订单或物流事实变化、重查超时/失败均阻断提交。
7. 重复批准返回同一编号；同进程并发批准只记录一次。
8. 终态拒绝后批准、批准后拒绝等冲突决策不改变已有结果。
9. 同用户/会话/订单的不同申请实例隔离；不同用户/会话不能串恢复。
10. 关闭存储/Agent，再用相同数据库路径创建新实例，仍能恢复等待审批的申请。
11. 子进程退出再启动后仍可恢复，证明不是只换了对象却仍依赖原进程内存。
12. 模拟申请落库后、图快照/响应完成前注入故障，重试不产生第二个申请。
13. 存储不可写、记录缺失、Schema 不支持时安全报错，不执行模拟提交。
14. State 可序列化，重启恢复不用反序列化 Hook/HTTP 客户端；公共 Trace 不泄露凭证或身份。
15. Resume 不重新调用规划/回答模型；原有 RAG、工具、上下文、成本与评测测试无回归。

离线测试注入假业务服务和模型客户端，不使用真实支付接口，也不为验证恢复机制调用收费模型。现有运行时评测可受外部服务配置影响，不未经确认跑真实付费评测。

### 10.2 完成标准

只有实现和测试均通过后，才可在 README 写“原生中断/恢复”“SQLite 持久化”“重启后恢复”和“模拟申请幂等”。不编造吞吐、P95、可用性或真实退款成功率。

保存测试命令、用例数量、结果和故障注入验证记录；README 架构图与实际实现一致。

## 11. 实施顺序与风险控制

1. 用户确认本方案及 `main` 推送目标。
2. 再次检查工作区与远端状态；保留已有 `docs/` 和其他用户改动，发现冲突先沟通。
3. 锁定 SQLite 插件版本，建立临时库验证最小的原生暂停/恢复。
4. 建立业务存储、原生 Checkpointer 与可序列化 State，保留对外协议。
5. 接入真正的图中断和恢复节点，补齐业务事实复查、终态检查、防重及故障重放。
6. 接入 Agent/FastAPI 生命周期，运行专项测试和完整回归。
7. 更新 README、CHANGELOG、架构图，检查 Diff 和暂存内容。
8. 本地测试通过、远端没有待处理分叉后，使用中文提交信息提交并推送。

本期不增加双套恢复机制或自动切回内存的兼容分支。新旧内存申请不做虚假迁移。需要回退代码时另行确认，以可追踪的提交回退方式处理，不使用强制推送或破坏性重置。

## 12. 参考依据

- [LangGraph：Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)：原生中断、线程关联、恢复命令与中断节点重执行语义。
- [LangGraph：Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)：图快照与业务存储职责；内存不会跨重启保留，SQLite 可用于本地持久化。

## 13. 拟提交类型、命令与推送门槛

拟采用 `feat(workflow)`，因为本次新增真实图中断及跨重启恢复能力，虽然内部包含重构，但不只是等价整理代码。

以下为确认后拟执行的 PowerShell 命令，本轮不执行。文件名单按最终实际改动核实，新增文件名如有调整同步更新命令；不使用 `git add .`，也不提交 `.env`、SQLite 数据库或既有未跟踪 `docs/`。

```powershell
Set-Location -LiteralPath 'D:\Code\Agent\ecommerce-customer-service-agent'

# 实施前先核对 main 与远端，出现分叉或他人新提交时先处理，不强推。
git status --short --branch
git fetch origin
git rev-list --left-right --count main...origin/main

# 修改依赖文件后安装经过锁定的版本，测试使用临时存储和假客户端。
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
$env:AGENT_DISABLE_LLM = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check

# 只暂存本任务经核实的改动。下列名称是方案中的拟定文件名单。
git add -- AFTER_SALE_CHECKPOINT_PLAN.md requirements.txt .env.example README.md CHANGELOG.md backend/main.py backend/config/settings.py backend/agents/customer_service_agent.py backend/api/schemas.py backend/approvals/hitl.py backend/workflows/after_sale_workflow.py backend/state/checkpoints.py backend/state/native_checkpoint.py backend/state/approval_store.py tests/test_langgraph_workflow.py tests/test_workflow_persistence.py tests/test_workflow_resume.py tests/test_workflow_api.py readmeImg/architecture.svg readmeImg/architecture.png readmeImg/architecture-prompt.md
# 其余确有修改的现有测试/评测文件，审核后逐一显式 git add，不整目录暂存。
git diff --cached --check
git diff --cached --stat
git diff --cached

git commit -m 'feat(workflow): 接入 LangGraph 原生中断与持久化审批恢复' -m '使用 SQLite 保存图快照、审批记录和模拟申请幂等结果；保留 Resume 协议，补齐事实复查、终态约束、重启恢复及回归测试，并更新架构文档。'

# 推送前再次核对远端，确认 origin/main 是本地 main 的祖先。
git fetch origin
git merge-base --is-ancestor origin/main main
# 仅上述检查成功且测试通过后执行；检查失败则停止，不自动覆盖远端。
git push origin main
git status --short --branch
```
