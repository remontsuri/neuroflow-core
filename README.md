# neuroflow-core :: OSW Engine + State-Machine Segmenter

**Mythos build — production-grade multi-agent orchestration for Hermes Agent.**

## Что внутри

```
neuroflow-core/
├── osw_engine.py             # DAG-based Orchestrator-Supervisor-Worker
│   ├── DAG                   # Топологическая сортировка, параллельные батчи
│   ├── StateMachine          # Жизненный цикл агента (8 состояний)
│   ├── GraphMemory           # Entity-relationship память с trust/decay
│   ├── AgentCard             # Machine-readable agent cards
│   └── OSWEngine             # End-to-end оркестратор
│
├── telegram_segmentation.py  # State-Machine сегментация Telegram-пользователей
│   ├── 7 UserStates          # lead → active → warm → hot → cold → churned → banned
│   ├── 16 Triggers           # join, view, reaction, reply, click, dm, purchase...
│   └── TelegramSegmenter     # Thread-safe, decay, hot leads pipeline
│
├── runner.py                 # Демо: всё сразу
└── README.md
```

## OSW Engine — как работает

```
Goal → Orchestrator → Supervisor(s) → Worker(s) → Aggregate → Verify → Deliver
       │                  │               │
       └── DAG graph ─────┴───────────────┘
           топ. сорт. → параллельные батчи → retry с backoff
```

**DAG execution**: задачи сортируются топологически, выполняются параллельными слоями. Если worker упал — retry до 3 раз. Если supervisor упал — весь sub-DAG помечается failed.

**GraphMemory**: каждая сущность (run, plan, task, worker) хранится с trust score. Есть decay — непрочитанные сущности теряют вес. Нейросвязи можно трассировать через `neighbors(name, relation)`.

## Telegram Segmenter — почему это лучше тегов

**Теги** — статика. Пользователь поставил реакцию → тег "engaged". Через месяц молчания тег тот же.

**State Machine** — динамика. Пользователь молчит 7 дней → автоматический переход в COLD. Написал снова → реюзается в ACTIVE. Купил → HOT.

```
        ┌─── LEAD ───→ ACTIVE ───→ WARM ───→ HOT ───→ CONVERTED
        │     │           │          │          │
        │     └──→ COLD ←┘          │          │
        │          │                │          │
        └──────────┴──── CHURNED ←──┴──────────┘
                          SPAM → BANNED
```

## Как использовать

### Из кода
```python
from osw_engine import OSWEngine, AgentCard
from telegram_segmentation import TelegramSegmenter

# Оркестратор
engine = OSWEngine()
engine.ingest_goal("Проанализировать каналы")
engine.decompose(run_id, [...])
engine.execute(my_executor)
print(engine.report())

# Сегментация
seg = TelegramSegmenter()
seg.process_message(user_id, "reaction")
seg.process_message(user_id, "purchase")
print(seg.segment_counts())
print(seg.hot_leads())
```

### Из Hermes CLI
Сделай cron job который:
1. Раз в час дёргает Telegram API → отдаёт события в `process_message()`
2. Раз в день запускает `run_decay()` → сегментирует холодных
3. При достижении HOT — алерт "горячий лид! отправь оффер"

## Интеграция с Hermes Agent

Подключить как skill:
```bash
hermes config set skills.neuroflow-core /opt/code/neuroflow-core/SKILL.md
```

Или запускать через `eni-worker`:
```bash
eni-worker "запусти нейрофлоу: кто сегодня перешёл в HOT, отправь авто-оффер"
```
