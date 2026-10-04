# HBS / PersonBench: сетка экспериментов для AAMAS 2027

**Дата:** 4 октября 2026. **Статус:** подготовленный план; новые научные эксперименты в этой сессии не запускались.

**Основная рекомендация:** использовать накопленный поиск как разработку, заморозить его артефакты и доказать, что обученный промпт переносится на новых людей без знания их целевых ответов. Приоритет — независимая оценка, сильный best-of-B, статистические модели с теми же данными и внешний поведенческий тест. Завершение всей старой сетки из 780 ячеек не является условием сильной статьи.

**Проблема:** как получить пригодного для повторного использования симулятора индивидуальных ответов из ограниченной истории человека, с измеримой ценностью и понятной стоимостью.

**Method thesis, пока гипотеза:** proper-score search обучает переносимый шаблон представления персоны; выигрыш сохраняется при честном сравнении с обычным поиском и сильными моделями на той же человеческой информации. Если данные показывают, что достаточно GA или best-of-B, выбираем простой работающий вариант, а не объявляем сложный оптимизатор победителем.

Регистрационные заметки от 1 октября называют статью **Predictive Personas: Proper-Score Optimization of LLM Respondent Simulators**. Это контекст из предыдущей сессии; форму OpenReview сейчас не перепроверяли. Название согласуется с предлагаемой работой. Изменять его или abstract в этой сессии не требуется.

## 1. Основание решений и текущий результат

Прочитаны центральные и проектные curated memories Claude, старый HSE/MAGE index, планы от 4 и 25 сентября, весь `arr2026/strengthen/BRIEF.md` до §26, текущие `icml2027/{PLAN,PREREGISTRATION,STATUS,RESULTS_BRIEF}.md`, код формирования выборок, поиска, calibration, transfer и итоговых тестов. Сырые диалоги Claude не потребовались.

| Предыдущий запрос / ограничение | Решение в новой сетке |
|---|---|
| Руководитель: больше современных оптимизаторов и моделей, не только GigaEvo | Полная старая development-сетка сохраняется; main comparison концентрируется на GA, MAP-Elites-style, GEPA-style, random, best-of-B и двух реально измеренных backbone families |
| Пользователь: успешный и полезный результат, не каталог неудач | Основной deliverable — замороженный переносимый симулятор и доказанная область полезности; null используется как граница, не как обещанный вклад |
| NeurIPS: маленькие эффекты, слабые baselines, k=4, нет external behaviour | B1 закрывает честные сравнения, B2 механизм, B3 внешний перенос, B4 практическую ценность |
| BRIEF: same-history decoder, disjoint respondents/items, budget tuning | Обязательные контролы и отдельный протокол новых подтверждающих данных |
| Старый Paper A / новый Paper B | Метрика m* — измерительный инструмент из Paper A, а новый вклад — обучение и внешняя проверка симулятора; overlap проверяется до подачи |
| Slurm only, сохранность артефактов, finite walltime | Никаких bare CPU/GPU задач на Euler, безопасная синхронизация и отдельное пространство результатов |

### 1.1 Что проверено сейчас

SSH-проверка Euler около 18:15 UTC / 21:15 МСК 4 октября: HBS jobs не выполнялись; diagnostic `8311` был `PENDING (Resources)`, 38 cell tasks — `JobHeldUser`, stage — по dependency. Это моментальный снимок, не прогноз старта. Удержания и broker не изменялись.

Скопированы только сохранённые metadata/eval JSON в отдельный архив. [Проверяемая сводка](hbs_aamas_20261004/CURRENT_EVIDENCE.md), [полный реестр 780 ячеек](hbs_aamas_20261004/evidence/cell_inventory.csv), [машинная сводка и SHA256](hbs_aamas_20261004/evidence/snapshot_summary.json).

- **266/780 completed**, 405 pending, 96 `failed_oom`, 10 `failed_other`, 3 старых `running` в файлах. Метка `failed_oom` также включает ошибки связи с сервером, поэтому не означает 96 отдельно доказанных OOM.
- **254** completed с основной fitness + **12** S0 ablations. Это валидность наличия сохранённых файлов, не признание чистоты научного протокола.
- Основные результаты: Qwen3.6 — 80 ячеек, Granite-30B — 120, Gemma — 30, GLM — 24. Qwen3.8 пока отсутствует среди completed. GLM завершил только DE и best-of-B: считать его полным сравнением нельзя.
- Сводка `RESULTS_BRIEF.md` на 89 ячейках устарела; её результаты m* нельзя переносить на все 266.

| Модель и покрытие | GA: ΔRPS к base | MAP: ΔRPS к base | best-of-B: ΔRPS к base | MAP − best-of-B | MAP − GA |
|---|---:|---:|---:|---:|---:|
| Qwen3.6, кластеры 0–2 × 3 seeds | −0.004242 | −0.004395 | −0.003750 | −0.000647 | −0.000151 |
| Granite-30B, 4 кластера × 3 seeds | −0.004374 | −0.004389 | −0.003993 | −0.000396 | −0.000015 |

Последние два столбца рассчитаны на совпадающих `(model, cluster, seed)` с одинаковыми хешами eval IDs. Это **описательные средние**, без новых CI/p-values. Они показывают, почему GA нужно сохранить как реального претендента, а best-of-B — как сильный контроль. По этим данным превосходство MAP-Elites над GA не установлено.

Во всех **12/12** завершённых S0-ячеек calibrated RPS тоже улучшился относительно base. Теоретическая неправильность S0 как scoring rule не доказывает, что именно proper-score objective дала эмпирический выигрыш в этом конечном пространстве промптов. Нужен прямой objective contrast.

### 1.2 Исправления устаревшего контекста

- Память «0/506 тестов» устарела: BRIEF §19 сообщает hardened rerun на 160 допустимых ячейках, **1/426** после BH, с отсутствием репликации этого эффекта. Не цитировать старую нулевую границу как текущий итог.
- Старый transfer уже дал m*=5.38 из 120 для Qwen3.6; IPIP-300 давно изучался. Этот набор больше не является untouched holdout всей исследовательской программы.
- m*=5.38 после предъявления 120 ответов означает сохранённую прогнозную ценность, а **не экономию 114.62 опросных ответов**. Экономию нужно измерить отдельной кривой качества при меньшем числе входов.
- Профиль не универсально лучше transcript: small-model Gemma менял направление эффекта. Не превращать частный результат в общий закон.
- Случайный человек из того же кластера — point-forecast reference, а не «человеческий потолок» probabilistic RPS.

## 2. Claim map: два утверждения, которые стоит защищать

| Claim | Статус сейчас | Минимальное убедительное свидетельство | Блоки |
|---|---|---|---|
| **C1. Обученный шаблон персоны повышает индивидуальную прогнозную ценность на новых людях при фиксированных данных и бюджете.** | Гипотеза с development-сигналом | Новый глобальный test без target-derived routing; paired RPS против base, random и best-of-B; correct-person advantage; неопределённость Δm*; две backbone families | B0, B1, B2 |
| **C2. Этот выигрыш полезен за пределами исходного опросника в обозначенном режиме информации и затрат.** | Не установлено | Независимые внешние respondents и target tasks; same-history статистический comparator; отдельные zero-target-label и target-adapted режимы; выигрыш в потере или честно измеренная экономия human observations | B3, B4 |

**Anti-claims:** выигрыш создают oracle cluster, дополнительная калибровочная разметка, больший budget, другой formatter, selection на test, ослабленный decoder или population correction без идентичности человека.

**Не заявляем заранее:** новый SOTA optimizer; превосходство над всеми статистическими моделями; «цифровой двойник человека»; cross-instrument individual prediction по разным людям; сохранение joint distribution из одного роста дисперсии.

В главной статье — working simulator, честное сравнение и внешний результат. В appendix — полная старая сетка и все исходные H1–H6 с реальным статусом, failures/exclusions, остальные модели. Не закрытые гипотезы помечаются incomplete, а не отрицательными.

## 3. B0 — готовность доказательства и фиксация протокола

**Приоритет: MUST, до любых новых confirmatory scores.** Цель — не объявить всю старую работу бесполезной, а точно отделить её от нового доказательства.

| Проверка | Что найдено / риск | Точное действие и критерий готовности |
|---|---|---|
| **Target-derived clusters** | `panels.py:make_panels` фильтрует по готовым `clusters`; `seed_genotype(cluster)` выбирает prose; prior тоже per-cluster. `paper/main.tex`, §disjoint, прямо говорит: эти кластеры построены из всех 120 ответов | Новая primary evaluation глобальная. Один фиксированный prompt на систему/seed для всех людей, global prior; колонка `clusters` недоступна inference path. Старый режим — oracle-stratified development. Без этого новый predictive claim запрещён |
| **Исторические respondents** | Exclusion config охватывает только часть `evoprompt_iter2`; разные seed draw могут перекрывать роли | Полный exposure ledger из arr2026, icml2027 и их train/opt/cal/eval: `(dataset fingerprint, respondent ID)`. Проверка пересечений новых eval с любыми старыми и новыми fit/cal/dev. Если реконструкция ID ненадёжна — confirmatory label не присваивается |
| **Выбор по test** | `select_headline.py` выбирает arm и models по уже прочитанным frozen-eval Δ | Эти результаты — development. Новый test не участвует ни в выборе, ни в sample-size stopping; seeds 4–5 не делают старые respondents вновь закрытыми |
| **Регистрация** | У всех 266 completed в metadata пустой `prereg_sha`; OSF pending | Сохранить старый документ и фактические даты. До новых test predictions создать amendment с data hashes, executable config, commit/source hashes и analysis plan. Не называть её регистрацией задним числом |
| **Calibration / projection** | `iso_project` сначала clip, затем isotonic; это не всегда точная bounded L2 projection. `crossfit_rps_cal` центрирует на scoring fold, а frozen eval — на calibration fold | G0 regression на bounded isotonic против независимого PAVA; обученные F0/Qbar/alpha только из разрешённых частей. Например `(1.4,.5,.9,1)` имеет bounded projection `(.9333,.9333,.9333,1)`, а предварительный clip даёт `(.75,.75,.9,1)`. Версионировать исправленный estimator; affected old cells не смешивать с новым протоколом |
| **Common support / missingness** | В transfer decoder встречается `Y <= t` без предварительной маски NaN; arm-wise nanmean может менять знаменатель | Один support по наличию human labels; coverage каждого метода; invalid forecast — заранее заданный fallback prior или failure с числом случаев. Проверить валидные simplex, ориентацию, full digit mass; sensitivity на common support |
| **Поиск и честность названий** | `gepa` и `mapelites` — местные реализации; не установлен parитет с official GEPA / GigaEvo | В таблице `GEPA-style`, `MAP-Elites-style`. Для SOTA claim нужен pinned official GEPA adapter на том же evaluator. Для quality-diversity claim — deletion archive/islands/insight; пока этого нет, не заявлять новый алгоритм |
| **Budget** | GEPA использует fractional evaluations и дозаписывает остаток budget без обязательного forward pass | Считать actual evaluated respondent–item predictions, mutator calls/tokens, repair calls, retries, GPU-seconds отдельно от ledger. B=80 — full-panel equivalents, не всегда 80 кандидатов. Проверка attempted / executed / cached, никакого фиктивного вычисления |
| **Serving** | 8311 ещё не прошёл; Qwen startup тоже падал; GLM smoke не проверяет другие модели | Отдельный measured smoke на каждой реально используемой паре backbone+mutator, затем один limited retry без eval. Сервер внутри finite Slurm allocation, не persistent bare service |

Дополнительные analysis fixes: H3 сейчас отбрасывает never-reached cells и меняет парный тест; H6 строит непарную разность; H2 проверяет доли отдельно для двух контролей, а не их пересечение. Новая статистика описана в §9. В этой сессии код scorer/runner не менялся: таблица — конкретный backlog готовности.

**Выход B0:** подписанный источниками integrity receipt, immutable split manifest, hash всех prompt candidates, tested scorer, measurement-based throughput. Exit `0:0` подтверждает механику запуска; проходит научный gate только содержимое receipt.

## 4. B1 — основная таблица: сильное подтверждение уже обученных шаблонов

**Приоритет: MUST.** Сначала дешёвый frozen-artifact route; новый поиск только после его оценки. Это не перезапуск всех 780 ячеек.

### 4.1 Данные и выбор системы

IPIP-120: 60 input / 60 target, фиксированный item map. Новый общий manifest после исключения всего exposure ledger:

- `T_fit`: 8,000 respondents для global prior и statistical models.
- `D_select`: 160 новых respondents для выбора глобального шаблона из уже обученных кандидатов.
- `C_cal`: 160 других respondents для alpha, Qbar и выбора baseline hyperparameters.
- `E_confirm`: **800** новых respondents, недоступны selection/calibration. Размер предварительный: уточняется power calculation на development variance до раскрытия E; минимально 400 допустимо только с заявленной меньшей мощностью.
- Независимые master seeds назначить в amendment. Search seeds `{1,2,3}` — три уже обученных шаблона, не три новых respondent datasets. Один общий E используется всеми системами для paired comparison.

**Модели фиксируются сейчас:** `Qwen/Qwen3.6-35B-A3B` и `ibm-granite/granite-4.2-30b`. Это сильный ранний результат и полная другая family; точные weight/tokenizer revisions и serving settings должны попасть в manifest. Не заявлять их «самыми новыми» по непроверенным release dates из старого config.

**Пять search arms:** GA, MAP-Elites-style, GEPA-style, random, best-of-B. Для каждого `(model, arm, search seed)` взять ровно **три финальных кандидата** из исторических development-кластеров `{0,1,2}`. Это 2 × 5 × 3 × 3 = **90 кандидатов**, прошедших одинаковые исторические B=80 full-panel-equivalent бюджеты. Для Qwen GEPA c1/s3 отсутствует: до её безопасного восстановления либо сохранять incomplete comparator, либо выбрать общий two-seed вариант *до* раскрытия нового E. Не подставлять чужой seed.

В D_select всем 90 кандидатам дать одинаковую процедуру cross-fitted corrected RPS с global F0. В каждой тройке выбрать один global prompt. Получается **30 замороженных learned systems**. Ни одному человеку на inference не присваивается старый кластер. Историческое использование target labels при обучении шаблонов является supervised development; использование их для маршрутизации нового человека устранено.

Честная стоимость такой системы: исторический поиск **3 × 80** full-panel equivalents плюс одинаковый D_select cost. Нельзя писать «весь global method стоит B=80». Можно отдельно описывать B=80 каждого исходного run.

### 4.2 Сравнения и метрики

Три baseline families:

1. **Без learned search:** нейтральный фиксированный profile prompt, numeric profile, direct observed `(question, answer)` history; no-history prior. Контекст ответа и формат readout одинаковые. Candidate template не выбирается на E.
2. **Поиск при одинаковом бюджете:** random, best-of-B, GA, MAP, GEPA-style. Base остаётся общим. Paraphrase chain из старого кода — no-selection control, не полноценный конкурент лучшему поиску.
3. **Статистические модели с теми же наблюдениями:** one-hot raw-answer ridge-CDF, tuned ordinal/boosted model и компактная item-factor/IRT или другая text-conditioned модель для cold items. На warm IPIP primary — лучший из заранее фиксированного небольшого набора, выбранный только по C_cal; neural capacity sweep не нужен.

Primary: normalized per-person RPS, сначала усреднение по 60 target items человека, затем по людям. Secondary: ordinal log loss с заранее фиксированным clipping, Δm*, calibration, identity gain. На всех основных таблицах actual n и coverage.

**C1 success:** adjusted paired CI улучшения против base и сильнейшего search control ниже 0, знак одинаков у двух моделей; дополнительно заранее заданная useful margin **0.001 absolute RPS или 1% relative loss** как целевой уровень значимости для применения. Это новый proposed planning threshold, не старый preregistered gate и не обещание его преодолеть. Слабее этой границы можно сообщать положительный эффект, но нельзя называть его крупным.

Δm* — supporting effect size; bootstrap включает randomness item subsets и calibration/decoder fitting на соответствующих разрешённых панелях. Zero / right-censored crossing выводятся отдельно, не заменяются произвольным числом.

Если best-of-B не хуже сложного метода с убедительным equivalence/noninferiority интервалом заранее выбранной ширины, предпочесть best-of-B. Если только CI включает ноль — это **не доказанная ничья**. MAP против GA с observed разницей порядка 1e−5 на Granite не несёт main story без нового подтверждения.

**Table 1:** model × system, RPS, paired Δ к сильнейшему control, Δm*, actual human observations, forward/token/GPU cost. **Figure 1:** decoder loss curve и позиции base/learned, без превращения m* в экономию опроса.

### 4.3 Расширение при доступном бюджете

Чистый global fresh-search run: opt panel n=80 из нового development пула; fixed common folds для каждого кандидата; те же fit/cal/E роли. **2 модели × 5 arms × 3 seeds = 30 search cells**, B=80. Изоляция без кластеров начинается уже в обучении. Search checkpoints 10/20/40/80 выбираются на opt, не на E. Это отдельная серия B1-FRESH, не условие завершения frozen-artifact оценки. К old+new результатам не применять один общий preregistration label.

## 5. B2 — механизм, простота и falsification

**Приоритет: MUST для границ C1; часть ablations условна.**

| ID | Сравнение | Размер | Что решает |
|---|---|---|---|
| B2-ID | правильная persona / permutation / no-history | Каждый frozen learned system, тот же E; 100 фиксированных permutations saved predictions | Есть ли person-specific signal сверх population correction? Shuffle вместе весь профиль одного человека; group-matched shuffle — sensitivity, без oracle cluster |
| B2-OBJ | GA-S0 vs GA-RPS с одинаковыми budget, template и panels | Сначала Qwen: 3 seeds, по 3 historical candidates каждой fitness; **3 дополнительных выбранных S0 prompts** к уже имеющимся GA-RPS | Какая часть эффекта обязана objective? Проверять прямую разность, а не значимость одного arm и незначимость другого |
| B2-REP | learned profile vs neutral verbal profile vs numeric profile vs raw history | 2 модели, те же 60 observations; base/numeric/history — 6 static systems | Семантика текста, агрегация или дополнительный контекст? Учитывать длину и latency; token-matched sensitivity, не обрезающий разный смысл |
| B2-CAL | raw / calibrated для одного prompt; global prior / identity component | Из сохранённых predictions на train/cal/E; alpha=0 admissible | Что добавляет обучение сверх calibration? Qbar/alpha учатся без E. Растущая alpha сама по себе не доказательство причинного механизма |
| B2-SIMPLE | GA vs MAP; при превосходстве MAP — delete archive / islands / insights | Existing paired table; **3 × 2 models × 3 seeds = 18 fresh ablation cells** только если нужен QD-method claim | Оправдана ли сложность? Без выигрыша MAP fresh QD ablations откладываются |
| B2-OFFICIAL | официальный GEPA на pinned revision и адаптере той же задачи | 2 модели × 3 seeds = 6 search cells, если заявляется optimizer SOTA | Нельзя использовать превосходство над локальным port как превосходство над опубликованным GEPA |

На B2-OBJ основной исход — direct paired objective contrast и uncertainty. Улучшение RPS у S0 допускается и обязательно показывается. Raw-RPS GA как третья fitness — 6 fresh cells только после пары S0/cal-RPS; не нужен новый полный factorial.

**Success:** correct identity даёт положительный gain с CI, no-history не объясняет основной выигрыш; простейший выбранный method сохраняет преимущество. S0-versus-RPS может остаться неразличимым: тогда proper scoring — корректный принцип оценки, а не эмпирически доказанный источник всего улучшения.

**Figure 2:** paired objective contrasts, identity gain и representation ablation. Увеличение VR alone не считается fidelity. Старый `VR_between` и текущий ratio of variances имеют разные формулы; фиксировать название и формулу, не сравнивать численно через разные реализации.

## 6. B3 — перенос, который действительно усиливает AAMAS

**Приоритет: внешний behavioural block MUST для C2.** IPIP-300 нужен как связующий анализ, но он уже development-exposed.

### B3-A. IPIP-300: повторная проверка переноса внутри семейства

- Данные: `data/PAlign/Test-set.json`, 300 человек, targets — 180 дополнительных items; linkage исключительно внутри файла. Совпадения case ID с IPIP-120 не являются теми же людьми.
- В основном matched-input варианте использовать те же **60** short-form observed items, что в B1; вариант со всеми 120 — отдельная sensitivity, не смешивать преимущества наблюдений и переноса.
- Две модели × {fixed base, best simple search control, выбранный method} × 3 learned seeds; static base общий. Промпты и choice rule замораживаются до внешних scores. Для transfer select seed по D_select или показать все, не выбрать лучший внешний.
- `target-adapted`: 150 train / 50 cal / 100 eval, как реализовано сейчас, но честно обозначить 200 людей с размеченными target items и историческую экспозицию всего корпуса. Fixed split не делает данные впервые закрытыми.
- `zero-target-label`: никакого F0 или alpha по ответам на новые items; перенос source calibration плюс разрешённый text-conditioned prior. Comparator — cold sibling / item-text decoder, не warm decoder с чужими label privileges. Этот estimator нужно явно реализовать; существующий `transfer.py` не является zero-shot path.
- RPS и Δm* по режимам отдельно, per-facet breakdown, uncertainty over respondents и families. Сохранение знака поддерживает перенос; оно не заменяет B3-B.

### B3-B. Twin-2K-500: новый внешний поведенческий тест

Выбран потому, что даёт same-person self-reports, economic/cognitive tasks и повторные наблюдения. Dataset card и официальный repository проверены; локальный готовый adapter в проекте не найден. До запуска требуется loader audit и freeze точных question IDs, но не нужен новый сбор людей.

**Источники:** [paper](https://arxiv.org/abs/2505.17479), [dataset card](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), [official code](https://github.com/tianyipeng-lab/Digital-Twin-Simulation).

Предлагаемый split на уровне `pid`: 1,000 fit / 200 dev / 200 cal / оставшиеся eligible (~658 при полном N=2058) test. Число 658 — арифметическое ожидание по описанию корпуса, не проверенное число complete cases. Пропуски не заполнять синтетическими labels; публиковать число людей по каждому endpoint.

**Два заранее выбранных семейства:** economic preferences/decision tasks и heuristics/cognitive-bias tasks. Конкретные blocks/QIDs выбираются по codebook, формату допустимых прогнозов и достаточности human labels **до просмотра model outcomes**, затем подписываются manifest. Оба семейства остаются в отчёте независимо от результата; нельзя заменить неудобное успешным третьим.

**Input:** только разрешённые wave1–3 responses, исключая любые повторы и парафразы target task. Wave4 ответы — только scoring. `full_persona` и готовые `persona_summary` не использовать: card прямо указывает, что full_persona содержит ответы на все вопросы, в том числе wave4. Wave-split JSON или raw catalog с whitelist входов безопаснее. Wave1–3 ответы на повторённые wave4 вопросы идут только в отдельный test-retest reference, не в persona.

**Track T0:** frozen source prompt без выбора/обучения на external target labels. Проверить, что 30-facet representation реально извлекается из разрешённых входов. Нельзя выдумывать отсутствующие NEO facets из Big Five. Если такого mapping нет, T0 оценивает общий instruction component с явно указанным изменением representation, а полного method-transfer claim нет.

**Track Tsmall:** тот же search/evaluation framework переобучается на external development с нейтральным domain-valid profile и B=40; **2 models × {GA, best-of-B} × 3 seeds = 12 search cells**. Source-best arm допускается вместо GA только если это правило заранее зафиксировано. Это перенос *процедуры обучения*, не zero-shot transfer source genotype. Для дедлайна Tsmall запускается только если adapter+data gate закрыты к 5 октября.

**Главные comparators:** same-history ridge/ordinal/boosted models с retuning при каждом N; no-history prior; neutral history LLM; best-of-B; выбранный learned method. Все получают те же наблюдения и метаданные. Опубликованные GPT4.1-mini simulations пригодны как contextual reference, но без equal information/budget не заменяют matched baseline.

Для ordinal outcomes — RPS, binary/nominal — Brier/log loss, continuous — CRPS при определённом distributional readout; mean-only RMSE отдельная вспомогательная метрика. Не кодировать произвольные nominal категории как порядковую шкалу. Main aggregate — макроусреднение относительного skill к заранее заданному prior отдельно по семействам; показать и исходные потери, не усреднять несопоставимые сырые шкалы.

**Сильный C2 gate:** улучшение против strongest same-information comparator, pooled paired CI исключает 0, знак сохраняется на обоих task families; целевой **5% relative loss reduction** из старого BRIEF как высокий planning bar. Smaller robust effects сообщаются в масштабе, но не объявляются достигнутым 5% gate. Один favourable endpoint — exploratory subgroup result.

**Table 2 / Figure 3:** unseen respondents × external tasks, режим label access, strong statistical baseline, effect size и test-retest reference. Два task families не лицензируют обобщение на всё человеческое поведение.

## 7. B4 — полезность: экономия данных и структура поведения

**Приоритет: MUST для слов «data-efficient / сокращает опрос»; иначе supporting, сужение C2.**

Разделить два бюджета, которые старые обсуждения иногда смешивали:

- `m`: наблюдаемые ответы именно тестируемого человека. Новый основной набор **{0,5,10,20,40,60}**; в IPIP-300 опционально 120. Для каждого m profile строится заново только из этих ответов. Missing facets помечаются unknown, не восстанавливаются из скрытой полной анкеты.
- `N`: число размеченных training humans на target task, например **{25,100,400,1000}** на Twin при наличии такого train pool. Cal/dev labels считаются отдельно в total supervision; N=25 + 200 cal нельзя рекламировать как «25 labels total». Строгий total-label-budget анализ делит доступный N_total между train/tune без extra labels.

Чтобы не строить дорогое декартово произведение, основной срез — все m при одном frozen N_total и все N_total при m=20. Только две модели, один выбранный method и best simple/strong statistical comparator; остальные arms здесь не нужны.

**Savings metric:** минимальное m, при котором верхняя граница loss метода достигает заранее выбранного reference loss/no-harm margin. Reference фиксируется на development, например strongest decoder at m=60; если crossing отсутствует, savings не определена. Дополнительно m*(simulator) как information-retention metric; эти оси нельзя взаимозаменять.

**Structure guardrails:** covariance/correlation structure по группам items, per-trait errors, subgroup calibration там, где достаточно людей; C2ST с отдельным style-only comparator только supporting. Marginal belief distributions не задают joint distribution: independent categorical sampling явно обозначается как product-of-marginals generator. Joint-fidelity claim требует генератора полных векторов и дополнительной проверки.

**Noninferiority:** proposed RPS degradation margin 0.002 absolute на заранее выбранных крупных subgroups; freeze до E. Для малых групп descriptive intervals, не «вреда нет» из отсутствия значимости. C2ST AUC около 0.5 не является доказательством эквивалентности без мощности.

**Figure 4:** качество–human answers–latency с uncertainty; единственная adoptable recommendation — в какой области затрат выбранный симулятор действительно лучше альтернативы.

## 8. Что запускать первым и что убрать с критического пути

| Milestone | Эксперименты | Stop/go | Место в статье |
|---|---|---|---|
| M0, 4–5 октября | B0: fresh snapshot уже сделан; затем correctness, exposure ledger, split freeze, два serving smoke | Нет чистого E/scorer → только development, никаких новых confirmatory claims | Protocol, reproducibility |
| M1, 5 октября | B1 selection 90 candidates → 30 global learned systems; common base/statistical comparators | Scope/comparator eligibility фиксируются до E; достаточный throughput | Table 1 |
| M2, 5–6 октября | B1 new E; B2 identity/objective/representation; независимый external adapter | Baseline выигрывает → не расширять дорогой поиск автоматически; сохраняем эффект и CI | Table 1, Fig. 1–2 |
| M3, 6–7 октября | B3-A descriptive transfer; B3-B внешняя оценка, Tsmall только по готовности | Независимый внешний результат нужен для сильного C2; новый full grid не приоритет | Table 2, Fig. 3 |
| M4, 7–8 октября | B4 subset/budget curves, full denominators, frozen tables | Один финальный read E; никаких новых ручных настройки/выбора по E | Fig. 4, limitations |
| M5, до 8 октября 18:00 МСК | Восьмистраничная история, repeatable tables, archive, проверка overlap | Внутренний target готовности, не обещание завершить вычисления | Submission packet |

Официальный full-paper deadline — **8 октября AoE = 9 октября 2026, 11:59:59 UTC = 14:59:59 МСК**. Источник: [AAMAS Q&A](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/qa/). При текущей дате окно порядка 4.7 суток. Оценки времени не служат основанием отменять научно валидный уже работающий эксперимент: он заканчивается в конечной allocation, а в submission идут только успевшие проверенные результаты.

**Первые три задания для исполнителя:** (1) CPU integrity+split receipt; (2) короткие serving/readout smoke на Qwen и Granite вне E; (3) frozen global selection с strongest control и baseline fit. GLM smoke 8311 — отдельный engineering task, не обязательный вход в научную серию на этих двух моделях.

Отложить с critical path: GLM full recovery, Qwen3.8 full grid, 4-GPU 235B, API-model zoo, NSGA-II/Pareto story, all-to-all model transfer, новый CARL/MAESTRO блок. Завершённые и неудачные старые runs остаются в ledger. Это расстановка приоритетов плана, не команда отменить чьи-либо Slurm jobs.

## 9. Статистика и правила принятия решений

1. **Единица uncertainty — человек**, не 60 независимых items и не 266 независимых репликаций. На общей E усреднить loss каждого человека по search seeds для main contrast. Resample человека один раз совместно для всех methods/seeds. Отдельно показать variation 3 search seeds; generalization over search stochasticity с n=3 ограничена. Дополнительные seeds 4–5 — только для зафиксированных method **и controls**, не winner alone.
2. Cross-task claim: paired respondent bootstrap плюс variation across task families/item blocks; при двух families показывать обе, не делать широкую random-task population claim.
3. CI: 5,000 paired respondent bootstrap replicates, fixed analysis seed. Для гипотез — valid paired randomization/permutation test на person-level differences с фиксированным числом permutations; знак/assumptions и family заранее задать. Не выдавать долю обычных uncentred bootstrap means за универсально корректный p-value.
4. C1 primary family: заранее frozen simple-method candidate (GA по простоте) vs {base, random, best-of-B}, две модели — 6 tests; Holm family-wise correction. MAP/GEPA и остальные contrasts — secondary BH с полным family denominator. Если нужен выбор winner по D_select, one-winner rule и его новая E проверка фиксируются до scoring. Не менять primary после E.
5. H1 causal objective test — прямой GA-RPS minus GA-S0. «Один significant, второй нет» не равняется значимой разности. Theory of propriety не заменяет empirical contrast.
6. H2 original family отчитывается полностью как development analysis. Критерий «лучше random и paraphrase в ≥50% cells» требует пересечения significant sets для тех же cells, с пропусками в denominator. Incomplete family не объявляется окончательно неуспешной.
7. H3 time/evaluations-to-threshold: правое цензурирование тех, кто не достиг цели к B; paired strata `(model, seed, data panel)`, restricted mean cost или success-at-budget. Не удалять неуспехи и не сравнивать только достигших.
8. H6: paired source prompts и одинаковые destination panels; raw loss contrasts primary. Gain retention ratio нестабилен при source gain≈0: заранее ε, undefined count, без удаления неудобных отрицательных результатов.
9. m*: random input subsets — одинаковые у всех методов; capacity retuned при каждом m/N; bootstrap по людям и subsets. Crossing вне измеренного диапазона — left/right censor, а не clip к максимуму. Не предполагать точную эмпирическую монотонность decoder curve; правило monotone fit фиксировать до результата и показывать raw curve рядом.
10. Power planning до E: paired SD σ_d на development, δ=0.001 RPS; базовая оценка n≈(z_(1−α/2)+z_(0.8))²σ_d²/δ² с multiplicity-adjusted α. Это приближение; проверить feasibility и dropout. Если n недостаточно — заранее уменьшить scope claims, не искать эффект по мере набора E.

## 10. Стоимость и исполнимая сетка

[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) — порядок и зависимости. [GRID.json](hbs_aamas_20261004/GRID.json) — **planning manifest**, не вход текущего queue.py: его нельзя случайно отправить как готовую сетку.

### 10.1 Frozen-artifact route — основная ставка до дедлайна

- D_select: 90 × 160 × 60 = **864,000** respondent–item forecasts.
- C_cal + E: 30 × (160+800) × 60 = **1,728,000** forecasts.
- Итого learned core **2,592,000** forecasts; static baselines/representations, objective controls и transfer считаются сверху. Cache повторяющихся base predictions; wrong-person controls — перестановки сохранённых deterministic predictions, без нового inference. Цена calibration cross-fitting вычислительная, не новые LLM calls.
- B2-OBJ: 9 S0 nominees × 160 × 60 + 3 selected × 960 × 60 = **259,200** дополнительных forecasts, если matched Qwen GA-RPS уже в B1.
- Static base/numeric/history: 6 × 960 × 60 = **345,600** forecasts. Numeric/history definitions фиксируются до scoring.
- Базовый B1+B2 readout order of magnitude **3.20 млн forecasts**, без внешнего Twin/IPIP-300. Это подсчёт запросов, не measured throughput и не число независимых наблюдений.

Из saved run metadata: elapsed медианы 1.594 h/cell Qwen и 3.227 h/cell Granite на старых 40-person opt. Они выполнялись параллельно и с разным размещением mutator; **это не GPU-hours**. Использовать их для обещания throughput нельзя.

Начальный planning envelope: **40–80 H200 GPU-hours** на frozen B1/B2 + limited transfer, **64–192 CPU core-hours** на baselines/statistics. Это допущение, а не измеренная смета. После non-test smoke: `hours = forecasts / measured forecasts_per_second / 3600 × GPU_count × 1.2`, отдельно startup, mutator и token lengths. Показать два сценария доступности: 1 GPU и 2 GPU; idle hardware не означает выделенный слот. Если envelope не выдерживается, сужать число secondary representations/transfer variants, сохраняя controls и новый test.

### 10.2 Дальняя / условная серия

| Расширение | Поисковых ячеек | Условие |
|---|---:|---|
| B1-FRESH: global search | 2 × 5 × 3 = 30 | Correctness и ресурсы подтверждены; не вытесняет внешний test |
| Дополнительные fitness S0 / raw RPS к GA | 2 × 2 × 3 = 12 | Нужна objective attribution; GA-cal уже в 30 |
| Official GEPA | 2 × 3 = 6 | Нужен published-optimizer superiority claim |
| MAP deletion studies | 2 × 3 × 3 = 18 | MAP выигрывает у GA на D_select, QD — заявляемый вклад |
| Twin Tsmall | 2 × 2 × 3 = 12 | Data adapter готов; это отдельный transfer-of-procedure track |

Не суммировать эти optional branches в обязательный заказ. Отдельный provisional cap **до 160 GPU-hours** на выбранное расширение после measured feasibility, без обещания уложить всю таблицу в эту сумму. Здесь план не запускает и не резервирует ресурсы.

**Human work:** новая разметка для core не предполагается; требуется проверка item mapping/allowed histories автором, сверка источников и claims. Existing self-reports — ground truth ответов, не клиническая валидность личности. Новое человеческое исследование не помещается на critical path и требует отдельного проекта.

## 11. Область новизны и публикационный пакет

| Работа / источник | Чем ограничивает claim | Наше проверяемое отличие |
|---|---|---|
| [GEPA](https://arxiv.org/abs/2507.19457), [official code](https://github.com/gepa-ai/gepa) | Reflection/evolution prompt search уже существует | Новизна не в изобретении reflective optimizer, а в conditional human-response objective, equal-budget evidence и переносе |
| [Park et al., revised 2026](https://arxiv.org/abs/2411.10109) | Self-report grounded prediction individuals на разных outcomes уже показана авторами | Нужны одинаковая information budget, сильный statistical baseline, uncertainty и цена наблюдений; «мы сделали persona agent» недостаточно |
| [SubPOP, ACL 2025](https://aclanthology.org/2025.acl-long.1028/) | Survey distribution prediction и unseen-survey adaptation уже исследованы | Individual identity signal и same-history forecast, а не только совпадение population marginals |
| [Twin-2K-500](https://arxiv.org/abs/2505.17479) | Доступна внешняя behavioral ground truth | Используем независимый тест и чёткие input exclusions, не повторяем dataset contribution |

Это targeted primary-source check, не исчерпывающая novelty certification. Непроверенные названия и версии из старых списков не становятся citations автоматически.

**Восемь страниц:** 1 — проблема и main result; 1 — метод/формула scoring; 1 — data/causal controls; 2 — core + baselines; 1 — mechanism; 1 — external task/budget utility; 1 — related work, limitations и выводы. Все load-bearing results внутри восьми страниц; technical appendix отдельно.

[Официальные instructions](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/) задают 8 pages + references, supplement до 25 MB и ограничение substantial overlap с параллельной archival submission. Поэтому Paper A и Paper B нельзя подавать как две версии одной таблицы с разными заголовками. Авторское решение об overlap и готовности научного вклада нужно принять до отправки, а не после.

Там же требуется раскрытие использования AI для hypotheses/methodology. Сохранить исходный запрос этой сессии, дату и фактически доступную model/version информацию; не угадывать отсутствующие сведения. Формы OpenReview, reviewer acceptance и публикация здесь не изменялись.

## 12. Решение о достаточности результата

**Сильная версия для AAMAS:** чистый B1, сильный same-budget контроль, identity gain, независимый external B3-B с utility против same-history comparator, проверенная область human-data savings или transfer efficiency. Это целевой evidence package, не гарантия принятия.

**Более узкая, честная версия:** clean B1 и external result относительно LLM controls, но strongest statistical model лучше. Тогда вклад ограничен улучшением LLM respondent simulators; не заявлять superiority/savings и оценить, достаточно ли это для основной статьи. Простое расширение model zoo недостаток не устраняет.

**Недостаточная версия:** только старые oracle-cluster gains, reanalysis уже exposed IPIP-300, либо положительная метрика без сильного comparator. Её нельзя обозначать как достигнутый уровень AAMAS. Сохраняем работающие артефакты и продолжаем независимый эксперимент после deadline вместо усиления формулировок.

### Checklist плана

- [x] Текущие 266 receipts проверены независимо от устаревшей сводки.
- [x] Предыдущие запросы Claude и ограничения BRIEF учтены.
- [x] Primary claims, baselines, data roles, gates, budget и run order заданы.
- [x] Старые и новые подтверждающие данные разделены.
- [x] Полезный положительный результат сформулирован как проверяемая цель.
- [ ] B0 implementation и независимый test manifest готовы.
- [ ] B1–B4 выполнены; заявляемые эффекты подтверждены.
- [ ] Авторы проверили scientific claims, overlap, форму подачи и disclosure.

Последние три пункта — работа следующей стадии, не скрытые незавершённые результаты этой planning-сессии.
