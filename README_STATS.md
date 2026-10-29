# Статистика сравнения моделей

## Как это делают в статьях
1. **Не сравнивают все со всеми.** Заранее формулируют несколько вопросов (группы), в каждой - эталон и кандидаты. Число проверок
   фиксировано до просмотра результатов; поправка на множественность (Холм) применяется внутри группы.
2. **Отчёт = эффект + доверительный интервал**, а не только p. Вводят SESOI (наименьшая практически значимая разница): если весь 95% ДИ
   внутри ±SESOI - модели эквивалентны; значимо, но ≪ SESOI - «статистически есть, практически нет».
3. **Два источника случайности:** (а) обучение (сиды) - Welch t-тест / ДИ по сидам; (б) тестовая выборка - бутстрап по молекулам
   (единица = молекула, не атом: атомы одной молекулы зависимы). Сиды отвечают «воспроизведётся ли при переобучении», молекулы - «не
   случайность ли этого теста». Показывают оба; при 3 сидах тест по сидам имеет мало мощности, ДИ широкий - это честно.
4. **Сравнивают при равных условиях:** размер, расписание, число эпох, одинаковые данные. Всё, что нарушает, помечают в примечании группы.
5. Ранговые/композитные баллы (physical_score_and_stats) - только как описательная сводка, не как доказательство.

## Группы (configs/stats/groups_ablation.yaml)
Q1 xTB vs без xTB; Q2 enhanced vs без при равном размере; Q3 ёмкость (0.5x/1x/2x/4x); Q4 голова/loss при том же backbone;
Q5 варианты sigma-головы; Q6 backbone DimeNet++ vs PaiNN; Q7 все против лучшей. Редактируйте yaml под свои вопросы.

## Порядок запуска (из корня проекта)
```
python -m scripts.stats_compare --dirs results/all_pred_ablation_experiments_results/metrics results/xtb/metrics --list
python -m scripts.stats_compare --dirs <те же> --groups configs/stats/groups_ablation.yaml --out-csv results/stats_seeds.csv
# молекулярный уровень (GPU, чекпоинты нужны):
python -m scripts.molecule_eval_batch --dirs results/all_pred_ablation_experiments_results results/xtb --out-dir results/per_mol_all \
       --only dimenet_pp_enhanced dimenet_pp_enhanced_xtb sigma_dn_geom dimenet_pp_delta --seeds 0 1 2
python -m scripts.stats_compare --dirs <те же> --groups configs/stats/groups_ablation.yaml --mol-dir results/per_mol_all --out-csv results/stats_all.csv
```
Тест по сидам требует >= 2 сидов у обеих моделей. Для xtb-модели запустите сиды 1 и 2.
