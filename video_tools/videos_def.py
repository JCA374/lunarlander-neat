"""
Scene lists for the explainer and the lessons. Each video's narration lives in
video_tools/narration/<slug>.md (one `## <scene>[.n]` section per beat) and is voiced with
the voice-text pipeline into voice/<slug>/.
"""

import scenes as S
from common import title_card

LESSONS = [
    ('lesson-01', 'The task', 'what the lander senses, does, and is scored on'),
    ('lesson-02', 'A network as a genome', 'the recipe evolution works on'),
    ('lesson-03', 'Mutation grows structure', 'new weights, connections and nodes'),
    ('lesson-04', 'Crossover', 'innovation numbers line genes up'),
    ('lesson-05', 'Speciation', 'protecting new ideas long enough to work'),
    ('lesson-06', 'Watching it evolve', 'the real run, generation by generation'),
    ('lesson-07', 'Reading the evolved network', 'what evolution actually built'),
]


def explainer(ctx):
    return [
        S.scene_title(ctx), S.scene_task(ctx), S.scene_reward(ctx), S.scene_network(ctx),
        S.scene_genome(ctx), S.scene_mutation(ctx), S.scene_crossover(ctx), S.scene_speciation(ctx),
        S.scene_loop(ctx), S.scene_fitness(ctx), S.scene_gen_fitness(ctx), S.scene_species(ctx),
        S.scene_gen_ghosts(ctx), S.scene_growth(ctx), S.scene_evolution(ctx), S.scene_read(ctx),
        S.scene_ablation(ctx), S.scene_outro(ctx),
    ]


def lesson(ctx, n):
    slug, title, sub = LESSONS[n - 1]
    card = title_card(f'l{n}_title', f'Lesson {n}', title, sub)
    body = {
        1: lambda: [S.scene_task(ctx), S.scene_reward(ctx)],
        2: lambda: [S.scene_network(ctx), S.scene_genome(ctx)],
        3: lambda: [S.scene_mutation(ctx), S.scene_growth(ctx)],
        4: lambda: [S.scene_crossover(ctx), S.scene_loop(ctx)],
        5: lambda: [S.scene_speciation(ctx), S.scene_species(ctx)],
        6: lambda: [S.scene_fitness(ctx), S.scene_gen_fitness(ctx), S.scene_gen_ghosts(ctx),
                    S.scene_evolution(ctx)],
        7: lambda: [S.scene_read(ctx), S.scene_ablation(ctx), S.scene_failures(ctx)],
    }[n]()
    return slug, [card] + body
