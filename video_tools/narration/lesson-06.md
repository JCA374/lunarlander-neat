# Lesson 6: Watching it evolve

## l6_title

Lesson six. Now we watch the real training run, one generation at a time.

## fitness

First, how do we score a genome? Its fitness is the average reward over five flights.

## fitness.2

Every genome in a generation flies the same starting positions, so the comparison is fair. The starting positions change each generation, so no genome can memorise one.

## fitness.3

But five flights are noisy, and the top genome is often just lucky. So each generation, its champion also flies a hundred flights that no genome ever trained on. We only call the problem solved when that unseen average reaches two hundred.

## gen_fitness

Every dot is one genome, and its color shows its species. In generation zero, every network is just the inputs wired straight to the outputs, with random weights. Almost all of them crash.

## gen_fitness.2

Generation by generation, the cloud of dots creeps upward. Notice how wide it stays. Mutation keeps producing bad genomes too, and that variety is what evolution searches through.

## gen_fitness.3

The green line is the champion on the hundred unseen flights. It first passes two hundred in generation thirty-seven. The best champion came from generation one hundred and sixty-four. On another hundred fresh flights, it lands ninety-five percent of the time. Late in the run the line sags, as the networks grow bigger without getting better.

## gen_ghosts

Here is one whole generation flying together. All 150 genomes start from the same spot, and each flight is drawn as a trail. This is generation zero.

## gen_ghosts.2

At first, nearly every flight ends in a crash, or drifts off the screen.

## gen_ghosts.3

Fifty generations in, a few dozen trails curve down onto the pad.

## gen_ghosts.4

And late in the run, about a third of the population lands from this start, and their trails bundle together into one shared path. The rest still crash. Mutation keeps breaking things, and that is the price of searching.

## evolution

Finally, here are the champions from different generations, all flying the same start.

## evolution.2

The early ones crash or drift away. Later champions slow down, straighten up, and land between the flags.
