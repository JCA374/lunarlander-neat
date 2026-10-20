# Lesson 4: Crossover

## l4_title

Lesson four. How do you combine two neural networks with different shapes? NEAT's answer is the innovation number.

## crossover

Here are two parents. Because every gene has an innovation number, we can line their genes up by number, even though the networks look different. Parent A is the fitter one.

## crossover.2

Genes that both parents have are called matching. A gene that one parent has in the middle of its list, where the other has a gap, is called disjoint. And a gene that comes after the other parent's last gene is called excess.

## crossover.3

The child takes each matching gene from either parent at random. The disjoint and excess genes come from the fitter parent. No matter how different the two parents look, the child is always a sensible network, because every gene keeps its place in history.

## loop

Crossover is one part of a generation. Let's put the whole loop together. First, evaluate: every one of the 150 genomes flies, and gets a fitness score.

## loop.2

Then, speciate: sort the genomes into species of similar networks. That's the topic of the next lesson.

## loop.3

Then, select: inside each species, the best twenty percent become parents, and the two very best are copied unchanged into the next generation.

## loop.4

Finally, reproduce: crossover and mutation fill the population back up to 150. Then the loop starts again, a few hundred times.
