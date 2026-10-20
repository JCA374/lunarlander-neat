# NEAT Lunar Lander explainer

## title

What if evolution could design a neural network for you? Not just its weights, but its whole shape. This is NEAT, learning to land on the Moon.

## task

This is Lunar Lander. The goal is to touch down softly between the two flags. Gravity pulls the lander down, and every burst of an engine costs fuel.

## task.2

At every step, the lander senses eight numbers. Where it is, how fast it is moving, how it is tilted and spinning, and whether each leg touches the ground.

## task.3

And at every step, it chooses one of four actions. Do nothing, fire the left engine, fire the main engine, or fire the right engine.

## reward

The game hands out points as it goes. Moving toward the pad, slowing down and staying level earn points. Drifting away, speeding up and tilting cost points.

## reward.2

There are also one-off events. Each leg that touches the ground earns ten. Coming to rest earns a hundred. A crash costs a hundred.

## reward.3

And fuel isn't free. Every frame the main engine fires costs a little.

## reward.4

An average of two hundred points counts as solved.

## network

The pilot is a small neural network. The eight inputs come in on the left. They flow along weighted connections, and come out on the right as four scores, one per action.

## network.2

The action with the highest score fires. Watch the scores change as the lander flies.

## network.3

This network has no hidden nodes at all, just seventeen of the thirty-two possible connections. Nobody designed it. Evolution found every weight, and decided which connections to keep.

## genome

Evolution doesn't work on the network directly. It works on a genome: a list of genes. Node genes say which neurons exist. Connection genes say which neurons are wired together, and with what weight. Read the list, and you can build the network.

## genome.2

Each connection gene also carries a historical marker, called its innovation number. It is given out the first time that connection appears anywhere in the population. So when two genomes share a number, those genes have the same origin. The library we use names each gene by the two nodes it connects, which does the same job.

## mutation

Each new genome starts as a copy of a parent, and then it mutates. There are three kinds of mutation.

## mutation.2

The most common one simply nudges a weight. Here, a positive connection becomes negative.

## mutation.3

The second kind adds a connection, joining two nodes that were not connected before. It gets a new innovation number.

## mutation.4

The third kind adds a node. It splits an existing connection in two. The old gene is switched off, and two new genes take its place. The connection into the new node gets a weight of one, and the one out keeps the old weight, so the network behaves almost the same as before. The structure has grown, without breaking anything.

## crossover

Two parents can also be combined. This is where the innovation numbers pay off. Line the genes up by their numbers, and you can see which genes the two parents share.

## crossover.2

Genes that both parents have are called matching. A gene missing in the middle of the other parent is called disjoint. And a gene beyond the end of the other parent is called excess.

## crossover.3

The child takes each matching gene from either parent at random. The disjoint and excess genes come from the fitter parent. No matter how different the two networks look, the child is always a sensible network.

## speciation

Here is a problem. A mutation that adds a node usually makes the network worse at first, because its new weights are not tuned yet. Ranked against everyone, it would be thrown out before it could ever become useful.

## speciation.2

So NEAT measures how different two genomes are. It counts the genes that don't match, adds the weight differences of the genes that do, and divides by the size of the genome. If that distance is small enough, the two genomes belong to the same species.

## speciation.3

Genomes then compete mostly inside their own species. Each species shares its offspring among its members, so a big species can't crowd out a small new one. And a species that stops improving for twenty generations dies out.

## loop

So here is one generation of NEAT. First, evaluate: every one of the 150 genomes flies, and gets a fitness score.

## loop.2

Then, speciate: sort the genomes into species.

## loop.3

Then, select: inside each species, the best ones become parents, and the two very best are kept unchanged.

## loop.4

And reproduce: crossover and mutation fill the population back up. Then it all starts again.

## fitness

Fitness is the average reward over five flights.

## fitness.2

Every genome in a generation flies the same starting positions, so the comparison is fair. The starting positions change each generation, so no genome can memorise one.

## fitness.3

But five flights are noisy, and the top genome is often just lucky. So each generation, its champion also flies a hundred flights that no genome ever trained on. We only call the problem solved when that unseen average reaches two hundred.

## gen_fitness

Now the real run. Every dot is one genome, and its color shows its species. In generation zero, every network is just the inputs wired straight to the outputs, with random weights. Almost all of them crash.

## gen_fitness.2

Generation by generation, the cloud of dots creeps upward. New species appear in new colors.

## gen_fitness.3

The green line is the champion on the hundred unseen flights. It first passes two hundred in generation thirty-seven. The best champion of all came from generation one hundred and sixty-four.

## species

This is the classic NEAT picture. Each band is one species, and its height is how many genomes it has. At the start, there is just one.

## species.2

New species split off, grow when they do well, and shrink and die when they stop improving. Over the whole run, twenty-seven species came and went.

## gen_ghosts

Here is one generation flying together. All 150 genomes start from the same spot, and we draw each flight as a trail. This is generation zero.

## gen_ghosts.2

At first, nearly every flight ends in a crash.

## gen_ghosts.3

Fifty generations in, a few dozen trails curve down onto the pad.

## gen_ghosts.4

And late in the run, about a third of the population lands from this start, and their trails bundle together into one shared path. The rest still crash. Mutation keeps breaking things, and that is the price of searching.

## growth

This is how the champion's network changed. It started fully connected, with all eight inputs wired to all four outputs, and no hidden nodes.

## growth.2

For its first hundred and sixty generations, evolution mostly removed connections. The winner, from generation one hundred and sixty-four, has no hidden nodes at all, and just seventeen of the thirty-two connections it started with. Later, the population grew much bigger networks, with fifteen or more hidden nodes. But those flew worse on unseen flights, not better. A bigger network is not automatically a better one, and that is exactly why NEAT starts small.

## evolution

Here are the champions from different generations, all flying the same start.

## evolution.2

The early ones crash or drift away. Later champions slow down, straighten up, and land between the flags.

## read

So what did evolution build? Let's switch connections off one at a time, and see which ones matter.

## read.2

The most important connection runs from the falling speed to the main engine, with a big negative weight. When the lander falls fast, its vertical speed is negative, so the main engine's score shoots up, and it brakes. Switch this one connection off, and the lander never lands at all.

## read.3

Next, the tilt angle feeds the left engine. This is the connection that keeps the lander upright. Without it, only five percent of flights land.

## read.4

And here is a surprise. Once the right leg touches the ground, its sensor tells the engines to stop, and votes for doing nothing. Without that, the lander keeps nudging itself and lands only fifteen percent of the time. Evolution ignored the left leg sensor completely. One leg was enough.

## ablation

Here are the connections that matter most, ranked by how much the landing rate drops without them.

## ablation.2

Nine of the seventeen connections can go with almost no loss. A handful of them carry the whole skill.

## outro

On a hundred new flights, the winner lands ninety-five percent of the time, with an average reward of two hundred and fifty-four. And it does that with no hidden nodes, and seventeen connections.

## outro.2

The code, the training run, and these videos are all on GitHub. Clone it, train your own, and watch your network grow.
