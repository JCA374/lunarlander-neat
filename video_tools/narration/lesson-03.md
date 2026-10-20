# Lesson 3: Mutation grows structure

## l3_title

Lesson three. NEAT starts every network as small as possible, and lets mutation make it bigger, one careful step at a time.

## mutation

Each new genome starts as a copy of a parent, and then it mutates. There are three kinds of mutation.

## mutation.2

The most common one simply nudges a weight, or the bias of a node. In our setup, each weight has an eighty percent chance of a small nudge. Here, a positive connection drifts until it becomes negative.

## mutation.3

The second kind adds a connection, joining two nodes that were not connected before. It gets a new innovation number. Half of all new genomes get one of these.

## mutation.4

The third kind adds a node. It picks a connection and splits it in two. The old gene is switched off, and two new genes take its place. The connection into the new node gets a weight of one, and the one out keeps the old weight. So the network behaves almost the same as before. The structure has grown, without breaking what already worked. Connections and nodes can also be deleted, so networks can shrink too.

## growth

So what happened in the real run? Here is the champion of each generation. Every run starts fully connected, with all eight inputs wired to all four outputs, and no hidden nodes.

## growth.2

For its first hundred and sixty generations, evolution mostly removed connections. The winner, from generation one hundred and sixty-four, has no hidden nodes at all, and just seventeen of the thirty-two connections it started with. Later, the population grew much bigger networks, with fifteen or more hidden nodes. But those flew worse on unseen flights, not better. A bigger network is not automatically a better one, and that is exactly why NEAT starts small.
