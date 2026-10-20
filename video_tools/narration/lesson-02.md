# Lesson 2: A network as a genome

## l2_title

Lesson two. The pilot is a neural network. But evolution never touches the network directly. It works on a genome, a recipe for building one.

## network

Here is the network that evolution found, flying. The eight inputs come in on the left. Each connection multiplies a value by its weight. Orange connections are positive, blue ones are negative, and thicker means stronger. Each node adds up what comes in, and squashes it into a value between minus one and one.

## network.2

On the right are four scores, one per action. The action with the highest score fires. Watch the scores change from moment to moment as the lander flies.

## network.3

This network has no hidden nodes at all, just seventeen of the thirty-two possible connections. Nobody drew it. Its shape and every one of its weights came out of evolution. We will see why it stayed this small in the next lesson.

## genome

The genome is a list of genes. Node genes say which neurons exist. Connection genes say which two neurons are wired together, with what weight, and whether the connection is switched on. Read the list from top to bottom, and you can build the network on the right.

## genome.2

Each connection gene also carries a historical marker, called its innovation number. It is handed out the first time that connection appears anywhere in the population, and every later copy keeps it. So if two genomes share a number, those genes have a common origin. This small idea is what makes crossover work, as we will see in lesson four. The library we use, neat-python, names each gene by the two nodes it connects, which does the same job.
