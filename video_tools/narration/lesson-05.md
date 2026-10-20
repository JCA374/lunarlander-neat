# Lesson 5: Speciation

## l5_title

Lesson five. Speciation is the part of NEAT that gives new ideas time to work.

## speciation

Here is the problem. A mutation that adds a node usually makes the network worse at first, because its new weights are not tuned yet. If every genome competed against every other one, that network would be thrown out right away, before it ever had a chance to improve.

## speciation.2

So NEAT measures how different two genomes are, using their genes. It counts the genes that don't match. It adds half of the weight differences of the genes that do match. And it divides the total by the size of the larger genome. Our library does this for the connections and for the nodes, and adds the two. If that distance is below two point two, the two genomes belong to the same species.

## speciation.3

Genomes then compete mostly inside their own species. Each species shares its offspring among its members, so a big species can't simply crowd out a small new one. And a species that stops improving for twenty generations is removed, so the population doesn't waste effort on dead ends.

## species

This is the classic NEAT picture, from our run. Each band is one species, and its height shows how many of the 150 genomes belong to it. At the very start there is just one species, because every genome has the same shape.

## species.2

As mutations pile up, new species split off. They grow when they do well, and shrink and die when they stop improving. Over the whole run, twenty-seven species came and went.
