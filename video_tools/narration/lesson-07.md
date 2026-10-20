# Lesson 7: Reading the evolved network

## l7_title

Lesson seven. Evolution built a network that lands. But what did it actually build? Let's take it apart.

## read

One simple way to find out what matters is to switch a connection off, fly again, and see what breaks. Here is the winner flying, with all of its connections.

## read.2

The most important connection runs from the falling speed to the main engine, with a big negative weight. When the lander falls fast, its vertical speed is negative, so the main engine's score shoots up, and it brakes. Switch this one connection off, and the lander never lands at all.

## read.3

Next, the tilt angle feeds the left engine. This is the connection that keeps the lander upright. Without it, only five percent of flights land.

## read.4

And here is a surprise. Once the right leg touches the ground, its sensor tells the engines to stop, and votes for doing nothing. Without that, the lander keeps nudging itself and lands only fifteen percent of the time. Evolution ignored the left leg sensor completely. One leg was enough.

## ablation

Here are the connections that matter most, ranked by how much the landing rate drops without them. Red bars mean the lander falls apart. Green bars mean it barely notices.

## ablation.2

Nine of the seventeen connections can go with almost no loss. A handful of them carry the whole skill. That's typical for evolved networks. They are not tidy, but the important parts are usually few, and you can find them.

## failures

Switching connections off shows what the network relies on. Watching where it fails shows what it is missing. On a thousand new flights, it landed ninety-seven percent of the time. Sixteen of its eighteen crashes ended to the right of the pad, where it tipped over on the slope.

## failures.2

The other failures are stranger. Fourteen flights ran out of time, sitting on the pad. In eight of them, only the left leg was touching. Remember, the network ignores its left leg sensor. It waits for the right leg, and keeps nudging itself until the time runs out.

## failures.3

And every failure began with a hard sideways push. A network with no hidden nodes handles the usual flight very well. These rare, hard starts are where a better network could still improve.
