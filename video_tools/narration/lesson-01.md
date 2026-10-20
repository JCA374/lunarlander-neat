# Lesson 1: The task

## l1_title

Lesson one. Before we evolve anything, let's look at the problem: what the lander senses, what it can do, and how it is scored.

## task

This is Lunar Lander, from the Gymnasium library. The lander starts high above the surface, with a random push. The goal is to touch down softly between the two flags.

## task.2

At every step, the lander senses eight numbers. Its position, measured from the pad. Its horizontal and vertical speed. Its tilt, and how fast it is spinning. And two switches that say whether each leg touches the ground.

## task.3

And at every step, it must choose one of four actions. Do nothing, fire the left engine, fire the main engine, or fire the right engine. So whatever our network looks like, it has eight inputs and four outputs.

## reward

The game itself hands out points as it goes. Moving toward the pad, slowing down and staying level all earn points. Drifting away, speeding up and tilting cost them.

## reward.2

There are also one-off events. Each leg that touches the ground earns ten points. Coming to rest earns a hundred. A crash, or flying off the screen, costs a hundred.

## reward.3

And fuel isn't free. Every frame the main engine fires costs a little, and the side engines cost a tenth of that. So hovering forever is a losing strategy.

## reward.4

Add it all up over one flight, and you get the episode's reward. An average of two hundred or more counts as solved. That reward is exactly what we will use as fitness.
