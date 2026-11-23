# Raw recording diagnostics: `demonstrations_real_v7`

* Per-frame timestamps recorded: no (the 20 Hz sampling cannot be verified).
* Null poses (RTAB-Map tracking loss stored as identity orientation and -initial position): 4 frames in 2 episodes (episode_23, episode_24).
* Duplicate consecutive samples: 4 states, 0 images.
* Jumps > 30 mm between frames: 926; persistent discontinuities > 20 mm: 784 in 71 episodes (median 32.7 mm, p90 54.9 mm).
* done labels: 72/83 episodes have only the 40-frame floor; 21.0% of done frames move faster than 2 cm/s (filtered); 26 episodes with > 25% moving done frames.

![Raw diagnostics](fig4_raw_recording_diagnostics.png)
