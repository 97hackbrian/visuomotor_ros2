# Raw recording diagnostics: `demonstrations_real_v8`

* Per-frame timestamps recorded: yes.
* Null poses (RTAB-Map tracking loss stored as identity orientation and -initial position): 0 frames in 0 episodes (none).
* Duplicate consecutive samples: 3418 states, 0 images.
* Jumps > 30 mm between frames: 709; persistent discontinuities > 20 mm: 652 in 96 episodes (median 46.4 mm, p90 85.0 mm).
* done labels: 0/252 episodes have only the 40-frame floor; 18.6% of done frames move faster than 2 cm/s (filtered); 52 episodes with > 25% moving done frames.

![Raw diagnostics](fig4_raw_recording_diagnostics.png)
