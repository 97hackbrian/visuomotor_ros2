# Raw recording diagnostics: `demonstrations_real_v8_droid`

* Per-frame timestamps recorded: yes.
* Null poses (RTAB-Map tracking loss stored as identity orientation and -initial position): 0 frames in 0 episodes (none).
* Duplicate consecutive samples: 485 states, 0 images.
* Jumps > 30 mm between frames: 0; persistent discontinuities > 20 mm: 0 in 0 episodes.
* done labels: 2/230 episodes have only the 40-frame floor; 6.5% of done frames move faster than 2 cm/s (filtered); 37 episodes with > 25% moving done frames.

![Raw diagnostics](fig4_raw_recording_diagnostics.png)
