**Project:** Real-Time Alignment & Tracking

**Background**: In E207 we learned about DTW, which estimates the optimal alignment between two sequences in an offline setting.  DTW is the primary tool for computing the similarity between time series data.  In some applications we want to estimate the alignment between two sequences in real-time.  For example, one application we are interested in is automated piano concerto accompaniment, in which a pianist can play a piano concerto, and their cell phone can provide virtual orchestral accompaniment that adapts to their playing in real-time.  Assume that one has two synchronized recordings: a recording of the piano part only (P), and a recording of the orchestra part only (O).  As the user plays the piano part (Pquery, which is an audio stream), we would like to track where we are in the P recording, and then time-scale modify the O recording to match the current position and velocity.

**Goal**: The goal of this project is to develop an online alignment algorithm, characterize its failure modes, explore ways to improve it, and compare its performance to existing approaches.

**Approach**

* Milestone 1: Characterize the performance of several baseline systems: offline DTW, OLTW (an existing method), and OLTW-global (existing method) on the Mazurka alignment benchmark  
  * There are existing implementations of these methods  
  * Since our focus is on audio-audio alignment of piano, we will use the Mazurka benchmark.  This benchmark contains lots of recordings of 5 Chopin Mazurkas.  For each recording, the dataset contains ground truth beat timestamps.  The benchmark consists of aligning pairs of recordings (of the same Mazurka), estimating the alignment between each pair of recordings, and measuring the accuracy of the estimated alignment.  
* Milestone 2: develop an online alignment algorithm that combines a Kalman filter with a naive online alignment algorithm  
  * This is essentially subsequence DTW which is computed in a streaming fashion (row by row), but using a normalized cumulative path score where instead of considering total cumulative path score, we normalize by the length of the path in order to consider average score per unit length.  This m akes it possible to fairly compare paths of very different length.  
  * This will require you to learn the basics of a Kalman filter.  
* Milestone 3: Do error analysis on your proposed method, characterize the failure modes, and iterate to improve.  Ideally, find something that works better than the other existing methods.
