We have Finished organizing the code for generating the syndrome empirical distance spectra for both specific keys and random keys.
- Modify **Lines 625 and 626** to set the block length for Type 1 weak keys and the clustering interval *m* for m-gather weak keys.
- Adjust **Line 673 or 674** to define the key type of $h_0$ and $h_1$: both can be set as random keys, or $h_i$ as a Type 1 weak key, or $h_i$ as a m-gather key, i=0 or 1.
- **Lines 630 and 631** correspond to the parameters `num` and `main_num` respectively:
  - `num`: The number of error samples per parity-check matrix $H$ (used for plotting a single distance spectrum).
  - `main_num`: The total number of experiments (i.e., how many distance spectra are generated for different keys).

The rest will be available soon...

Pipeline subsequent updates...
